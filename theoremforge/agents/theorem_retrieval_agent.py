from theoremforge.agents.base_agent import BaseAgent
from theoremforge.state import TheoremForgeContext
from openai import AsyncOpenAI
import re
from typing import List
import aiohttp
from theoremforge.utils import payload_to_string
from theoremforge.prompt_manager import prompt_manager
from loguru import logger


class TheoremRetrievalAgent(BaseAgent):
    def __init__(
        self,
        context: TheoremForgeContext,
        base_url: str,
        api_key: str,
        model_name: str,
        retriever_url: str,
        sampling_params: dict,
    ):
        super().__init__(
            agent_name="theorem_retrieval_agent",
            context=context,
        )
        self.client = AsyncOpenAI(base_url=base_url, api_key=api_key)
        self.model_name = model_name
        self.retriever_url = retriever_url
        self.session = aiohttp.ClientSession()
        self.sampling_params = sampling_params

    def _extract_search_queries(self, text: str) -> List[str]:
        pattern = r"<search>(.*?)</search>"
        matches = re.findall(pattern, text, re.DOTALL)
        return [match.strip() for match in matches][:5]

    def _extract_theorems(self, text: str) -> List[str]:
        pattern = r"<theorem>(.*?)</theorem>"
        matches = re.findall(pattern, text, re.DOTALL)
        return [match.strip() for match in matches]

    async def run(self):
        while True:
            try:
                state = await self.task_queue.get()
                logger.info(
                    f"Theorem Retrieval Agent: Start to process state {state.id}"
                )

                # Check black_list with lock
                async with self.context.black_list_lock:
                    is_blacklisted = state.id in self.context.black_list

                if is_blacklisted:
                    logger.debug(f"Theorem Retrieval Agent: State {state.id} is blacklisted, routing to finish")
                    await self.add_state_request("finish_agent", state)
                    continue

                query_generation_prompt = prompt_manager.search_query_generation(
                    state.formal_statement
                )

                response = await self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "user", "content": query_generation_prompt},
                    ],
                    **self.sampling_params,
                )
                query_generation_output = response.choices[0].message.content
                queries = self._extract_search_queries(query_generation_output)
                logger.info(
                    f"Theorem Retrieval Agent: Extracted {len(queries)} search queries for state {state.id}"
                )
                async with self.session.post(
                    f"{self.retriever_url}/search", json={"queries": queries, "topk": 5}
                ) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        raise Exception(f"Server error: {error_text}")
                    result = await response.json()
                query_results = sum(result["results"], [])
                theorems = "\n".join(
                    [payload_to_string(result["payload"]) for result in query_results]
                )
                theorem_selection_prompt = prompt_manager.theorem_selection(
                    state.formal_statement, theorems
                )
                response = await self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "user", "content": theorem_selection_prompt},
                    ],
                    **self.sampling_params,
                )
                theorem_selection_output = response.choices[0].message.content
                theorem_names = self._extract_theorems(theorem_selection_output)
                selected_theorems = [
                    theorem["payload"]
                    for theorem in query_results
                    if theorem["payload"]["name"] in theorem_names
                ]
                logger.info(
                    f"Theorem Retrieval Agent: Selected {len(selected_theorems)} theorems for state {state.id}"
                )
                theorem_selection_results = [
                    payload_to_string(theorem) for theorem in selected_theorems
                ]

                await self.db.theoremretrievaltrace.create(
                    data={
                        "queryGenerationPrompt": query_generation_prompt,
                        "queryGenerationOutput": query_generation_output,
                        "queryResults": [
                            payload_to_string(result["payload"])
                            for result in query_results
                        ],
                        "theoremSelectionPrompt": theorem_selection_prompt,
                        "theoremSelectionOutput": theorem_selection_output,
                        "theoremSelectionResults": theorem_selection_results,
                        "stateId": state.id,
                    }
                )
                state.metadata["useful_theorems"] = "\n".join(theorem_selection_results)
                await self.add_state_request("informal_proof_agent", state)
            except Exception as e:
                logger.error(f"Theorem Retrieval Agent: Error processing state: {e}")
                import traceback

                traceback.print_exc()
                # Try to route to finish even on error
                try:
                    if "state" in locals():
                        await self.add_state_request("finish_agent", state)
                except Exception:
                    pass
