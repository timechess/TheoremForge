from theoremforge.agents.base_agent import BaseAgent
from theoremforge.state import TheoremForgeContext
from openai import AsyncOpenAI
import re
from typing import List
from theoremforge.utils import payload_to_string, call_llm_interruptible, CancellationError
from theoremforge.prompt_manager import prompt_manager
from loguru import logger
from theoremforge.retriever import Retriever


class TheoremRetrievalAgent(BaseAgent):
    def __init__(
        self,
        context: TheoremForgeContext,
        base_url: str,
        api_key: str,
        model_name: str,
        retriever: Retriever,
        sampling_params: dict,
    ):
        super().__init__(
            agent_name="theorem_retrieval_agent",
            context=context,
        )
        self.client = AsyncOpenAI(base_url=base_url, api_key=api_key)
        self.model_name = model_name
        self.retriever = retriever
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

                # Register cancellation event for this state
                await self.register_cancellation_event(state)

                # Check if state should be skipped (blacklisted or cancelled)
                if await self.should_skip_state(state):
                    await self.add_state_request("finish_agent", state)
                    await self.cleanup_cancellation_event(state)
                    continue

                query_generation_prompt = prompt_manager.theorem_query_generation(
                    state.formal_statement
                )

                response = await call_llm_interruptible(
                    state,
                    self.context,
                    self.client,
                    self.model_name,
                    query_generation_prompt,
                    self.sampling_params,
                    "theorem_retrieval_agent",
                )
                query_generation_output = response[0]
                queries = self._extract_search_queries(query_generation_output)
                logger.info(
                    f"Theorem Retrieval Agent: Extracted {len(queries)} search queries for state {state.id}"
                )
                
                # Check for cancellation before expensive retrieval operation
                if await self.is_cancelled(state):
                    logger.info(
                        f"Theorem Retrieval Agent: State {state.id} cancelled before retrieval"
                    )
                    await self.add_state_request("finish_agent", state)
                    await self.cleanup_cancellation_event(state)
                    continue
                
                result = await self.retriever.search_theorems(queries, 5)
                query_results = sum(result["results"], [])
                theorems = "\n".join(
                    [payload_to_string(result["payload"]) for result in query_results]
                )
                theorem_selection_prompt = prompt_manager.theorem_selection(
                    state.formal_statement, theorems
                )
                response = await call_llm_interruptible(
                    state,
                    self.context,
                    self.client,
                    self.model_name,
                    theorem_selection_prompt,
                    self.sampling_params,
                    "theorem_retrieval_agent",
                )
                theorem_selection_output = response[0]
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
                
                # Cleanup cancellation event after routing
                await self.cleanup_cancellation_event(state)
            except CancellationError as e:
                # State was cancelled during processing
                logger.info(f"Theorem Retrieval Agent: {e}")
                if "state" in locals():
                    await self.add_state_request("finish_agent", state)
                    await self.cleanup_cancellation_event(state)
            except Exception as e:
                logger.error(f"Theorem Retrieval Agent: Error processing state: {e}")
                import traceback
                traceback.print_exc()
                # Try to route to finish even on error
                try:
                    if "state" in locals():
                        await self.add_state_request("finish_agent", state)
                        await self.cleanup_cancellation_event(state)
                except Exception:
                    pass
