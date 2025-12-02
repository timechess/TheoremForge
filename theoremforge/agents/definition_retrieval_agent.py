from theoremforge.agents.base_agent import BaseAgent
from theoremforge.state import TheoremForgeContext
from openai import AsyncOpenAI
from theoremforge.retriever import Retriever
import re
from typing import List
from loguru import logger

from theoremforge.prompt_manager import prompt_manager
from theoremforge.utils import call_llm_interruptible, CancellationError, payload_to_string


class DefinitionRetrievalAgent(BaseAgent):
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
            agent_name="definition_retrieval_agent",
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

    def _extract_definitions(self, text: str) -> List[str]:
        pattern = r"<definition>(.*?)</definition>"
        matches = re.findall(pattern, text, re.DOTALL)
        return [match.strip() for match in matches]

    async def run(self):
        while True:
            try:
                state = await self.task_queue.get()
                logger.info(
                    f"Definition Retrieval Agent: Start to process state {state.id}"
                )

                # Register cancellation event for this state
                await self.register_cancellation_event(state)

                # Check if state should be skipped (blacklisted or cancelled)
                if await self.should_skip_state(state):
                    await self.add_state_request("finish_agent", state)
                    await self.cleanup_cancellation_event(state)
                    continue

                query_generation_prompt = prompt_manager.definition_query_generation(
                    state.normalized_statement
                )

                response = await call_llm_interruptible(
                    state,
                    self.context,
                    self.client,
                    self.model_name,
                    query_generation_prompt,
                    self.sampling_params,
                    "definition_retrieval_agent",
                )

                query_generation_output = response[0]
                queries = self._extract_search_queries(query_generation_output)
                logger.info(
                    f"Definition Retrieval Agent: Extracted {len(queries)} search queries for state {state.id}"
                )
                
                # Check for cancellation before expensive retrieval operation
                if await self.is_cancelled(state):
                    logger.info(
                        f"Definition Retrieval Agent: State {state.id} cancelled before retrieval"
                    )
                    await self.add_state_request("finish_agent", state)
                    await self.cleanup_cancellation_event(state)
                    continue
                
                result = await self.retriever.search_definitions(queries, 5)
                query_results = sum(result["results"], [])
                definitions = "\n".join(
                    [payload_to_string(result["payload"]) for result in query_results]
                )
                definition_selection_prompt = prompt_manager.definition_selection(
                    state.normalized_statement, definitions
                )
                response = await call_llm_interruptible(
                    state,
                    self.context,
                    self.client,
                    self.model_name,
                    definition_selection_prompt,
                    self.sampling_params,
                    "definition_retrieval_agent",
                )
                definition_selection_output = response[0]
                definition_names = self._extract_definitions(
                    definition_selection_output
                )
                selected_definitions = [
                    definition["payload"]
                    for definition in query_results
                    if definition["payload"]["name"] in definition_names
                ]
                logger.info(
                    f"Definition Retrieval Agent: Selected {len(selected_definitions)} definitions for state {state.id}"
                )
                definition_selection_results = [
                    payload_to_string(definition) for definition in selected_definitions
                ]
                await self.db.definitionretrievaltrace.create(
                    data={
                        "queryGenerationPrompt": query_generation_prompt,
                        "queryGenerationOutput": query_generation_output,
                        "queryResults": [
                            payload_to_string(result["payload"])
                            for result in query_results
                        ],
                        "definitionSelectionPrompt": definition_selection_prompt,
                        "definitionSelectionOutput": definition_selection_output,
                        "definitionSelectionResults": definition_selection_results,
                        "stateId": state.id,
                    }
                )
                state.metadata["useful_definitions"] = "\n".join(
                    definition_selection_results
                )
                await self.add_state_request("autoformalization_agent", state)
                
                # Cleanup cancellation event after routing
                await self.cleanup_cancellation_event(state)
            except CancellationError as e:
                # State was cancelled during processing
                logger.info(f"Definition Retrieval Agent: {e}")
                if "state" in locals():
                    await self.add_state_request("finish_agent", state)
                    await self.cleanup_cancellation_event(state)
            except Exception as e:
                logger.error(f"Definition Retrieval Agent: Error processing state: {e}")
                import traceback
                traceback.print_exc()
                try:
                    if "state" in locals():
                        await self.add_state_request("finish_agent", state)
                        await self.cleanup_cancellation_event(state)
                except Exception:
                    pass

