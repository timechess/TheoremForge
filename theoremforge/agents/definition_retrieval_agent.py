from theoremforge.agents.base_agent import BaseAgent
from theoremforge.state import TheoremForgeContext, TheoremForgeState
from openai import AsyncOpenAI
from google.genai import Client
from theoremforge.retriever import Retriever
import re
from typing import List
from loguru import logger

from theoremforge.prompt_manager import prompt_manager
from theoremforge.utils import call_llm_interruptible, format_search_results


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
        if model_name.startswith("gemini-"):
            self.client = Client(
                api_key=api_key, http_options={"base_url": base_url}, vertexai=True
            )
        else:
            self.client = AsyncOpenAI(base_url=base_url, api_key=api_key, timeout=1500)
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

    async def _run(self, state: TheoremForgeState):
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

        # Use async search to avoid blocking the event loop
        results = await self.retriever.search_async(queries, 5)
        definitions = [format_search_results(result) for result in results]
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
        definition_names = self._extract_definitions(definition_selection_output)
        selected_definitions = [
            definition
            for definition in results
            if definition.primary_declaration.lean_name in definition_names
        ]
        logger.info(
            f"Definition Retrieval Agent: Selected {len(selected_definitions)} definitions for state {state.id}"
        )
        definition_selection_results = [
            format_search_results(definition) for definition in selected_definitions
        ]

        state.metadata["useful_definitions"] = "\n".join(definition_selection_results)
        await self.add_state_request("autoformalization_agent", state)
