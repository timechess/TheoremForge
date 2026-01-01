from theoremforge.agents.base_agent import BaseAgent
from theoremforge.state import TheoremForgeContext, TheoremForgeState
from openai import AsyncOpenAI
from google.genai import Client
import re
from typing import List
from theoremforge.utils import format_search_results, call_llm_interruptible
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
        if model_name.startswith("gemini-"):
            self.client = Client(api_key=api_key, http_options={"base_url": base_url}, vertexai=True)
        else:
            self.client = AsyncOpenAI(base_url=base_url, api_key=api_key, timeout=1500)
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

    async def _run(self, state: TheoremForgeState):
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
        logger.debug(
            f"Theorem Retrieval Agent: Extracted {len(queries)} search queries for state {state.id}"
        )
        
        # Check for cancellation before expensive retrieval operation
        if await self.is_cancelled(state):
            logger.debug(
                f"Theorem Retrieval Agent: State {state.id} cancelled before retrieval"
            )
            await self.add_state_request("finish_agent", state)
            return
        
        # Use async search to avoid blocking the event loop
        results = await self.retriever.search_async(queries, 5)
        query_results = [format_search_results(result) for result in results]
        theorem_selection_prompt = prompt_manager.theorem_selection(
            state.formal_statement, query_results
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
            theorem
            for theorem in results
            if theorem.primary_declaration.lean_name in theorem_names
        ]
        logger.debug(
            f"Theorem Retrieval Agent: Selected {len(selected_theorems)} theorems for state {state.id}"
        )
        theorem_selection_results = [
            format_search_results(theorem) for theorem in selected_theorems
        ]

        state.metadata["useful_theorems"] = "\n".join(theorem_selection_results)
        await self.add_state_request("informal_proof_agent", state)
