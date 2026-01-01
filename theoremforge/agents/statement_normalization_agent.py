from theoremforge.agents.base_agent import BaseAgent
from theoremforge.state import TheoremForgeContext, TheoremForgeState
from openai import AsyncOpenAI
from google.genai import Client
from loguru import logger
from theoremforge.prompt_manager import prompt_manager
from theoremforge.utils import call_llm_interruptible
import re


class StatementNormalizationAgent(BaseAgent):
    def __init__(
        self,
        context: TheoremForgeContext,
        base_url: str,
        api_key: str,
        model_name: str,
        sampling_params: dict,
    ):
        super().__init__(agent_name="statement_normalization_agent", context=context)
        if model_name.startswith("gemini-"):
            self.client = Client(api_key=api_key, http_options={"base_url": base_url}, vertexai=True)
        else:
            self.client = AsyncOpenAI(base_url=base_url, api_key=api_key, timeout=1500)
        self.model_name = model_name
        self.sampling_params = sampling_params

    def _extract_statement_normalization(self, response: str) -> str:
        pattern = r"<normalized>(.*?)</normalized>"
        match = re.search(pattern, response, re.DOTALL)
        return match.group(1).strip() if match else None

    async def _run(self, state: TheoremForgeState):
        logger.debug(f"Statement Normalization Agent: Start to process state {state.id}")

        if not state.informal_statement:
            logger.error(
                f"Statement Normalization Agent: No informal statement for state {state.id}"
            )
            raise ValueError(
                f"Statement Normalization Agent: No informal statement for state {state.id}"
            )

        statement_normalization_prompt = prompt_manager.statement_normalization(
            state.informal_statement
        )
        statement_normalization_output = (
            await call_llm_interruptible(
                state,
                self.context,
                self.client,
                self.model_name,
                statement_normalization_prompt,
                self.sampling_params,
                "statement_normalization_agent",
            )
        )[0]
        normalized_statement = self._extract_statement_normalization(
            statement_normalization_output
        )
        if not normalized_statement:
            logger.warning(
                f"Statement Normalization Agent: Failed to normalize statement {state.id}"
            )
            state.normalized_statement = state.informal_statement
        else:
            logger.debug(f"Statement Normalization Agent: Normalized statement {state.id}")
            state.normalized_statement = normalized_statement
        logger.debug(
            f"Statement Normalization Agent: Routing state {state.id} to definition_retrieval_agent"
        )
        await self.add_state_request("definition_retrieval_agent", state)
