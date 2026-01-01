from loguru import logger
from theoremforge.agents.base_agent import BaseAgent
from theoremforge.state import TheoremForgeContext, TheoremForgeState
from openai import AsyncOpenAI
from google.genai import Client
from theoremforge.utils import (
    extract_lean_code,
    call_llm_interruptible,
    statement_check,
)
from theoremforge.prompt_manager import prompt_manager


class SketchCorrectionAgent(BaseAgent):
    def __init__(
        self,
        context: TheoremForgeContext,
        base_url: str,
        api_key: str,
        model_name: str,
        sampling_params: dict,
    ):
        super().__init__(
            agent_name="sketch_correction_agent",
            context=context,
        )
        if model_name.startswith("gemini-"):
            self.client = Client(api_key=api_key, http_options={"base_url": base_url}, vertexai=True)
        else:
            self.client = AsyncOpenAI(base_url=base_url, api_key=api_key, timeout=1500)
        self.model_name = model_name
        self.sampling_params = sampling_params

    async def _run(self, state: TheoremForgeState):
        prompt = prompt_manager.sketch_correction(
            state.formal_statement,
            state.metadata["failed_sketch"]["code"],
            state.metadata["failed_sketch"]["error_str"],
        )

        response = await call_llm_interruptible(
            state,
            self.context,
            self.client,
            self.model_name,
            prompt,
            self.sampling_params,
            "sketch_correction_agent",
        )
        code = extract_lean_code(response[0])
        if not code:
            logger.debug(
                f"Sketch Correction Agent: Failed to extract code from response for state {state.id}"
            )
            await self.add_state_request("finish_agent", state)
            return

        if not statement_check(state.formal_statement, code):
            logger.debug(
                f"Sketch Correction Agent: Proof does not contain the formal statement for state {state.id}"
            )
            await self.add_state_request("finish_agent", state)
            return
            
        valid, messages, error_str = await self.context.verifier.verify(
            code, True
        )
        if valid:
            logger.debug(
                f"Sketch Correction Agent: Successfully corrected proof sketch for state {state.id}"
            )
            state.proof_sketch = code
            await self.add_state_request("subgoal_extraction_agent", state)
        else:
            logger.debug(
                f"Sketch Correction Agent: Failed to correct proof sketch for state {state.id}, routing to finish"
            )
            await self.add_state_request("finish_agent", state)
