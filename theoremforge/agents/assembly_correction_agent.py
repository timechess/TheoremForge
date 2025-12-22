from theoremforge.agents.base_agent import BaseAgent
from theoremforge.state import TheoremForgeContext, TheoremForgeState
from openai import AsyncOpenAI
from google.genai import Client
from theoremforge.utils import extract_lean_code, call_llm_interruptible
from theoremforge.prompt_manager import prompt_manager
from loguru import logger


class AssemblyCorrectionAgent(BaseAgent):
    def __init__(
        self,
        context: TheoremForgeContext,
        base_url: str,
        api_key: str,
        model_name: str,
        sampling_params: dict,
    ):
        super().__init__(agent_name="assembly_correction_agent", context=context)
        if model_name.startswith("gemini-"):
            self.client = Client(api_key=api_key, http_options={"base_url": base_url}, vertexai=True)
        else:
            self.client = AsyncOpenAI(base_url=base_url, api_key=api_key, timeout=1500)
        self.model_name = model_name
        self.sampling_params = sampling_params

    async def _run(self, state: TheoremForgeState):
        if state.metadata.get("failed_assembly"):
            failed_assembly = state.metadata["failed_assembly"]
            prompt = prompt_manager.proof_correction(
                failed_assembly["code"], failed_assembly["error"]
            )
            response = await call_llm_interruptible(
                state,
                self.context,
                self.client,
                self.model_name,
                prompt,
                self.sampling_params,
                "assembly_correction_agent",
            )
            code = extract_lean_code(response[0])
            if not code:
                logger.info(
                    f"Assembly Correction Agent: Failed to extract Lean code from response for state {state.id}"
                )
                await self.add_state_request("finish_agent", state)
                return
               
            valid, messages, error_str = await self.context.verifier.verify(
                code, False
            )
            if valid:
                logger.info(
                    f"Assembly Correction Agent: Successfully corrected assembly for state {state.id}"
                )
                state.formal_proof = code
                state.success = True
                await self.add_state_request("finish_agent", state)
            else:
                logger.info(
                    f"Assembly Correction Agent: Failed to correct assembly for state {state.id}, routing to finish_agent"
                )
                await self.add_state_request("finish_agent", state)
