from openai import AsyncOpenAI
from google.genai import Client
from theoremforge.agents.base_agent import BaseAgent
from theoremforge.state import TheoremForgeContext, TheoremForgeState
from theoremforge.utils import (
    extract_lean_code,
    call_llm_interruptible,
)
from theoremforge.prompt_manager import prompt_manager
from loguru import logger


class ProofSketchAgent(BaseAgent):
    def __init__(
        self,
        context: TheoremForgeContext,
        base_url: str,
        api_key: str,
        model_name: str,
        sampling_params: dict,
    ):
        super().__init__(
            agent_name="proof_sketch_agent",
            context=context,
        )
        if model_name.startswith("gemini-"):
            self.client = Client(api_key=api_key, http_options={"base_url": base_url}, vertexai=True)
        else:
            self.client = AsyncOpenAI(base_url=base_url, api_key=api_key, timeout=1500)
        self.model_name = model_name
        self.sampling_params = sampling_params

    async def _run(self, state: TheoremForgeState):
        prompt = prompt_manager.proof_sketch_generation(
            state.formal_statement,
            state.informal_proof,
            state.metadata["useful_theorems"],
        )
        response = await call_llm_interruptible(
            state,
            self.context,
            self.client,
            self.model_name,
            prompt,
            self.sampling_params,
            "proof_sketch_agent",
        )
        proof_sketch = extract_lean_code(response[0])
        if not proof_sketch:
            logger.info(
                "Proof Sketch Agent: Proof sketch generation failed. Moving to finished."
            )
            await self.add_state_request("finish_agent", state)
            return

        valid, messages, error_str = await self.context.verifier.verify(
            proof_sketch, True
        )
        if valid:
            logger.info(
                f"Proof Sketch Agent: Successfully generated proof sketch for state {state.id}"
            )
            state.proof_sketch = proof_sketch
            await self.add_state_request("subgoal_extraction_agent", state)
        else:
            logger.info(
                f"Proof Sketch Agent: Failed to generate proof sketch for state {state.id}, routing to sketch_correction"
            )
            state.metadata["failed_sketch"] = {
                "code": proof_sketch,
                "error_str": error_str,
            }
            await self.add_state_request("sketch_correction_agent", state)
