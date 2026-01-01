import asyncio
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


class ProofCorrectionAgent(BaseAgent):
    def __init__(
        self,
        context: TheoremForgeContext,
        base_url: str,
        api_key: str,
        model_name: str,
        sampling_params: dict,
    ):
        super().__init__(
            agent_name="proof_correction_agent",
            context=context,
        )
        if model_name.startswith("gemini-"):
            self.client = Client(api_key=api_key, http_options={"base_url": base_url}, vertexai=True)
        else:
            self.client = AsyncOpenAI(base_url=base_url, api_key=api_key, timeout=1500)
        self.model_name = model_name
        self.sampling_params = sampling_params

    async def _run(self, state: TheoremForgeState):
        failed_attempts = state.metadata["failed_attempts"]

        logger.debug(
            f"Proof Correction Agent: Processing {len(failed_attempts)} failed attempts for state {state.id}"
        )
        prompts = [
            prompt_manager.proof_correction(
                failed_attempt["code"],
                failed_attempt["error_str"],
            )
            for failed_attempt in failed_attempts
        ]

        # Send all LLM requests in parallel with cancellation support
        logger.debug(
            f"Proof Correction Agent: Sending {len(prompts)} correction requests in parallel for state {state.id}"
        )
        responses = await asyncio.gather(
            *[
                call_llm_interruptible(
                    state,
                    self.context,
                    self.client,
                    self.model_name,
                    prompt,
                    self.sampling_params,
                    "proof_correction_agent",
                )
                for prompt in prompts
            ]
        )

        # Process all responses and verify codes
        valid_flag = False
        for attempt_idx, (response, prompt) in enumerate(zip(responses, prompts)):
            if valid_flag:
                break

            codes = [extract_lean_code(code) for code in response]

            for i, code in enumerate(codes):
                if valid_flag:
                    break

                # Check for cancellation during verification loop
                if await self.is_cancelled(state):
                    logger.debug(
                        f"Proof Correction Agent: State {state.id} cancelled during verification"
                    )
                    await self.add_state_request("finish_agent", state)
                    return

                if not code:
                    continue

                if not statement_check(state.formal_statement, code):
                    continue

                valid, messages, error_str = await self.context.verifier.verify(
                    code, False
                )

                if valid:
                    logger.debug(
                        f"Proof Correction Agent: Successfully corrected formal proof for state {state.id} (attempt {attempt_idx + 1})"
                    )
                    state.formal_proof = code
                    state.success = True
                    valid_flag = True
                    await self.add_state_request("finish_agent", state)
                    return

        if not valid_flag:
            # Check if this is a subgoal (has parent_id)
            if state.parent_id:
                logger.debug(
                    f"Proof Correction Agent: Failed to correct subgoal {state.id}, routing to informal_proof_agent"
                )
                await self.add_state_request("informal_proof_agent", state)
            else:
                logger.debug(
                    f"Proof Correction Agent: Failed to correct formal proof for state {state.id}, routing to theorem_retrieval_agent"
                )
                await self.add_state_request("theorem_retrieval_agent", state)
