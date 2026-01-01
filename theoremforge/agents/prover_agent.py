from loguru import logger
from openai import AsyncOpenAI
from theoremforge.state import TheoremForgeContext, TheoremForgeState
from theoremforge.utils import (
    extract_lean_code,
    call_llm_interruptible,
    statement_check,
)
from theoremforge.agents.base_agent import BaseAgent
from theoremforge.prompt_manager import prompt_manager


class ProverAgent(BaseAgent):
    def __init__(
        self,
        context: TheoremForgeContext,
        base_url: str,
        api_key: str,
        model_name: str,
        sampling_params: dict,
    ) -> None:
        super().__init__(
            agent_name="prover_agent",
            context=context,
        )
        self.client = AsyncOpenAI(base_url=base_url, api_key=api_key, timeout=1500)
        self.model_name = model_name
        self.sampling_params = sampling_params

    async def _run(self, state: TheoremForgeState):
        prompt = prompt_manager.proof_attempt(state.formal_statement)
        response = await call_llm_interruptible(
            state,
            self.context,
            self.client,
            self.model_name,
            prompt,
            self.sampling_params,
            "prover_agent",
        )
        codes = [extract_lean_code(code) for code in response]
        if not any(codes):
            if state.parent_id:
                logger.debug(
                    f"Prover Agent: Failed to generate formal proof for state {state.id}, routing to finish_agent"
                )
                await self.add_state_request("shallow_solve_agent", state)
            else:
                logger.debug(
                    f"Prover Agent: Failed to generate formal proof for state {state.id}, routing to theorem_retrieval_agent"
                )
                await self.add_state_request("theorem_retrieval_agent", state)
            return

        valid_flag = False
        failed_proofs = []
        for i, code in enumerate(codes):
            if valid_flag:
                break

            if not code:
                continue
            
            if not statement_check(state.formal_statement, code):
                continue
            # Check for cancellation before verification
            if await self.is_cancelled(state):
                logger.debug(
                    f"Prover Agent: State {state.id} cancelled during verification loop"
                )
                await self.add_state_request("finish_agent", state)
                return

            valid, messages, error_str = await self.context.verifier.verify(code, False)
            if valid:
                logger.debug(
                    f"Prover Agent: Successfully generated formal proof for state {state.id}"
                )
                state.formal_proof = code
                state.success = True
                valid_flag = True
                await self.add_state_request("finish_agent", state)
            else:
                failed_proofs.append((code, error_str))

        logger.debug(
            f"Prover Agent: Finished processing {len(codes)} codes, valid_flag={valid_flag}, failed_proofs={len(failed_proofs)}"
        )

        if not valid_flag:
            if failed_proofs:
                logger.debug(
                    f"Prover Agent: All codes failed for state {state.id}, routing to proof_correction"
                )
                # Store ALL failed attempts for proof correction
                state.metadata["failed_attempts"] = [
                    {"code": code, "error_str": error_str}
                    for code, error_str in failed_proofs
                ]
                await self.add_state_request("proof_correction_agent", state)
                logger.debug(
                    f"Prover Agent: Successfully routed state {state.id} to proof_correction with {len(failed_proofs)} failed attempts"
                )
            else:
                if state.parent_id:
                    await self.add_state_request("informal_proof_agent", state)
                    logger.debug(
                        f"Prover Agent: Successfully routed state {state.id} to informal_proof_agent"
                    )
                else:
                    await self.add_state_request("theorem_retrieval_agent", state)
                    logger.debug(
                        f"Prover Agent: Successfully routed state {state.id} to theorem_retrieval_agent"
                    )
