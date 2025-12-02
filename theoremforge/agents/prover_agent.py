from loguru import logger
from openai import AsyncOpenAI
from theoremforge.state import TheoremForgeContext
from theoremforge.utils import extract_lean_code, call_llm_interruptible, CancellationError
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

    async def run(self):
        while True:
            try:
                state = await self.task_queue.get()
                logger.info(f"Prover Agent: Start to process state {state.id}")

                # Register cancellation event for this state
                await self.register_cancellation_event(state)

                # Check if state should be skipped (blacklisted or cancelled)
                if await self.should_skip_state(state):
                    await self.add_state_request("finish_agent", state)
                    await self.cleanup_cancellation_event(state)
                    continue

                useful_theorems = state.metadata.get("useful_theorems", "")
                prompt = prompt_manager.proof_attempt(
                    state.formal_statement, useful_theorems
                )

                # Use interruptible LLM call
                response = await call_llm_interruptible(
                    state,
                    self.context,
                    self.client,
                    self.model_name,
                    prompt,
                    self.sampling_params,
                    "prover_agent",
                )
                codes = [
                    extract_lean_code(code)
                    for code in response
                ]
                logger.debug(
                    f"Prover Agent: Generated {len(codes)} codes for state {state.id}"
                )

                if not any(codes):
                    if state.parent_id:
                        logger.info(
                            f"Prover Agent: Failed to generate formal proof for state {state.id}, routing to finish_agent"
                        )
                        await self.add_state_request("finish_agent", state)
                    else:
                        logger.info(
                            f"Prover Agent: Failed to generate formal proof for state {state.id}, routing to proof_sketch_agent"
                        )
                        await self.add_state_request("proof_sketch_agent", state)
                    continue

                valid_flag = False
                failed_proofs = []
                failed_response = []
                for i, code in enumerate(codes):
                    if valid_flag:
                        break

                    if not code:
                        continue

                    # Check for cancellation before verification
                    if await self.is_cancelled(state):
                        logger.info(
                            f"Prover Agent: State {state.id} cancelled during verification loop"
                        )
                        await self.add_state_request("finish_agent", state)
                        await self.cleanup_cancellation_event(state)
                        break

                    valid, messages, error_str = await self.context.verifier.verify(
                        code, False
                    )
                    if valid:
                        logger.info(
                            f"Prover Agent: Successfully generated formal proof for state {state.id}"
                        )
                        state.formal_proof = code
                        state.success = True
                        valid_flag = True
                        await self.add_state_request("finish_agent", state)
                    else:
                        failed_proofs.append((code, error_str))
                        failed_response.append(response[i])

                    await self.db.provertrace.create(
                        data={
                            "prompt": prompt,
                            "output": response[i],
                            "formalStatement": state.formal_statement,
                            "outputCode": code,
                            "valid": valid,
                            "errorMessage": error_str,
                            "stateId": state.id,
                        }
                    )

                logger.debug(
                    f"Prover Agent: Finished processing {len(codes)} codes, valid_flag={valid_flag}, failed_proofs={len(failed_proofs)}"
                )

                # Clean up cancellation event if processing completed normally
                if valid_flag:
                    await self.cleanup_cancellation_event(state)
                
                if not valid_flag:
                    if failed_proofs:
                        logger.info(
                            f"Prover Agent: All codes failed for state {state.id}, routing to proof_correction"
                        )
                        # Store ALL failed attempts for proof correction
                        state.metadata["failed_attempts"] = [
                            {"code": code, "error_str": error_str}
                            for code, error_str in failed_proofs
                        ]
                        state.metadata["prev_attempts"] = [
                            [
                                {"role": "user", "content": prompt},
                                {"role": "assistant", "content": failed_response[i]},
                            ]
                            for i in range(len(failed_response))
                        ]
                        await self.add_state_request("proof_correction_agent", state)
                        logger.debug(
                            f"Prover Agent: Successfully routed state {state.id} to proof_correction with {len(failed_proofs)} failed attempts"
                        )
                    else:
                        # All codes were None/empty, route to finish_agent
                        logger.info(
                            f"Prover Agent: No valid codes generated for state {state.id}, routing to finish_agent"
                        )
                        await self.add_state_request("finish_agent", state)
                        logger.debug(
                            f"Prover Agent: Successfully routed state {state.id} to finish_agent"
                        )
                
                # Cleanup cancellation event after routing
                await self.cleanup_cancellation_event(state)

            except CancellationError as e:
                # State was cancelled during processing
                logger.info(f"Prover Agent: {e}")
                if "state" in locals():
                    await self.add_state_request("finish_agent", state)
                    await self.cleanup_cancellation_event(state)
            except Exception as e:
                logger.error(f"Prover Agent: Error processing state: {e}")
                import traceback

                traceback.print_exc()
                # Try to route to finish even on error
                try:
                    if "state" in locals():
                        await self.add_state_request("finish_agent", state)
                except Exception:
                    pass
