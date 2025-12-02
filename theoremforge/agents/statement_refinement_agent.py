from theoremforge.agents.base_agent import BaseAgent
from theoremforge.state import TheoremForgeContext
from loguru import logger
from openai import AsyncOpenAI
from theoremforge.prompt_manager import prompt_manager
from theoremforge.utils import extract_lean_code, call_llm_interruptible, CancellationError


class StatementRefinementAgent(BaseAgent):
    def __init__(
        self,
        context: TheoremForgeContext,
        base_url: str,
        api_key: str,
        model_name: str,
        sampling_params: dict,
    ):
        super().__init__(agent_name="statement_refinement_agent", context=context)
        self.client = AsyncOpenAI(base_url=base_url, api_key=api_key)
        self.model_name = model_name
        self.sampling_params = sampling_params

    async def run(self):
        while True:
            try:
                state = await self.task_queue.get()
                logger.info(
                    f"Statement Refinement Agent: Start to process state {state.id}"
                )

                # Register cancellation event for this state
                await self.register_cancellation_event(state)

                # Check if state should be skipped (blacklisted or cancelled)
                if await self.should_skip_state(state):
                    await self.add_state_request("finish_agent", state)
                    await self.cleanup_cancellation_event(state)
                    continue

                if not state.informal_statement:
                    logger.error(
                        f"Statement Refinement Agent: Missing informal statement for state {state.id}"
                    )
                    await self.add_state_request("finish_agent", state)
                    await self.cleanup_cancellation_event(state)
                    continue

                # Get the selected formalization from the previous agent
                selected_formalization = state.formal_statement
                if not selected_formalization:
                    logger.error(
                        f"Statement Refinement Agent: No formal statement to refine for state {state.id}"
                    )
                    await self.add_state_request("finish_agent", state)
                    await self.cleanup_cancellation_event(state)
                    continue

                logger.info(
                    f"Statement Refinement Agent: Attempting to refine formalization with hints for state {state.id}"
                )

                # Generate refinement prompt with hints
                prompt = prompt_manager.statement_refinement(
                    informal_statement=state.informal_statement,
                    failed_code=selected_formalization,
                    normalized_statement=state.normalized_statement or "",
                    useful_definitions=state.metadata.get("useful_definitions", ""),
                )

                # Send LLM request
                logger.debug(
                    f"Statement Refinement Agent: Sending refinement request for state {state.id}"
                )
                response = await call_llm_interruptible(
                    state,
                    self.context,
                    self.client,
                    self.model_name,
                    prompt,
                    self.sampling_params,
                    "statement_refinement_agent",
                )

                # Process response and verify codes
                final_formalization = None
                refined_codes = [
                    extract_lean_code(code)
                    for code in response
                ]

                for i, code in enumerate(refined_codes):
                    if not code:
                        await self.db.statementrefinementtrace.create(
                            data={
                                "prompt": prompt,
                                "output": response[i],
                                "outputCode": None,
                                "valid": False,
                                "errorMessage": "Failed to extract Lean code",
                                "stateId": state.id,
                                "attemptIndex": 0,
                            }
                        )
                        continue

                    # Check for cancellation during verification loop
                    if await self.is_cancelled(state):
                        logger.info(
                            f"Statement Refinement Agent: State {state.id} cancelled during verification"
                        )
                        await self.add_state_request("finish_agent", state)
                        await self.cleanup_cancellation_event(state)
                        break

                    valid, messages, error_str = await self.context.verifier.verify(
                        code, True
                    )

                    # Save trace to database
                    await self.db.statementrefinementtrace.create(
                        data={
                            "prompt": prompt,
                            "output": response[i],
                            "outputCode": code,
                            "valid": valid,
                            "errorMessage": error_str if not valid else "",
                            "stateId": state.id,
                            "attemptIndex": 0,
                        }
                    )

                    # Use the first valid refined code
                    if valid and final_formalization is None:
                        final_formalization = code
                        logger.info(
                            f"Statement Refinement Agent: Found valid refined formalization for state {state.id}"
                        )

                # If no valid refined codes, use the original formalization
                if final_formalization is None:
                    logger.warning(
                        f"Statement Refinement Agent: No valid refined code produced for state {state.id}, using original formalization"
                    )
                    final_formalization = selected_formalization

                logger.info(
                    f"Statement Refinement Agent: Using final formalization for state {state.id}, routing to theorem_retrieval_agent"
                )

                # Set the final formalization as the formal statement
                state.formal_statement = final_formalization
                await self.add_state_request("theorem_retrieval_agent", state)
                
                # Cleanup cancellation event after routing
                await self.cleanup_cancellation_event(state)

            except CancellationError as e:
                # State was cancelled during processing
                logger.info(f"Statement Refinement Agent: {e}")
                if "state" in locals():
                    await self.add_state_request("finish_agent", state)
                    await self.cleanup_cancellation_event(state)
            except Exception as e:
                logger.error(f"Error in Statement Refinement Agent: {e}")
                import traceback
                traceback.print_exc()
                try:
                    if "state" in locals():
                        await self.add_state_request("finish_agent", state)
                        await self.cleanup_cancellation_event(state)
                except Exception:
                    pass

