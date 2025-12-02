from theoremforge.agents.base_agent import BaseAgent
from theoremforge.state import TheoremForgeContext
from loguru import logger
from openai import AsyncOpenAI
import asyncio
from theoremforge.prompt_manager import prompt_manager
from theoremforge.utils import extract_lean_code, remove_comments, call_llm_interruptible, CancellationError
from theoremforge.lean_server.server import erase_header


class StatementCorrectionAgent(BaseAgent):
    def __init__(
        self,
        context: TheoremForgeContext,
        base_url: str,
        api_key: str,
        model_name: str,
        sampling_params: dict,
    ):
        super().__init__(agent_name="statement_correction_agent", context=context)
        self.client = AsyncOpenAI(base_url=base_url, api_key=api_key)
        self.model_name = model_name
        self.sampling_params = sampling_params

    async def run(self):
        while True:
            try:
                state = await self.task_queue.get()
                logger.info(
                    f"Statement Correction Agent: Start to process state {state.id}"
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
                        f"Statement Correction Agent: Missing informal statement for state {state.id}"
                    )
                    await self.add_state_request("finish_agent", state)
                    await self.cleanup_cancellation_event(state)
                    continue

                # Get failed formalizations with errors from metadata
                failed_formalizations = state.metadata.get("failed_formalizations", [])
                if not failed_formalizations:
                    logger.error(
                        f"Statement Correction Agent: No failed formalizations to correct for state {state.id}"
                    )
                    await self.add_state_request("finish_agent", state)
                    await self.cleanup_cancellation_event(state)
                    continue

                logger.info(
                    f"Statement Correction Agent: Attempting to correct {len(failed_formalizations)} failed formalizations concurrently for state {state.id}"
                )

                # Prepare correction tasks for concurrent processing
                normalized_statement = state.normalized_statement or ""
                useful_definitions = state.metadata.get("useful_definitions", "")

                correction_prompts = [
                    prompt_manager.statement_correction(
                        informal_statement=state.informal_statement,
                        failed_code=failed_formalization["code"],
                        error_message=failed_formalization["error"],
                        normalized_statement=normalized_statement,
                        useful_definitions=useful_definitions,
                    )
                    for failed_formalization in failed_formalizations
                ]
                correction_tasks = [
                    call_llm_interruptible(
                        state,
                        self.context,
                        self.client,
                        self.model_name,
                        prompt,
                        self.sampling_params,
                        "statement_correction_agent",
                    )
                    for prompt in correction_prompts
                ]
                responses = await asyncio.gather(*correction_tasks)

                # Process results and save to database
                valid_codes = []
                for i, response in enumerate(responses):
                    # Check for cancellation during verification loop
                    if await self.is_cancelled(state):
                        logger.info(
                            f"Statement Correction Agent: State {state.id} cancelled during verification"
                        )
                        await self.add_state_request("finish_agent", state)
                        await self.cleanup_cancellation_event(state)
                        break
                    
                    code = extract_lean_code(response[0])
                    if not code:
                        continue
                    valid, messages, error_str = await self.context.verifier.verify(
                        code, True
                    )
                    if valid:
                        cleaned_code = remove_comments(erase_header(code.strip("\n")))
                        if cleaned_code not in valid_codes:
                            valid_codes.append(cleaned_code)
                            logger.info(
                                f"Statement Correction Agent: Found valid corrected code for state {state.id} at attempt {i}"
                            )
                            await self.db.statementcorrectiontrace.create(
                                data={
                                    "prompt": correction_prompts[i],
                                    "output": response[0],
                                    "outputCode": code,
                                    "valid": valid,
                                    "errorMessage": error_str,
                                    "stateId": state.id,
                                }
                            )
                    else:
                        logger.info(
                            f"Statement Correction Agent: Found invalid corrected code for state {state.id} at attempt {i}"
                        )
                        await self.db.statementcorrectiontrace.create(
                            data={
                                "prompt": correction_prompts[i],
                                "output": response[0],
                                "outputCode": code,
                                "valid": valid,
                                "errorMessage": error_str,
                                "stateId": state.id,
                            }
                        )
                # If we found valid corrected codes, route to semantic check
                if valid_codes:
                    logger.info(
                        f"Statement Correction Agent: Found {len(valid_codes)} valid corrected codes for state {state.id}, routing to semantic_check_agent"
                    )
                    # Store all valid codes in metadata for semantic check agent
                    state.metadata["valid_formalizations"] = valid_codes
                    await self.add_state_request("semantic_check_agent", state)
                else:
                    logger.warning(
                        f"Statement Correction Agent: Failed to correct any code for state {state.id}, routing to finish_agent"
                    )
                    await self.add_state_request("finish_agent", state)
                
                # Cleanup cancellation event after routing
                await self.cleanup_cancellation_event(state)

            except CancellationError as e:
                # State was cancelled during processing
                logger.info(f"Statement Correction Agent: {e}")
                if "state" in locals():
                    await self.add_state_request("finish_agent", state)
                    await self.cleanup_cancellation_event(state)
            except Exception as e:
                logger.error(f"Error in Statement Correction Agent: {e}")
                import traceback
                traceback.print_exc()
                try:
                    if "state" in locals():
                        await self.add_state_request("finish_agent", state)
                        await self.cleanup_cancellation_event(state)
                except Exception:
                    pass
