from theoremforge.agents.base_agent import BaseAgent
from theoremforge.state import TheoremForgeContext
from loguru import logger
from openai import AsyncOpenAI
import asyncio
from theoremforge.prompt_manager import prompt_manager
from theoremforge.utils import extract_lean_code, remove_comments
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

    async def _correct_single_formalization(
        self,
        state_id: str,
        informal_statement: str,
        normalized_statement: str,
        useful_definitions: str,
        failed_code: str,
        error_message: str,
        attempt_idx: int,
    ) -> tuple[int, list[tuple[str, str, str, bool, str]]]:
        """
        Correct a single failed formalization.
        
        Returns:
            Tuple of (attempt_idx, list of (output, code, prompt, valid, error_str))
        """
        results = []
        try:
            # Generate correction prompt with error message
            prompt = prompt_manager.statement_correction(
                informal_statement=informal_statement,
                failed_code=failed_code,
                error_message=error_message,
                normalized_statement=normalized_statement,
                useful_definitions=useful_definitions,
            )

            # Send LLM request
            logger.debug(
                f"Statement Correction Agent: Sending correction request for attempt {attempt_idx} of state {state_id}"
            )
            response = await self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                **self.sampling_params,
            )

            # Process response and verify codes
            corrected_codes = [
                extract_lean_code(choice.message.content)
                for choice in response.choices
            ]

            for i, code in enumerate(corrected_codes):
                output = response.choices[i].message.content
                
                if not code:
                    results.append((output, None, prompt, False, "Failed to extract Lean code"))
                    continue

                valid, messages, error_str = await self.context.verifier.verify(
                    code, True
                )
                results.append((output, code, prompt, valid, error_str if not valid else ""))

            return (attempt_idx, results)
            
        except Exception as e:
            logger.error(
                f"Statement Correction Agent: Error correcting formalization at attempt {attempt_idx} for state {state_id}: {e}"
            )
            return (attempt_idx, [])

    async def run(self):
        while True:
            try:
                state = await self.task_queue.get()
                logger.info(
                    f"Statement Correction Agent: Start to process state {state.id}"
                )

                # Check black_list with lock
                async with self.context.black_list_lock:
                    is_blacklisted = state.id in self.context.black_list or state.parent_id in self.context.black_list

                if is_blacklisted:
                    logger.debug(
                        f"Statement Correction Agent: State {state.id} is blacklisted, routing to finish"
                    )
                    await self.add_state_request("finish_agent", state)
                    continue

                if not state.informal_statement:
                    logger.error(
                        f"Statement Correction Agent: Missing informal statement for state {state.id}"
                    )
                    await self.add_state_request("finish_agent", state)
                    continue

                # Get failed formalizations with errors from metadata
                failed_formalizations = state.metadata.get("failed_formalizations", [])
                if not failed_formalizations:
                    logger.error(
                        f"Statement Correction Agent: No failed formalizations to correct for state {state.id}"
                    )
                    await self.add_state_request("finish_agent", state)
                    continue

                logger.info(
                    f"Statement Correction Agent: Attempting to correct {len(failed_formalizations)} failed formalizations concurrently for state {state.id}"
                )

                # Prepare correction tasks for concurrent processing
                normalized_statement = state.normalized_statement or ""
                useful_definitions = state.metadata.get("useful_definitions", "")
                
                correction_tasks = []
                for attempt_idx, failed_item in enumerate(failed_formalizations):
                    failed_code = failed_item.get("code", "")
                    error_message = failed_item.get("error", "")
                    
                    if not failed_code or not error_message:
                        logger.warning(
                            f"Statement Correction Agent: Skipping failed formalization {attempt_idx} - missing code or error"
                        )
                        continue
                    
                    correction_tasks.append(
                        self._correct_single_formalization(
                            state.id,
                            state.informal_statement,
                            normalized_statement,
                            useful_definitions,
                            failed_code,
                            error_message,
                            attempt_idx,
                        )
                    )

                # Run all corrections concurrently
                correction_results = await asyncio.gather(*correction_tasks)
                
                # Process results and save to database
                valid_codes = []
                for attempt_idx, results in correction_results:
                    for output, code, prompt, valid, error_str in results:
                        # Save trace to database
                        await self.db.statementcorrectiontrace.create(
                            data={
                                "prompt": prompt,
                                "output": output,
                                "outputCode": code,
                                "valid": valid,
                                "errorMessage": error_str,
                                "stateId": state.id,
                                "attemptIndex": attempt_idx,
                            }
                        )

                        # Collect valid corrected codes
                        if valid and code:
                            cleaned_code = remove_comments(erase_header(code.strip("\n")))
                            if cleaned_code not in valid_codes:
                                valid_codes.append(cleaned_code)
                                logger.info(
                                    f"Statement Correction Agent: Found valid corrected code for state {state.id} at attempt {attempt_idx}"
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

            except Exception as e:
                logger.error(f"Error in Statement Correction Agent: {e}")
                import traceback

                traceback.print_exc()
                try:
                    if "state" in locals():
                        await self.add_state_request("finish_agent", state)
                except Exception:
                    pass
