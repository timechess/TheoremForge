import asyncio
from loguru import logger
from theoremforge.agents.base_agent import BaseAgent
from theoremforge.state import TheoremForgeContext
from openai import AsyncOpenAI
from theoremforge.utils import extract_lean_code
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
        self.client = AsyncOpenAI(base_url=base_url, api_key=api_key, timeout=1500)
        self.model_name = model_name
        self.sampling_params = sampling_params

    async def run(self):
        while True:
            try:
                state = await self.task_queue.get()
                logger.info(
                    f"Proof Correction Agent: Start to process state {state.id}"
                )

                # Check black_list with lock
                async with self.context.black_list_lock:
                    is_blacklisted = state.id in self.context.black_list or state.parent_id in self.context.black_list

                if is_blacklisted:
                    logger.debug(
                        f"Proof Correction Agent: State {state.id} is blacklisted"
                    )
                    await self.add_state_request("finish_agent", state)
                    continue

                # Support both old format (single failed_attempt) and new format (multiple failed_attempts)
                failed_attempts = []
                if "failed_attempts" in state.metadata:
                    failed_attempts = state.metadata["failed_attempts"]
                elif "failed_attempt" in state.metadata:
                    # Backward compatibility with old format
                    failed_attempts = [state.metadata["failed_attempt"]]
                else:
                    logger.error(
                        f"Proof Correction Agent: No failed attempts found for state {state.id}"
                    )
                    await self.add_state_request("finish_agent", state)
                    continue

                logger.info(
                    f"Proof Correction Agent: Processing {len(failed_attempts)} failed attempts for state {state.id}"
                )

                # Generate all prompts
                if "prev_attempts" in state.metadata:
                    prev_attempts = state.metadata["prev_attempts"]

                prompts = [
                    prompt_manager.proof_correction(
                        failed_attempt["code"],
                        failed_attempt["error_str"],
                    )
                    for failed_attempt in failed_attempts
                ]

                # Send all LLM requests in parallel
                logger.debug(
                    f"Proof Correction Agent: Sending {len(prompts)} correction requests in parallel for state {state.id}"
                )
                responses = await asyncio.gather(
                    *[
                        self.client.chat.completions.create(
                            model=self.model_name,
                            messages=prev_attempt
                            + [{"role": "user", "content": prompt}],
                            **self.sampling_params,
                        )
                        for prev_attempt, prompt in zip(prev_attempts, prompts)
                    ]
                )

                # Process all responses and verify codes
                valid_flag = False
                for attempt_idx, (response, prompt) in enumerate(
                    zip(responses, prompts)
                ):
                    if valid_flag:
                        break

                    codes = [
                        extract_lean_code(choice.message.content)
                        for choice in response.choices
                    ]

                    for i, code in enumerate(codes):
                        if valid_flag:
                            break
                        if not code:
                            await self.db.proofcorrectiontrace.create(
                                data={
                                    "prompt": prompt,
                                    "output": response.choices[i].message.content,
                                    "outputCode": None,
                                    "valid": False,
                                    "errorMessage": "Failed to extract Lean code",
                                    "stateId": state.id,
                                    "attemptIndex": attempt_idx,
                                }
                            )
                            continue

                        valid, messages, error_str = await self.context.verifier.verify(
                            code, False
                        )

                        if valid:
                            logger.info(
                                f"Proof Correction Agent: Successfully corrected formal proof for state {state.id} (attempt {attempt_idx + 1})"
                            )
                            state.formal_proof = code
                            state.success = True
                            valid_flag = True
                            await self.add_state_request("finish_agent", state)

                        await self.db.proofcorrectiontrace.create(
                            data={
                                "prompt": prompt,
                                "output": response.choices[i].message.content,
                                "outputCode": code,
                                "valid": valid,
                                "errorMessage": error_str,
                                "stateId": state.id,
                                "attemptIndex": attempt_idx,
                            }
                        )

                if not valid_flag:
                    # Check if this is a subgoal (has parent_id)
                    if state.parent_id:
                        logger.info(
                            f"Proof Correction Agent: Failed to correct subgoal {state.id}, routing to finish_agent"
                        )
                        await self.add_state_request("finish_agent", state)
                    else:
                        logger.info(
                            f"Proof Correction Agent: Failed to correct formal proof for state {state.id}, routing to proof_sketch_agent"
                        )
                        await self.add_state_request("proof_sketch_agent", state)

            except Exception as e:
                logger.error(f"Proof Correction Agent: Error processing state: {e}")
                import traceback

                traceback.print_exc()
                try:
                    if "state" in locals():
                        await self.add_state_request("finish_agent", state)
                except Exception:
                    pass
