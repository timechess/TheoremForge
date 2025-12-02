from theoremforge.agents.base_agent import BaseAgent
from theoremforge.state import TheoremForgeContext
from openai import AsyncOpenAI
from theoremforge.utils import extract_lean_code, call_llm_interruptible, CancellationError
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
        self.client = AsyncOpenAI(base_url=base_url, api_key=api_key)
        self.model_name = model_name
        self.sampling_params = sampling_params

    async def run(self):
        while True:
            try:
                state = await self.task_queue.get()
                logger.info(
                    f"Assembly Correction Agent: Start to process state {state.id}"
                )

                # Register cancellation event for this state
                await self.register_cancellation_event(state)

                # Check if state should be skipped (blacklisted or cancelled)
                if await self.should_skip_state(state):
                    await self.add_state_request("finish_agent", state)
                    await self.cleanup_cancellation_event(state)
                    continue

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
                        await self.db.assemblycorrectiontrace.create(
                            data={
                                "prompt": prompt,
                                "output": response[0],
                                "outputCode": None,
                                "valid": False,
                                "errorMessage": "Failed to extract Lean code",
                                "stateId": state.id,
                            }
                        )
                        await self.cleanup_cancellation_event(state)
                        continue
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

                    await self.cleanup_cancellation_event(state)

                    await self.db.assemblycorrectiontrace.create(
                        data={
                            "prompt": prompt,
                            "output": response[0],
                            "outputCode": code,
                            "valid": valid,
                            "errorMessage": error_str,
                            "stateId": state.id,
                        }
                    )
            except CancellationError as e:
                # State was cancelled during processing
                logger.info(f"Assembly Correction Agent: {e}")
                if "state" in locals():
                    await self.add_state_request("finish_agent", state)
                    await self.cleanup_cancellation_event(state)
            except Exception as e:
                logger.error(f"Assembly Correction Agent: Error processing state: {e}")
                import traceback
                traceback.print_exc()
                try:
                    if "state" in locals():
                        await self.add_state_request("finish_agent", state)
                        await self.cleanup_cancellation_event(state)
                except Exception:
                    pass
