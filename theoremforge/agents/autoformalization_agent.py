from theoremforge.agents.base_agent import BaseAgent
from theoremforge.state import TheoremForgeContext
from openai import AsyncOpenAI
from loguru import logger
import asyncio
from theoremforge.prompt_manager import prompt_manager
from theoremforge.utils import extract_lean_code, remove_comments
from theoremforge.lean_server.server import erase_header


class AutoformalizationAgent(BaseAgent):
    def __init__(
        self,
        context: TheoremForgeContext,
        base_url: str,
        api_key: str,
        model_name: str,
        sampling_params: dict,
    ):
        super().__init__(agent_name="autoformalization_agent", context=context)
        self.client = AsyncOpenAI(base_url=base_url, api_key=api_key, timeout=1500)
        self.model_name = model_name
        self.sampling_params = sampling_params

    async def run(self):
        while True:
            try:
                state = await self.task_queue.get()
                logger.info(
                    f"AutoformalizationAgent: Start to process state {state.id}"
                )

                # Check black_list with lock
                async with self.context.black_list_lock:
                    is_blacklisted = (
                        state.id in self.context.black_list
                        or state.parent_id in self.context.black_list
                    )

                if is_blacklisted:
                    logger.debug(
                        f"AutoformalizationAgent: State {state.id} is blacklisted, routing to finish"
                    )
                    await self.add_state_request("finish_agent", state)
                    continue
                if not state.normalized_statement:
                    logger.error(
                        f"AutoformalizationAgent: Normalized statement is not available for state {state.id}"
                    )
                    await self.add_state_request("finish_agent", state)
                    continue

                autoformalization_prompt = prompt_manager.autoformalization(
                    state.normalized_statement, state.metadata["useful_definitions"]
                )
                response = await self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": autoformalization_prompt}],
                    **self.sampling_params,
                )
                codes = [
                    extract_lean_code(choice.message.content)
                    for choice in response.choices
                ]
                if not any(codes):
                    logger.error(
                        f"AutoformalizationAgent: Failed to extract any code for state {state.id}"
                    )
                    await self.add_state_request("finish_agent", state)
                    continue

                # Collect all valid codes (ensuring uniqueness)
                valid_codes = []
                failed_codes = []

                for i, code in enumerate(codes):
                    if not code:
                        continue

                    # Check syntax validity
                    valid, messages, error_str = await self.context.verifier.verify(
                        code, True
                    )

                    if valid:
                        # Only add if it's different from existing valid codes
                        if code not in valid_codes:
                            valid_codes.append(
                                remove_comments(erase_header(code.strip("\n")))
                            )
                        else:
                            logger.info(
                                f"AutoformalizationAgent: Skipping duplicate valid code for state {state.id}"
                            )
                    else:
                        failed_codes.append({"code": code, "error": error_str})

                    # Save trace
                    await self.db.autoformalizationtrace.create(
                        data={
                            "prompt": autoformalization_prompt,
                            "output": response.choices[i].message.content,
                            "formalStatement": code,
                            "outputCode": code,
                            "valid": valid,
                            "errorMessage": error_str,
                            "stateId": state.id,
                        }
                    )

                # If no syntactically valid codes, route to statement correction
                if not valid_codes:
                    logger.info(
                        f"AutoformalizationAgent: No syntactically valid code for state {state.id}, routing to statement_correction_agent"
                    )
                    # Store failed attempts for correction
                    state.metadata["failed_formalizations"] = failed_codes
                    await self.add_state_request("statement_correction_agent", state)
                    continue

                logger.info(
                    f"AutoformalizationAgent: Found {len(valid_codes)} unique syntactically valid codes for state {state.id}, routing to semantic_check_agent"
                )

                # Store all valid codes in metadata for semantic check agent
                state.metadata["valid_formalizations"] = valid_codes
                await self.add_state_request("semantic_check_agent", state)
            except Exception as e:
                logger.error(f"AutoformalizationAgent: Error in run: {e}")
                import traceback

                traceback.print_exc()
                try:
                    if "state" in locals():
                        await self.add_state_request("finish_agent", state)
                except Exception:
                    pass
                await asyncio.sleep(1)
