from loguru import logger
from theoremforge.agents.base_agent import BaseAgent
from theoremforge.state import TheoremForgeContext
from openai import AsyncOpenAI
from theoremforge.utils import extract_lean_code
from theoremforge.prompt_manager import prompt_manager


class SketchCorrectionAgent(BaseAgent):
    def __init__(
        self,
        context: TheoremForgeContext,
        base_url: str,
        api_key: str,
        model_name: str,
        sampling_params: dict,
    ):
        super().__init__(
            agent_name="sketch_correction_agent",
            context=context,
        )
        self.client = AsyncOpenAI(base_url=base_url, api_key=api_key)
        self.model_name = model_name
        self.sampling_params = sampling_params

    async def run(self):
        while True:
            try:
                state = await self.task_queue.get()
                logger.info(f"Sketch Correction Agent: Start to process state {state.id}")

                # Check black_list with lock
                async with self.context.black_list_lock:
                    is_blacklisted = state.id in self.context.black_list or state.parent_id in self.context.black_list

                if is_blacklisted:
                    logger.debug(f"Sketch Correction Agent: State {state.id} is blacklisted")
                    await self.add_state_request("finish_agent", state)
                    continue

                if "failed_attempt" not in state.metadata:
                    logger.error(f"Sketch Correction Agent: No failed attempt found for state {state.id}")
                    await self.add_state_request("finish_agent", state)
                    continue

                prompt = prompt_manager.sketch_correction(
                    state.formal_statement,
                    state.metadata["failed_attempt"]["code"],
                    state.metadata["failed_attempt"]["error_str"],
                )

                response = await self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    **self.sampling_params,
                )
                codes = [
                    extract_lean_code(choice.message.content) for choice in response.choices
                ]
                valid_flag = False

                for i, code in enumerate(codes):
                    if valid_flag:
                        break
                    if not code:
                        continue

                    valid, messages, error_str = await self.context.verifier.verify(
                        code, True
                    )

                    if valid:
                        logger.info(
                            f"Sketch Correction Agent: Successfully corrected proof sketch for state {state.id}"
                        )
                        state.proof_sketch = code
                        valid_flag = True
                        await self.add_state_request("subgoal_extraction_agent", state)

                    await self.db.sketchcorrectiontrace.create(
                        data={
                            "prompt": prompt,
                            "output": response.choices[i].message.content,
                            "outputCode": code,
                            "valid": valid,
                            "errorMessage": error_str,
                            "stateId": state.id,
                        }
                    )

                if not valid_flag:
                    logger.info(
                        f"Sketch Correction Agent: Failed to correct proof sketch for state {state.id}, routing to finish"
                    )
                    await self.add_state_request("finish_agent", state)

            except Exception as e:
                logger.error(f"Sketch Correction Agent: Error processing state: {e}")
                import traceback
                traceback.print_exc()
                try:
                    if 'state' in locals():
                        await self.add_state_request("finish_agent", state)
                except Exception:
                    pass


