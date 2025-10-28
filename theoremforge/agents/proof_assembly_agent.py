from theoremforge.agents.base_agent import BaseAgent
from openai import AsyncOpenAI
from theoremforge.state import TheoremForgeContext
from theoremforge.prompt_manager import prompt_manager

from loguru import logger
from theoremforge.utils import extract_lean_code
import asyncio


class ProofAssemblyAgent(BaseAgent):
    def __init__(
        self,
        context: TheoremForgeContext,
        base_url: str,
        api_key: str,
        model_name: str,
        sampling_params: dict,
    ):
        super().__init__(
            agent_name="proof_assembly_agent",
            context=context,
        )
        self.client = AsyncOpenAI(base_url=base_url, api_key=api_key)
        self.model_name = model_name
        self.sampling_params = sampling_params

    async def run(self):
        while True:
            try:
                state = await self.task_queue.get()

                # Check black_list with lock
                async with self.context.black_list_lock:
                    is_blacklisted = state.id in self.context.black_list

                if is_blacklisted:
                    logger.debug(f"Proof Assembly Agent: State {state.id} is blacklisted, routing to finish")
                    await self.add_state_request("finish_agent", state)
                    continue

                # Check if subgoals are ready in record
                async with self.context.record_lock:
                    subgoals_in_record = any(
                        [subgoal_id in self.context.record for subgoal_id in state.subgoals]
                    )
                    all_subgoals_ready = False
                    subgoal_proofs = []

                    if subgoals_in_record:
                        all_subgoals_ready = all(
                            [
                                subgoal_id in self.context.record
                                and self.context.record[subgoal_id]
                                for subgoal_id in state.subgoals
                            ]
                        )
                        if all_subgoals_ready:
                            subgoal_proofs = [
                                self.context.record[subgoal_id]
                                for subgoal_id in state.subgoals
                            ]

                if subgoals_in_record:
                    if all_subgoals_ready:
                        logger.info(
                            f"Proof Assembly Agent: All subgoals found for state {state.id}"
                        )

                        prompt = prompt_manager.proof_assembly(
                            state.formal_statement,
                            state.proof_sketch,
                            "\n".join(subgoal_proofs),
                        )
                        response = await self.client.chat.completions.create(
                            model=self.model_name,
                            messages=[
                                {"role": "user", "content": prompt},
                            ],
                            **self.sampling_params,
                        )
                        content = response.choices[0].message.content
                        code = extract_lean_code(content)
                        if code:
                            valid, messages, error_str = await self.context.verifier.verify(
                                code, False
                            )
                            if valid:
                                state.formal_proof = code
                                state.success = True
                                await self.add_state_request("finish_agent", state)
                            else:
                                state.metadata["failed_attempt"] = {
                                    "code": code,
                                    "error_str": error_str,
                                    "type": "proof_assembly",
                                }
                                await self.add_state_request("self_correction_agent", state)
                            await self.db.proofassemblytrace.create(
                                data={
                                    "prompt": prompt,
                                    "output": content,
                                    "formalStatement": state.formal_statement,
                                    "proofSketch": state.proof_sketch,
                                    "subgoalProofs": subgoal_proofs,
                                    "finalProof": code,
                                    "valid": valid,
                                    "errorMessage": error_str,
                                    "stateId": state.id,
                                }
                            )
                        else:
                            logger.info(
                                f"Proof Assembly Agent: Failed to generate formal proof for state {state.id}"
                            )
                            await self.db.proofassemblytrace.create(
                                data={
                                    "prompt": prompt,
                                    "output": content,
                                    "formalStatement": state.formal_statement,
                                    "proofSketch": state.proof_sketch,
                                    "subgoalProofs": subgoal_proofs,
                                    "finalProof": None,
                                    "valid": False,
                                    "errorMessage": "No formal proof generated",
                                    "stateId": state.id,
                                }
                            )
                            await self.add_state_request("finish_agent", state)

                    else:
                        logger.info(
                            f"Proof Assembly Agent: Some subgoals failed for state {state.id}"
                        )
                        await self.add_state_request("finish_agent", state)
                else:
                    await self.task_queue.put(state)
                    await asyncio.sleep(1)
            except Exception as e:
                logger.error(f"Proof Assembly Agent: Error processing state: {e}")
                import traceback
                traceback.print_exc()
                try:
                    if "state" in locals():
                        await self.add_state_request("finish_agent", state)
                except Exception:
                    pass