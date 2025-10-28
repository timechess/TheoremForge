from theoremforge.agents.base_agent import BaseAgent
from theoremforge.state import TheoremForgeContext, TheoremForgeState
from loguru import logger
from uuid import uuid4


class SubgoalExtractionAgent(BaseAgent):
    def __init__(self, context: TheoremForgeContext):
        super().__init__(agent_name="subgoal_extraction_agent", context=context)

    async def run(self):
        while True:
            try:
                state = await self.task_queue.get()
                logger.info(
                    f"Subgoal Extraction Agent: Start to process state {state.id}"
                )

                # Check black_list with lock
                async with self.context.black_list_lock:
                    is_blacklisted = state.id in self.context.black_list

                if is_blacklisted:
                    await self.add_state_request("finish_agent", state)
                    continue
                subgoals = await self.context.verifier.extract_subgoals(
                    state.proof_sketch
                )
                if not subgoals:
                    logger.info(
                        f"Subgoal Extraction Agent: No subgoals found for state {state.id}"
                    )
                    await self.add_state_request("finish_agent", state)
                    continue
                logger.info(
                    f"Subgoal Extraction Agent: Found {len(subgoals)} subgoals for state {state.id}"
                )
                subgoal_ids = [str(uuid4()) for _ in subgoals]
                for subgoal_id, subgoal in zip(subgoal_ids, subgoals):
                    new_state = TheoremForgeState(
                        id=subgoal_id,
                        formal_statement=subgoal,
                        parent_id=state.id,
                        siblings=subgoal_ids,
                        depth=state.depth + 1,
                    )
                    await self.add_state_request("prover_agent", new_state)
                state.subgoals = subgoal_ids
                await self.add_state_request("proof_assembly_agent", state)
            except Exception as e:
                logger.error(f"Subgoal Extraction Agent: Error processing state: {e}")
                import traceback

                traceback.print_exc()
                try:
                    if "state" in locals():
                        await self.add_state_request("finish_agent", state)
                except Exception:
                    pass
