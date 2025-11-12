from theoremforge.agents.base_agent import BaseAgent
from theoremforge.state import TheoremForgeContext
from loguru import logger


class FinishAgent(BaseAgent):
    def __init__(self, context: TheoremForgeContext):
        super().__init__(agent_name="finish_agent", context=context)

    async def run(self):
        while True:
            state = await self.task_queue.get()
            logger.info(f"Finishing state {state.id}")

            # Update record with lock
            async with self.context.record_lock:
                self.context.statement_record[state.id] = state.formal_statement
                if state.success:
                    logger.info(f"Finishing state {state.id} successfully")
                    self.context.proof_record[state.id] = state.formal_proof
                else:
                    logger.info(f"Finishing state {state.id} failed")
                    self.context.proof_record[state.id] = None
            # Update black_list with lock
            if not state.success:
                if state.siblings:
                    async with self.context.black_list_lock:
                        for sibling_id in state.siblings:
                            self.context.black_list.add(sibling_id)

            await self.db.theoremforgestate.create(
                data={
                    "id": state.id,
                    "header": state.header,
                    "informalStatement": state.informal_statement,
                    "normalizedStatement": state.normalized_statement,
                    "formalStatement": state.formal_statement,
                    "formalProof": state.formal_proof,
                    "informalProof": state.informal_proof,
                    "proofSketch": state.proof_sketch,
                    "subgoals": state.subgoals,
                    "depth": state.depth,
                    "parentId": state.parent_id,
                    "success": state.success,
                }
            )
            logger.info(f"Saving state {state.id} to database")
