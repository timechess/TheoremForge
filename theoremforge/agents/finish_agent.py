from theoremforge.agents.base_agent import BaseAgent
from theoremforge.state import TheoremForgeContext
from loguru import logger
import asyncio


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
            # Update black_list with lock and trigger cancellation
            if not state.success:
                if state.siblings:
                    # Add siblings to blacklist
                    async with self.context.black_list_lock:
                        for sibling_id in state.siblings:
                            self.context.black_list.add(sibling_id)

                    # Trigger cancellation events for siblings to interrupt ongoing work
                    async with self.context.cancellation_lock:
                        for sibling_id in state.siblings:
                            if sibling_id not in self.context.cancellation_events:
                                self.context.cancellation_events[sibling_id] = (
                                    asyncio.Event()
                                )
                            self.context.cancellation_events[sibling_id].set()
                            logger.debug(
                                f"Triggered cancellation for sibling state {sibling_id}"
                            )

            await self.cleanup_cancellation_event(state)
            await self.context.db.create_state(
                state={
                    "id": state.id,
                    "informal_statement": state.informal_statement,
                    "formal_statement": state.formal_statement,
                    "formal_proof": state.formal_proof,
                    "subgoals": state.subgoals,
                    "parent_id": state.parent_id,
                    "success": state.success,
                }
            )
            logger.info(f"Saving state {state.id} to database")
