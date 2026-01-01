"""
Standalone FinishAgent that runs in a separate process.

This module provides a multiprocess-compatible FinishAgent that:
1. Reads from a multiprocessing.Queue
2. Updates shared state via multiprocessing.Manager proxies
3. Writes to the database independently
"""

import multiprocessing as mp
from typing import Any
import asyncio
import signal

from loguru import logger

from theoremforge.state import TheoremForgeState
from theoremforge.db import SQLiteClient
from theoremforge.utils import statement_check


class FinishAgentProcess:
    """
    FinishAgent that runs in a separate process.
    
    Communicates with the main process via:
    - mp.Queue for receiving states to finish
    - Manager proxies for shared state (black_list, records, etc.)
    """
    
    def __init__(
        self,
        task_queue: mp.Queue,
        black_list: Any,  # Manager.list proxy
        statement_record: Any,  # Manager.dict proxy
        proof_record: Any,  # Manager.dict proxy
        cancellation_flags: Any,  # Manager.dict proxy
        db_path: str,
    ):
        self.task_queue = task_queue
        self.black_list = black_list
        self.statement_record = statement_record
        self.proof_record = proof_record
        self.cancellation_flags = cancellation_flags
        self.db_path = db_path
        self._running = True
        
    def _add_to_blacklist(self, state_id: str):
        """Add state to blacklist (dict-based set, O(1))."""
        try:
            self.black_list[state_id] = True
        except Exception as e:
            logger.warning(f"Failed to add {state_id} to blacklist: {e}")
    
    def _set_cancelled(self, state_id: str):
        """Mark state as cancelled."""
        self.cancellation_flags[state_id] = True
    
    def _cleanup_cancellation(self, state_id: str):
        """Remove cancellation flag."""
        try:
            if state_id in self.cancellation_flags:
                del self.cancellation_flags[state_id]
        except KeyError:
            pass
    
    async def _process_state(self, state: TheoremForgeState, db: SQLiteClient):
        """Process a single state."""
        logger.info(f"[FinishProcess] Finishing state {state.id}")
        
        # Check statement
        if state.success and not statement_check(state.formal_statement, state.formal_proof):
            state.success = False
        
        # Update records
        self.statement_record[state.id] = state.formal_statement
        if state.success:
            logger.info(f"[FinishProcess] Finishing state {state.id} successfully")
            self.proof_record[state.id] = state.formal_proof
        else:
            logger.info(f"[FinishProcess] Finishing state {state.id} failed")
            self.proof_record[state.id] = None
        
        # Update blacklist and cancellation for siblings
        if not state.success and state.siblings:
            for sibling_id in state.siblings:
                self._add_to_blacklist(sibling_id)
                self._set_cancelled(sibling_id)
                logger.debug(f"[FinishProcess] Blacklisted and cancelled sibling {sibling_id}")
        
        # Cleanup cancellation for this state
        self._cleanup_cancellation(state.id)
        
        # Save to database
        await db.create_state(
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
        logger.info(f"[FinishProcess] Saved state {state.id} to database")
    
    async def run_async(self):
        """Main async loop for processing states."""
        db = SQLiteClient(self.db_path)
        await db.connect()
        logger.info("[FinishProcess] Started")
        
        try:
            while self._running:
                try:
                    # Get state from queue with timeout
                    try:
                        state_dict = self.task_queue.get(timeout=0.5)
                    except Exception:
                        # Queue empty or timeout, continue loop
                        continue
                    
                    # Check for shutdown signal (None)
                    if state_dict is None:
                        logger.info("[FinishProcess] Received shutdown signal")
                        break
                    
                    # Reconstruct state from dict
                    state = TheoremForgeState(**state_dict) if isinstance(state_dict, dict) else state_dict
                    await self._process_state(state, db)
                    
                except Exception as e:
                    logger.error(f"[FinishProcess] Error: {e}")
                    import traceback
                    traceback.print_exc()
        finally:
            await db.disconnect()
            logger.info("[FinishProcess] Stopped")
    
    def run(self):
        """Entry point for the process."""
        def signal_handler(sig, frame):
            logger.info(f"[FinishProcess] Received signal {sig}, shutting down...")
            self._running = False
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        asyncio.run(self.run_async())


def run_finish_agent_process(
    task_queue: mp.Queue,
    black_list: Any,
    statement_record: Any,
    proof_record: Any,
    cancellation_flags: Any,
    db_path: str,
):
    """
    Entry point function for spawning FinishAgent process.
    
    This function is called by multiprocessing.Process.
    """
    agent = FinishAgentProcess(
        task_queue=task_queue,
        black_list=black_list,
        statement_record=statement_record,
        proof_record=proof_record,
        cancellation_flags=cancellation_flags,
        db_path=db_path,
    )
    agent.run()
