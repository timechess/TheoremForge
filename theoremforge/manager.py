"""
Optimized TheoremForge State Manager (V2).

This is a fully async, high-performance version of the TheoremForge manager that:
- Continuously accepts new requests
- Processes multiple states concurrently
- Uses dependency injection for better modularity
- Provides real-time state persistence
- Includes comprehensive error handling
"""

import asyncio
import json
from pathlib import Path
from typing import Optional, Callable, Dict, Any
from hashlib import sha256
from loguru import logger

from theoremforge.state import TheoremForgeState
from theoremforge.async_queue_manager import AsyncQueueManager
from theoremforge.agent_factory import AgentFactory
from theoremforge.retry_handler import RetryHandler, RetryConfig
from theoremforge.config import config
from dotenv import load_dotenv

load_dotenv()

class TheoremForgeStateManager:
    """
    Optimized async state manager for TheoremForge.

    Key improvements:
    - Fully async with concurrent processing
    - Continuous request acceptance
    - Modular agent management via factory
    - Real-time state persistence
    - Comprehensive error handling and logging
    - Graceful shutdown
    """

    def __init__(
        self,
        max_workers: int = 5,
        output_file: str = "state_trace.jsonl",
        custom_config: Optional[Dict[str, Any]] = None,
        state_callback: Optional[Callable[[TheoremForgeState], None]] = None,
        enable_retry: bool = True,
        retry_config: Optional[RetryConfig] = None
    ):
        """
        Initialize the state manager.

        Args:
            max_workers: Maximum concurrent workers per stage
            output_file: Path to output file for state traces
            custom_config: Optional custom configuration
            state_callback: Optional custom callback for state persistence
            enable_retry: Whether to enable retry logic for failed operations
            retry_config: Custom retry configuration (uses defaults if None)
        """
        self.max_workers = max_workers
        self.output_file = Path(output_file)
        self.custom_config = custom_config
        self.enable_retry = enable_retry

        # Initialize retry handler
        if enable_retry:
            self.retry_handler = RetryHandler(retry_config or RetryConfig(
                max_retries=2,
                initial_delay=2.0,
                max_delay=30.0
            ))
        else:
            self.retry_handler = None

        # Initialize agent factory
        self.agent_factory = AgentFactory(custom_config)
        self.agent_factory.initialize()

        # Initialize queue manager
        self.queue_manager = AsyncQueueManager(
            max_workers=max_workers,
            state_callback=state_callback or self._default_state_callback
        )

        # Register stage handlers
        self._register_handlers()

        # Statistics
        self.stats = {
            "total_submitted": 0,
            "total_finished": 0,
            "successful": 0,
            "failed": 0,
            "retried": 0
        }
        self._stats_lock = asyncio.Lock()

        # Output file setup
        self._ensure_output_file()

        logger.info(
            f"TheoremForge Manager V2 initialized with {max_workers} workers per stage"
            f"{' (retry enabled)' if enable_retry else ''}"
        )

    def _ensure_output_file(self):
        """Ensure output file exists and is writable."""
        self.output_file.parent.mkdir(parents=True, exist_ok=True)
        # Create/clear the file
        self.output_file.touch()

    def _register_handlers(self):
        """Register all stage handlers with the queue manager."""
        agents = self.agent_factory.get_all_agents()

        # Wrapper to add retry logic
        async def with_retry(handler_func, state, *args, **kwargs):
            if self.enable_retry and self.retry_handler:
                try:
                    result = await self.retry_handler.execute_with_retry(
                        handler_func, state, *args, **kwargs
                    )
                    return result
                except Exception:
                    # All retries exhausted
                    async with self._stats_lock:
                        self.stats["retried"] += 1
                    raise
            else:
                return await handler_func(state, *args, **kwargs)

        # First attempt stage
        async def first_attempt_handler_impl(state: TheoremForgeState):
            prover_config = config.prover_agent
            return await agents['prover'].run(
                state,
                state.formal_statement,
                **prover_config.get("sampling_params", {})
            )

        async def first_attempt_handler(state: TheoremForgeState):
            return await with_retry(first_attempt_handler_impl, state)

        # Problem decomposition stage
        async def problem_decomposition_handler_impl(state: TheoremForgeState):
            decomp_config = config.decomposition_agent
            return await agents['decomposition'].run(
                state,
                state.formal_statement,
                **decomp_config.get("sampling_params", {})
            )

        async def problem_decomposition_handler(state: TheoremForgeState):
            return await with_retry(problem_decomposition_handler_impl, state)

        # Subgoal solving stage
        async def subgoal_solving_handler_impl(state: TheoremForgeState):
            prover_config = config.prover_agent
            for i in range(len(state.subgoals)):
                state, subgoal_solving_trace = await agents['subgoal_solving'].run(
                    state, i, **prover_config["sampling_params"]
                )
                if subgoal_solving_trace.formal_proofs:
                    return state, subgoal_solving_trace
            state.stage = "finished"
            state.result = "failure"
            return state, None

        async def subgoal_solving_handler(state: TheoremForgeState):
            return await with_retry(subgoal_solving_handler_impl, state)

        # Proof assembly stage
        async def proof_assembly_handler_impl(state: TheoremForgeState):
            decomp_config = config.decomposition_agent
            return await agents['proof_assembly'].run(
                state,
                **decomp_config.get("sampling_params", {})
            )

        async def proof_assembly_handler(state: TheoremForgeState):
            return await with_retry(proof_assembly_handler_impl, state)

        self.queue_manager.register_handler("first_attempt", first_attempt_handler)
        self.queue_manager.register_handler("problem_decoposition", problem_decomposition_handler)
        self.queue_manager.register_handler("subgoal_solving", subgoal_solving_handler)
        self.queue_manager.register_handler("proof_assembly", proof_assembly_handler)

    async def _default_state_callback(self, state: TheoremForgeState):
        """Default callback for persisting finished states."""
        if state.stage == "finished":
            # Update statistics
            async with self._stats_lock:
                self.stats["total_finished"] += 1
                if state.result == "success":
                    self.stats["successful"] += 1
                else:
                    self.stats["failed"] += 1

            # Persist to file
            try:
                with open(self.output_file, 'a') as f:
                    f.write(json.dumps(state.model_dump(), ensure_ascii=False) + "\n")
                logger.info(
                    f"State {state.id} finished with result: {state.result} "
                    f"(Success: {self.stats['successful']}, Failed: {self.stats['failed']})"
                )
            except Exception as e:
                logger.error(f"Error persisting state {state.id}: {e}")

    async def start(self):
        """Start the manager and begin processing."""
        await self.queue_manager.start()
        logger.info("TheoremForge Manager V2 started and ready to accept requests")

    async def stop(self, timeout: Optional[float] = 60.0):
        """
        Stop the manager gracefully.

        Args:
            timeout: Maximum time to wait for completion (seconds)
        """
        logger.info("Stopping TheoremForge Manager V2...")
        await self.queue_manager.stop(timeout=timeout)
        self._print_final_stats()
        logger.info("TheoremForge Manager V2 stopped")

    def _print_final_stats(self):
        """Print final statistics."""
        logger.info("=" * 60)
        logger.info("Final Statistics:")
        logger.info(f"  Total Submitted: {self.stats['total_submitted']}")
        logger.info(f"  Total Finished: {self.stats['total_finished']}")
        logger.info(f"  Successful: {self.stats['successful']}")
        logger.info(f"  Failed: {self.stats['failed']}")
        if self.stats['total_finished'] > 0:
            success_rate = (self.stats['successful'] / self.stats['total_finished']) * 100
            logger.info(f"  Success Rate: {success_rate:.2f}%")
        logger.info("=" * 60)

    async def submit_formal_statement(
        self,
        formal_statement: str,
        header: Optional[str] = None
    ) -> str:
        """
        Submit a new formal statement for proving.

        Args:
            formal_statement: The formal statement to prove
            header: Optional custom header (defaults to config)

        Returns:
            The unique ID of the submitted theorem
        """
        theorem_hash = sha256(formal_statement.encode("utf-8")).hexdigest()[:12]
        new_state = TheoremForgeState(
            id=theorem_hash,
            formal_statement=formal_statement,
            header=header if header else config.lean_server["LeanServerHeader"],
            stage="first_attempt",
            result="not_finished",
            trace=[],
        )

        await self.queue_manager.add_request("first_attempt", new_state)

        async with self._stats_lock:
            self.stats["total_submitted"] += 1

        logger.info(f"Submitted theorem {theorem_hash} (Total: {self.stats['total_submitted']})")
        return theorem_hash

    async def submit_multiple(
        self,
        formal_statements: list[str],
        header: Optional[str] = None,
        batch_size: int = 10
    ) -> list[str]:
        """
        Submit multiple formal statements.

        Args:
            formal_statements: List of formal statements
            header: Optional custom header
            batch_size: Number of statements to submit at once

        Returns:
            List of theorem IDs
        """
        theorem_ids = []

        for i in range(0, len(formal_statements), batch_size):
            batch = formal_statements[i:i+batch_size]
            batch_ids = await asyncio.gather(*[
                self.submit_formal_statement(stmt, header)
                for stmt in batch
            ])
            theorem_ids.extend(batch_ids)

        logger.info(f"Submitted {len(theorem_ids)} theorems in total")
        return theorem_ids

    def get_stats(self) -> Dict[str, Any]:
        """Get current statistics."""
        queue_stats = self.queue_manager.get_stats()
        return {
            **self.stats,
            **queue_stats
        }

    async def wait_for_completion(self, check_interval: float = 1.0):
        """
        Wait for all queued items to complete processing.

        Args:
            check_interval: How often to check for completion (seconds)
        """
        logger.info("Waiting for all tasks to complete...")
        prev_finished = 0

        while True:
            stats = self.get_stats()
            queue_sizes = stats['queue_sizes']
            active = stats['active_tasks']
            finished = stats['total_finished']

            # Check if all queues are empty and no active tasks
            if all(size == 0 for size in queue_sizes.values()) and active == 0:
                if stats['total_submitted'] == finished:
                    break

            # Log progress
            if finished > prev_finished:
                logger.info(
                    f"Progress: {finished}/{stats['total_submitted']} completed "
                    f"({stats['successful']} successful, {stats['failed']} failed)"
                )
                prev_finished = finished

            await asyncio.sleep(check_interval)

        logger.info("All tasks completed!")


# Convenience function for backward compatibility
async def run_theorem_forge(
    formal_statements: list[str],
    max_workers: int = 5,
    output_file: str = "state_trace.jsonl"
):
    """
    Convenience function to run TheoremForge with a list of statements.

    Args:
        formal_statements: List of formal statements to prove
        max_workers: Maximum concurrent workers per stage
        output_file: Output file path
    """
    manager = TheoremForgeStateManager(
        max_workers=max_workers,
        output_file=output_file
    )

    await manager.start()

    try:
        # Submit all statements
        await manager.submit_multiple(formal_statements)

        # Wait for completion
        await manager.wait_for_completion()

    finally:
        # Graceful shutdown
        await manager.stop()

