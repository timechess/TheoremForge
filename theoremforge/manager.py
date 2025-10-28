"""
TheoremForge State Manager - Agent-Based Architecture

This manager orchestrates multiple autonomous agents that communicate via queues.
Each agent processes states independently and routes them to other agents as needed.
"""

import asyncio
import os
from uuid import uuid4
from typing import Optional, Dict, Any, Callable
from loguru import logger
from dotenv import load_dotenv

from theoremforge.state import TheoremForgeState, TheoremForgeContext
from theoremforge.agents.prover_agent import ProverAgent
from theoremforge.agents.self_correction_agent import SelfCorrectionAgent
from theoremforge.agents.proof_sketch_agent import ProofSketchAgent
from theoremforge.agents.proof_assembly_agent import ProofAssemblyAgent
from theoremforge.agents.subgoal_extraction_agent import SubgoalExtractionAgent
from theoremforge.agents.informal_proof_agent import InformalProofAgent
from theoremforge.agents.theorem_retrieval_agent import TheoremRetrievalAgent
from theoremforge.agents.finish_agent import FinishAgent
from theoremforge.config import config
from theoremforge.db import MongoDBClient

load_dotenv()


class TheoremForgeStateManager:
    """
    Agent-based state manager for TheoremForge.

    Architecture:
    - Each agent has its own task queue and runs continuously
    - Agents communicate by routing states to each other
    - Shared context provides access to db, verifier, and inter-agent communication
    - Manager handles agent lifecycle and statistics tracking
    """

    def __init__(
        self,
        max_workers: int,
        custom_config: Optional[Dict[str, Any]] = None,
        state_callback: Optional[Callable[[TheoremForgeState], None]] = None,
    ):
        """
        Initialize the state manager.

        Args:
            verifier_url: URL of the Lean verification server
            retriever_url: URL of the theorem retrieval server
            custom_config: Optional custom configuration
            state_callback: Optional callback for finished states
        """
        self.max_workers = max_workers
        self.custom_config = custom_config or {}
        self.state_callback = state_callback

        # Get configurations
        lean_config = self.custom_config.get("lean_server") or config.lean_server
        prover_config = self.custom_config.get("prover_agent") or config.prover_agent
        self_correction_config = (
            self.custom_config.get("self_correction_agent")
            or config.self_correction_agent
        )
        informal_config = (
            self.custom_config.get("informal_proof_agent")
            or config.informal_proof_agent
        )
        sketch_config = (
            self.custom_config.get("proof_sketch_agent") or config.proof_sketch_agent
        )
        retriever_config = (
            self.custom_config.get("theorem_retriever_agent")
            or config.theorem_retrieval_agent
        )
        assembly_config = (
            self.custom_config.get("proof_assembly_agent")
            or config.proof_assembly_agent
        )

        # Resolve URLs
        self.verifier_url = (
            f"http://localhost:{lean_config.get('LeanServerPort', 8000)}"
        )
        self.retriever_url = retriever_config.get(
            "retriever_url", "http://localhost:8002"
        )

        # Initialize context
        self.context = TheoremForgeContext(verifier_url=self.verifier_url)

        # Initialize shared MongoDB client
        self.context.db = MongoDBClient()

        # Initialize agents
        self._initialize_agents(
            prover_config,
            self_correction_config,
            informal_config,
            sketch_config,
            retriever_config,
            assembly_config,
        )

        # Agent tasks for lifecycle management
        self.agent_tasks = []
        self._running = False

        # Statistics
        self.stats = {
            "total_submitted": 0,
            "total_finished": 0,
            "successful": 0,
            "failed": 0,
        }
        self._stats_lock = asyncio.Lock()

        # Finish agent callback
        self._setup_finish_callback()

        logger.info("TheoremForge Manager initialized with agent-based architecture")

    def _initialize_agents(
        self,
        prover_config,
        self_correction_config,
        informal_config,
        sketch_config,
        retriever_config,
        assembly_config,
    ):
        """Initialize all agents and register them in the context."""
        logger.info("Initializing agents...")
        openai_api_key = os.getenv("CLOSEAI_API_KEY", "EMPTY")
        # Get API keys and credentials
        prover_url = prover_config.get("base_url", "http://localhost:8001/v1")
        prover_model = prover_config.get("model", "model/Goedel-Prover-V2-32B")
        prover_sampling = prover_config.get("sampling_params", {})

        self_correction_url = self_correction_config.get(
            "base_url", "https://api.openai-proxy.org/v1"
        )
        self_correction_model = self_correction_config.get("model", "gpt-5-chat-latest")
        self_correction_sampling = self_correction_config.get("sampling_params", {})

        informal_url = informal_config.get(
            "base_url", "https://api.openai-proxy.org/v1"
        )
        informal_model = informal_config.get("model", "gpt-5-chat-latest")
        informal_sampling = informal_config.get("sampling_params", {})

        sketch_url = sketch_config.get("base_url", "https://api.openai-proxy.org/v1")
        sketch_model = sketch_config.get("model", "gpt-5-chat-latest")
        sketch_sampling = sketch_config.get("sampling_params", {})

        retriever_model_url = retriever_config.get(
            "base_url", "https://api.openai-proxy.org/v1"
        )
        retriever_model = retriever_config.get("model", "gpt-5-mini")
        retriever_sampling = retriever_config.get("sampling_params", {})

        assembly_url = assembly_config.get(
            "base_url", "https://api.openai-proxy.org/v1"
        )
        assembly_model = assembly_config.get("model", "gpt-5-chat-latest")
        assembly_sampling = assembly_config.get("sampling_params", {})

        # Create all agents
        self.context.agents["prover_agent"] = [
            ProverAgent(
                context=self.context,
                base_url=prover_url,
                api_key="EMPTY",
                model_name=prover_model,
                sampling_params=prover_sampling,
            )
            for _ in range(self.max_workers)
        ]

        self.context.agents["self_correction_agent"] = [
            SelfCorrectionAgent(
                context=self.context,
                base_url=self_correction_url,
                api_key=openai_api_key,
                model_name=self_correction_model,
                sampling_params=self_correction_sampling,
            )
            for _ in range(self.max_workers)
        ]

        self.context.agents["informal_proof_agent"] = [
            InformalProofAgent(
                context=self.context,
                base_url=informal_url,
                api_key=openai_api_key,
                model_name=informal_model,
                sampling_params=informal_sampling,
            )
            for _ in range(self.max_workers)
        ]

        self.context.agents["proof_sketch_agent"] = [
            ProofSketchAgent(
                context=self.context,
                base_url=sketch_url,
                api_key=openai_api_key,
                model_name=sketch_model,
                sampling_params=sketch_sampling,
            )
            for _ in range(self.max_workers)
        ]

        self.context.agents["subgoal_extraction_agent"] = [
            SubgoalExtractionAgent(
                context=self.context,
            )
        ]

        self.context.agents["proof_assembly_agent"] = [
            ProofAssemblyAgent(
                context=self.context,
                base_url=assembly_url,
                api_key=openai_api_key,
                model_name=assembly_model,
                sampling_params=assembly_sampling,
            )
            for _ in range(self.max_workers)
        ]

        self.context.agents["theorem_retrieval_agent"] = [
            TheoremRetrievalAgent(
                context=self.context,
                base_url=retriever_model_url,
                api_key=openai_api_key,
                model_name=retriever_model,
                retriever_url=self.retriever_url,
                sampling_params=retriever_sampling,
            )
            for _ in range(self.max_workers)
        ]

        self.context.agents["finish_agent"] = [
            FinishAgent(
                context=self.context,
            )
        ]

        logger.info(f"Initialized {len(self.context.agents)} agents")

    def _setup_finish_callback(self):
        """Setup callback to track finished states."""
        # We'll monitor the finish_agent's activity by wrapping its queue processing
        # This is done via a background task that monitors the record
        pass

    async def start(self):
        """Start all agents and begin processing."""
        if self._running:
            logger.warning("Manager already running")
            return

        logger.info("Starting TheoremForge Manager...")

        # Connect to shared MongoDB database
        await self.context.db.connect()
        logger.info("MongoDB database connected")

        # Start all agent tasks
        for agent_name, agents in self.context.agents.items():
            for i, agent in enumerate(agents):
                task = asyncio.create_task(agent.run(), name=f"agent_{agent_name}_{i}")
                self.agent_tasks.append(task)
                logger.debug(f"Started agent: {agent_name}_{i}")

        # Start statistics monitoring task
        task = asyncio.create_task(self._monitor_stats(), name="stats_monitor")
        self.agent_tasks.append(task)
        logger.debug("Started statistics monitoring task")

        self._running = True
        logger.info("TheoremForge Manager started - all agents running")

    async def stop(self, timeout: Optional[float] = 60.0):
        """
        Stop the manager gracefully.

        Args:
            timeout: Maximum time to wait for completion (seconds)
        """
        if not self._running:
            logger.warning("Manager not running")
            return

        logger.info("Stopping TheoremForge Manager...")
        self._running = False

        # Cancel all agent tasks
        for task in self.agent_tasks:
            task.cancel()

        # Wait for tasks to complete with timeout
        if self.agent_tasks:
            try:
                await asyncio.wait_for(
                    asyncio.gather(*self.agent_tasks, return_exceptions=True),
                    timeout=timeout,
                )
            except asyncio.TimeoutError:
                logger.warning("Some agent tasks did not complete within timeout")
            except Exception as e:
                logger.error(f"Error during shutdown: {e}")

        # Disconnect shared MongoDB database
        try:
            await self.context.db.disconnect()
            logger.info("MongoDB database disconnected")
        except Exception as e:
            logger.error(f"Error disconnecting MongoDB database: {e}")

        self._print_final_stats()
        logger.info("TheoremForge Manager stopped")

    async def _monitor_stats(self):
        """Monitor and update statistics by tracking the record dictionary."""
        try:
            last_record_size = 0
            while self._running:
                await asyncio.sleep(2.0)

                # Access record and root_state_ids with locks
                async with self.context.record_lock:
                    async with self.context.root_state_ids_lock:
                        # Only count root states (manually submitted), not subgoals
                        root_states_in_record = {
                            state_id for state_id in self.context.record.keys()
                            if state_id in self.context.root_state_ids
                        }
                        current_size = len(root_states_in_record)

                        if current_size > last_record_size:
                            # New root states finished
                            finished_count = current_size - last_record_size

                            # Count successes (non-None entries in record for root states)
                            successful = sum(
                                1 for state_id in root_states_in_record
                                if self.context.record[state_id] is not None
                            )
                            failed = len(root_states_in_record) - successful

                            async with self._stats_lock:
                                self.stats["total_finished"] += finished_count
                                self.stats["successful"] = successful
                                self.stats["failed"] = failed

                            last_record_size = current_size

        except asyncio.CancelledError:
            logger.debug("Stats monitor cancelled")
        except Exception as e:
            logger.error(f"Error in stats monitor: {e}")

    def _print_final_stats(self):
        """Print final statistics."""
        # Calculate total states including subgoals
        total_states_in_record = len(self.context.record)
        root_states_in_record = len([
            state_id for state_id in self.context.record.keys()
            if state_id in self.context.root_state_ids
        ])
        subgoals_in_record = total_states_in_record - root_states_in_record

        logger.info("=" * 60)
        logger.info("Final Statistics:")
        logger.info(f"  Root States (Manually Submitted): {self.stats['total_submitted']}")
        logger.info(f"  Root States Finished: {self.stats['total_finished']}")
        logger.info(f"  Root States Successful: {self.stats['successful']}")
        logger.info(f"  Root States Failed: {self.stats['failed']}")
        if self.stats["total_finished"] > 0:
            success_rate = (
                self.stats["successful"] / self.stats["total_finished"]
            ) * 100
            logger.info(f"  Root State Success Rate: {success_rate:.2f}%")
        logger.info(f"  Subgoals Generated & Finished: {subgoals_in_record}")
        logger.info(f"  Total States in Record: {total_states_in_record}")
        logger.info(f"  States in blacklist: {len(self.context.black_list)}")
        logger.info("=" * 60)

    async def submit_formal_statement(
        self,
        formal_statement: str,
        header: Optional[str] = None,
        metadata: Optional[dict] = None,
    ) -> str:
        """
        Submit a new formal statement for proving.

        Args:
            formal_statement: The formal statement to prove
            header: Optional custom header (defaults to config)
            metadata: Optional metadata dictionary

        Returns:
            The unique ID of the submitted theorem
        """
        if not self._running:
            raise RuntimeError("Manager not running. Call start() first.")

        theorem_id = str(uuid4())
        new_state = TheoremForgeState(
            id=theorem_id,
            formal_statement=formal_statement,
            header=header
            or config.lean_server.get("LeanServerHeader", "import Mathlib\n"),
            depth=0,
            success=False,
            metadata=metadata or {},
        )

        # Submit to prover agent (entry point)
        task_queue_lengths = [
            agent.task_queue.qsize() for agent in self.context.agents["prover_agent"]
        ]
        min_index = task_queue_lengths.index(min(task_queue_lengths))
        await self.context.agents["prover_agent"][min_index].task_queue.put(new_state)

        # Track this as a root state
        async with self.context.root_state_ids_lock:
            self.context.root_state_ids.add(theorem_id)

        async with self._stats_lock:
            self.stats["total_submitted"] += 1

        logger.info(
            f"Submitted theorem {theorem_id} (Total: {self.stats['total_submitted']})"
        )
        return theorem_id

    async def submit_multiple(
        self,
        formal_statements: list[str],
        header: Optional[str] = None,
        batch_size: int = 10,
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
            batch = formal_statements[i : i + batch_size]
            batch_ids = await asyncio.gather(
                *[self.submit_formal_statement(stmt, header) for stmt in batch]
            )
            theorem_ids.extend(batch_ids)

        logger.info(f"Submitted {len(theorem_ids)} theorems in total")
        return theorem_ids

    async def get_stats(self) -> Dict[str, Any]:
        """Get current statistics."""
        async with self.context.record_lock:
            total_record_len = len(self.context.record)

            async with self.context.root_state_ids_lock:
                root_states_count = len([
                    state_id for state_id in self.context.record.keys()
                    if state_id in self.context.root_state_ids
                ])

        async with self.context.black_list_lock:
            blacklist_len = len(self.context.black_list)

        return {
            **self.stats,
            "queue_sizes": {
                name: [a.task_queue.qsize() for a in agent]
                for name, agent in self.context.agents.items()
            },
            "blacklist_size": blacklist_len,
            "record_size": total_record_len,
            "root_states_finished": root_states_count,
            "subgoals_finished": total_record_len - root_states_count,
        }

    async def wait_for_completion(self, check_interval: float = 2.0):
        """
        Wait for all queued items to complete processing.

        Args:
            check_interval: How often to check for completion (seconds)
        """
        logger.info("Waiting for all tasks to complete...")
        prev_finished = 0

        while True:
            stats = await self.get_stats()
            queue_sizes = stats["queue_sizes"]
            finished = stats["total_finished"]
            submitted = stats["total_submitted"]

            # Check if all queues are empty (except finish_agent which might have pending writes)
            active_queues = {
                name: size
                for name, size in queue_sizes.items()
                if any(s > 0 for s in size) and name != "finish_agent"
            }

            # If no active queues and we've processed everything submitted
            if not active_queues and finished >= submitted:
                # Give finish_agent time to drain
                await asyncio.sleep(1.0)
                stats = await self.get_stats()
                if all(size == 0 for size in stats["queue_sizes"].get("finish_agent", [])):
                    break

            # Log progress
            if finished > prev_finished:
                logger.info(
                    f"Progress: {finished}/{submitted} completed "
                    f"({stats['successful']} successful, {stats['failed']} failed)"
                )
                if active_queues:
                    logger.debug(f"Active queues: {active_queues}")
                prev_finished = finished

            await asyncio.sleep(check_interval)

        logger.info("All tasks completed!")


# Convenience function for backward compatibility
async def run_theorem_forge(
    formal_statements: list[str],
    max_workers: int,
):
    """
    Convenience function to run TheoremForge with a list of statements.

    Args:
        formal_statements: List of formal statements to prove
        verifier_url: URL of the verification server
        retriever_url: URL of the retrieval server
    """
    manager = TheoremForgeStateManager(
        max_workers=max_workers,
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
