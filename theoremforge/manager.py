"""
TheoremForge State Manager - Agent-Based Architecture

This manager orchestrates multiple autonomous agents that communicate via queues.
Each agent processes states independently and routes them to other agents as needed.
"""

import asyncio
import json
from uuid import uuid4
from typing import Optional, Dict, Any, Callable
from pathlib import Path
from loguru import logger
from dotenv import load_dotenv

from theoremforge.state import TheoremForgeState, TheoremForgeContext
from theoremforge.agents.prover_agent import ProverAgent
from theoremforge.agents.proof_correction_agent import ProofCorrectionAgent
from theoremforge.agents.sketch_correction_agent import SketchCorrectionAgent
from theoremforge.agents.shallow_solve_agent import ShallowSolveAgent
from theoremforge.agents.proof_sketch_agent import ProofSketchAgent
from theoremforge.agents.proof_assembly_agent import ProofAssemblyAgent
from theoremforge.agents.subgoal_extraction_agent import SubgoalExtractionAgent
from theoremforge.agents.informal_proof_agent import InformalProofAgent
from theoremforge.agents.theorem_retrieval_agent import TheoremRetrievalAgent
from theoremforge.agents.statement_normalization_agent import (
    StatementNormalizationAgent,
)
from theoremforge.agents.definition_retrieval_agent import DefinitionRetrievalAgent
from theoremforge.agents.autoformalization_agent import AutoformalizationAgent
from theoremforge.agents.semantic_check_agent import SemanticCheckAgent
from theoremforge.agents.statement_correction_agent import StatementCorrectionAgent
from theoremforge.agents.statement_refinement_agent import StatementRefinementAgent
from theoremforge.agents.formalization_selection_agent import (
    FormalizationSelectionAgent,
)
from theoremforge.agents.finish_agent import FinishAgent
from theoremforge.config import config
from theoremforge.db import MongoDBClient
from theoremforge.retriever import Retriever

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
        proof_correction_config = (
            self.custom_config.get("proof_correction_agent")
            or config.proof_correction_agent
        )
        sketch_correction_config = (
            self.custom_config.get("sketch_correction_agent")
            or config.sketch_correction_agent
        )

        shallow_solve_config = (
            self.custom_config.get("shallow_solve_agent") or config.shallow_solve_agent
        )
        informal_config = (
            self.custom_config.get("informal_proof_agent")
            or config.informal_proof_agent
        )
        sketch_config = (
            self.custom_config.get("proof_sketch_agent") or config.proof_sketch_agent
        )
        subgoal_extraction_config = (
            self.custom_config.get("subgoal_extraction_agent")
            or config.subgoal_extraction_agent
        )
        retrieval_config = (
            self.custom_config.get("retrieval_agent") or config.retrieval_agent
        )
        assembly_config = (
            self.custom_config.get("proof_assembly_agent")
            or config.proof_assembly_agent
        )
        statement_norm_config = (
            self.custom_config.get("statement_normalization_agent")
            or config.statement_normalization_agent
        )
        autoformalization_config = (
            self.custom_config.get("autoformalization_agent")
            or config.autoformalization_agent
        )
        semantic_check_config = (
            self.custom_config.get("semantic_check_agent")
            or config.semantic_check_agent
        )
        statement_correction_config = (
            self.custom_config.get("statement_correction_agent")
            or config.statement_correction_agent
        )
        statement_refinement_config = (
            self.custom_config.get("statement_refinement_agent")
            or config.statement_refinement_agent
        )
        formalization_selection_config = (
            self.custom_config.get("formalization_selection_agent")
            or config.formalization_selection_agent
        )

        # Resolve URLs
        self.verifier_url = (
            f"http://localhost:{lean_config.get('LeanServerPort', 8000)}"
        )
        self.retrieval_url = retrieval_config.get(
            "retriever_url", "http://localhost:8002"
        )
        self.retriever = Retriever(self.retrieval_url)

        # Initialize context
        self.context = TheoremForgeContext(verifier_url=self.verifier_url)

        # Initialize shared MongoDB client
        self.context.db = MongoDBClient()

        # Initialize agents
        self._initialize_agents(
            prover_config,
            proof_correction_config,
            sketch_correction_config,
            shallow_solve_config,
            informal_config,
            sketch_config,
            subgoal_extraction_config,
            retrieval_config,
            assembly_config,
            statement_norm_config,
            autoformalization_config,
            semantic_check_config,
            statement_correction_config,
            statement_refinement_config,
            formalization_selection_config,
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
        proof_correction_config,
        sketch_correction_config,
        shallow_solve_config,
        informal_config,
        sketch_config,
        subgoal_extraction_config,
        retrieval_config,
        assembly_config,
        statement_norm_config,
        autoformalization_config,
        semantic_check_config,
        statement_correction_config,
        statement_refinement_config,
        formalization_selection_config,
    ):
        """Initialize all agents and register them in the context."""
        logger.info("Initializing agents...")

        # Get configurations and resolve API keys
        prover_url = prover_config.get("base_url", "http://localhost:8001/v1")
        prover_model = prover_config.get("model", "model/Goedel-Prover-V2-32B")
        prover_api_key = config.get_api_key(prover_config)
        prover_sampling = prover_config.get("sampling_params", {})

        proof_correction_url = proof_correction_config.get(
            "base_url", "https://api.openai-proxy.org/v1"
        )
        proof_correction_model = proof_correction_config.get(
            "model", "gpt-5-chat-latest"
        )
        proof_correction_api_key = config.get_api_key(proof_correction_config)
        proof_correction_sampling = proof_correction_config.get("sampling_params", {})

        sketch_correction_url = sketch_correction_config.get(
            "base_url", "https://api.openai-proxy.org/v1"
        )
        sketch_correction_model = sketch_correction_config.get(
            "model", "gpt-5-chat-latest"
        )
        sketch_correction_api_key = config.get_api_key(sketch_correction_config)
        sketch_correction_sampling = sketch_correction_config.get("sampling_params", {})

        shallow_solve_url = shallow_solve_config.get(
            "base_url", "https://api.openai-proxy.org/v1"
        )
        shallow_solve_model = shallow_solve_config.get("model", "gpt-5-chat-latest")
        shallow_solve_api_key = config.get_api_key(shallow_solve_config)
        shallow_solve_max_rounds = shallow_solve_config.get("max_rounds", 5)
        shallow_solve_sampling = shallow_solve_config.get("sampling_params", {})

        informal_url = informal_config.get(
            "base_url", "https://api.openai-proxy.org/v1"
        )
        informal_model = informal_config.get("model", "gpt-5-chat-latest")
        informal_api_key = config.get_api_key(informal_config)
        informal_sampling = informal_config.get("sampling_params", {})

        sketch_url = sketch_config.get("base_url", "https://api.openai-proxy.org/v1")
        sketch_model = sketch_config.get("model", "gpt-5-chat-latest")
        sketch_api_key = config.get_api_key(sketch_config)
        sketch_sampling = sketch_config.get("sampling_params", {})

        subgoal_extraction_url = subgoal_extraction_config.get(
            "base_url", "https://api.openai-proxy.org/v1"
        )
        subgoal_extraction_model = subgoal_extraction_config.get(
            "model", "gpt-5-chat-latest"
        )
        subgoal_extraction_api_key = config.get_api_key(subgoal_extraction_config)
        subgoal_extraction_sampling = subgoal_extraction_config.get(
            "sampling_params", {}
        )

        retrieval_model_url = retrieval_config.get(
            "base_url", "https://api.openai-proxy.org/v1"
        )
        retrieval_model = retrieval_config.get("model", "gpt-5-mini")
        retrieval_api_key = config.get_api_key(retrieval_config)
        retrieval_sampling = retrieval_config.get("sampling_params", {})

        assembly_url = assembly_config.get(
            "base_url", "https://api.openai-proxy.org/v1"
        )
        assembly_model = assembly_config.get("model", "gpt-5-chat-latest")
        assembly_api_key = config.get_api_key(assembly_config)
        assembly_sampling = assembly_config.get("sampling_params", {})

        statement_norm_url = statement_norm_config.get(
            "base_url", "https://api.openai-proxy.org/v1"
        )
        statement_norm_model = statement_norm_config.get("model", "gpt-5-chat-latest")
        statement_norm_api_key = config.get_api_key(statement_norm_config)
        statement_norm_sampling = statement_norm_config.get("sampling_params", {})

        autoformalization_url = autoformalization_config.get(
            "base_url", "http://localhost:8001/v1"
        )
        autoformalization_model = autoformalization_config.get(
            "model", "model/StepFun-Formalizer-7B"
        )
        autoformalization_api_key = config.get_api_key(autoformalization_config)
        autoformalization_sampling = autoformalization_config.get("sampling_params", {})

        semantic_check_url = semantic_check_config.get(
            "base_url", "https://api.openai-proxy.org/v1"
        )
        semantic_check_model = semantic_check_config.get("model", "gpt-5-chat-latest")
        semantic_check_api_key = config.get_api_key(semantic_check_config)
        semantic_check_sampling = semantic_check_config.get("sampling_params", {})

        statement_correction_url = statement_correction_config.get(
            "base_url", "https://api.openai-proxy.org/v1"
        )
        statement_correction_model = statement_correction_config.get(
            "model", "gpt-5-chat-latest"
        )
        statement_correction_api_key = config.get_api_key(statement_correction_config)
        statement_correction_sampling = statement_correction_config.get(
            "sampling_params", {}
        )

        statement_refinement_url = statement_refinement_config.get(
            "base_url", "https://api.openai-proxy.org/v1"
        )
        statement_refinement_model = statement_refinement_config.get(
            "model", "gpt-5-chat-latest"
        )
        statement_refinement_api_key = config.get_api_key(statement_refinement_config)
        statement_refinement_sampling = statement_refinement_config.get(
            "sampling_params", {}
        )

        formalization_selection_url = formalization_selection_config.get(
            "base_url", "https://api.openai-proxy.org/v1"
        )
        formalization_selection_model = formalization_selection_config.get(
            "model", "gpt-5-chat-latest"
        )
        formalization_selection_api_key = config.get_api_key(
            formalization_selection_config
        )
        formalization_selection_sampling = formalization_selection_config.get(
            "sampling_params", {}
        )

        # Create all agents
        self.context.agents["prover_agent"] = [
            ProverAgent(
                context=self.context,
                base_url=prover_url,
                api_key=prover_api_key,
                model_name=prover_model,
                sampling_params=prover_sampling,
            )
            for _ in range(self.max_workers)
        ]

        self.context.agents["proof_correction_agent"] = [
            ProofCorrectionAgent(
                context=self.context,
                base_url=proof_correction_url,
                api_key=proof_correction_api_key,
                model_name=proof_correction_model,
                sampling_params=proof_correction_sampling,
            )
            for _ in range(self.max_workers)
        ]

        self.context.agents["sketch_correction_agent"] = [
            SketchCorrectionAgent(
                context=self.context,
                base_url=sketch_correction_url,
                api_key=sketch_correction_api_key,
                model_name=sketch_correction_model,
                sampling_params=sketch_correction_sampling,
            )
            for _ in range(self.max_workers)
        ]

        self.context.agents["shallow_solve_agent"] = [
            ShallowSolveAgent(
                context=self.context,
                base_url=shallow_solve_url,
                api_key=shallow_solve_api_key,
                model_name=shallow_solve_model,
                sampling_params=shallow_solve_sampling,
                max_rounds=shallow_solve_max_rounds,
            )
            for _ in range(self.max_workers)
        ]

        self.context.agents["informal_proof_agent"] = [
            InformalProofAgent(
                context=self.context,
                base_url=informal_url,
                api_key=informal_api_key,
                model_name=informal_model,
                sampling_params=informal_sampling,
            )
            for _ in range(self.max_workers)
        ]

        self.context.agents["proof_sketch_agent"] = [
            ProofSketchAgent(
                context=self.context,
                base_url=sketch_url,
                api_key=sketch_api_key,
                model_name=sketch_model,
                sampling_params=sketch_sampling,
            )
            for _ in range(self.max_workers)
        ]

        self.context.agents["subgoal_extraction_agent"] = [
            SubgoalExtractionAgent(
                context=self.context,
                base_url=subgoal_extraction_url,
                api_key=subgoal_extraction_api_key,
                model_name=subgoal_extraction_model,
                sampling_params=subgoal_extraction_sampling,
            )
            for _ in range(self.max_workers)
        ]

        self.context.agents["proof_assembly_agent"] = [
            ProofAssemblyAgent(
                context=self.context,
                base_url=assembly_url,
                api_key=assembly_api_key,
                model_name=assembly_model,
                sampling_params=assembly_sampling,
            )
            for _ in range(self.max_workers)
        ]

        self.context.agents["theorem_retrieval_agent"] = [
            TheoremRetrievalAgent(
                context=self.context,
                base_url=retrieval_model_url,
                api_key=retrieval_api_key,
                model_name=retrieval_model,
                retriever=self.retriever,
                sampling_params=retrieval_sampling,
            )
            for _ in range(self.max_workers)
        ]

        self.context.agents["statement_normalization_agent"] = [
            StatementNormalizationAgent(
                context=self.context,
                base_url=statement_norm_url,
                api_key=statement_norm_api_key,
                model_name=statement_norm_model,
                sampling_params=statement_norm_sampling,
            )
            for _ in range(self.max_workers)
        ]

        self.context.agents["definition_retrieval_agent"] = [
            DefinitionRetrievalAgent(
                context=self.context,
                base_url=retrieval_model_url,
                api_key=retrieval_api_key,
                model_name=retrieval_model,
                retriever=self.retriever,
                sampling_params=retrieval_sampling,
            )
            for _ in range(self.max_workers)
        ]

        self.context.agents["autoformalization_agent"] = [
            AutoformalizationAgent(
                context=self.context,
                base_url=autoformalization_url,
                api_key=autoformalization_api_key,
                model_name=autoformalization_model,
                sampling_params=autoformalization_sampling,
            )
            for _ in range(self.max_workers)
        ]

        self.context.agents["semantic_check_agent"] = [
            SemanticCheckAgent(
                context=self.context,
                base_url=semantic_check_url,
                api_key=semantic_check_api_key,
                model_name=semantic_check_model,
                sampling_params=semantic_check_sampling,
            )
            for _ in range(self.max_workers)
        ]

        self.context.agents["statement_correction_agent"] = [
            StatementCorrectionAgent(
                context=self.context,
                base_url=statement_correction_url,
                api_key=statement_correction_api_key,
                model_name=statement_correction_model,
                sampling_params=statement_correction_sampling,
            )
            for _ in range(self.max_workers)
        ]

        self.context.agents["statement_refinement_agent"] = [
            StatementRefinementAgent(
                context=self.context,
                base_url=statement_refinement_url,
                api_key=statement_refinement_api_key,
                model_name=statement_refinement_model,
                sampling_params=statement_refinement_sampling,
            )
            for _ in range(self.max_workers)
        ]

        self.context.agents["formalization_selection_agent"] = [
            FormalizationSelectionAgent(
                context=self.context,
                base_url=formalization_selection_url,
                api_key=formalization_selection_api_key,
                model_name=formalization_selection_model,
                sampling_params=formalization_selection_sampling,
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

        await self.context.verifier.close()
        logger.info("Lean verification server disconnected")
        await self.retriever.close()
        logger.info("Retriever disconnected")
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
                            state_id
                            for state_id in self.context.statement_record.keys()
                            if state_id in self.context.root_state_ids
                        }
                        current_size = len(root_states_in_record)

                        if current_size > last_record_size:
                            # New root states finished
                            finished_count = current_size - last_record_size

                            # Count successes (non-None entries in record for root states)
                            successful = sum(
                                1
                                for state_id in root_states_in_record
                                if self.context.proof_record[state_id] is not None
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
        total_states_in_record = len(self.context.statement_record)
        root_states_in_record = len(
            [
                state_id
                for state_id in self.context.statement_record.keys()
                if state_id in self.context.root_state_ids
            ]
        )
        subgoals_in_record = total_states_in_record - root_states_in_record

        logger.info("=" * 60)
        logger.info("Final Statistics:")
        logger.info(
            f"  Root States (Manually Submitted): {self.stats['total_submitted']}"
        )
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
            agent.task_queue.qsize() for agent in self.context.agents["theorem_retrieval_agent"]
        ]
        min_index = task_queue_lengths.index(min(task_queue_lengths))
        await self.context.agents["theorem_retrieval_agent"][min_index].task_queue.put(new_state)

        # Track this as a root state
        async with self.context.root_state_ids_lock:
            self.context.root_state_ids.add(theorem_id)

        async with self._stats_lock:
            self.stats["total_submitted"] += 1

        logger.info(
            f"Submitted theorem {theorem_id} (Total: {self.stats['total_submitted']})"
        )
        return theorem_id

    async def submit_informal_statement(
        self,
        informal_statement: str,
        header: Optional[str] = None,
        metadata: Optional[dict] = None,
    ) -> str:
        """
        Submit a new informal statement for autoformalization and proving.

        Args:
            informal_statement: The informal mathematical statement
            header: Optional custom header (defaults to config)
            metadata: Optional metadata dictionary

        Returns:
            The unique ID of the submitted statement
        """
        if not self._running:
            raise RuntimeError("Manager not running. Call start() first.")

        statement_id = str(uuid4())
        new_state = TheoremForgeState(
            id=statement_id,
            informal_statement=informal_statement,
            header=header
            or config.lean_server.get("LeanServerHeader", "import Mathlib\n"),
            depth=0,
            success=False,
            metadata=metadata or {},
        )

        # Submit to statement normalization agent (entry point for informal statements)
        task_queue_lengths = [
            agent.task_queue.qsize()
            for agent in self.context.agents["statement_normalization_agent"]
        ]
        min_index = task_queue_lengths.index(min(task_queue_lengths))
        await self.context.agents["statement_normalization_agent"][
            min_index
        ].task_queue.put(new_state)

        # Track this as a root state
        async with self.context.root_state_ids_lock:
            self.context.root_state_ids.add(statement_id)

        async with self._stats_lock:
            self.stats["total_submitted"] += 1

        logger.info(
            f"Submitted informal statement {statement_id} (Total: {self.stats['total_submitted']})"
        )
        return statement_id

    async def submit_multiple(
        self,
        formal_statements: Optional[list[str]] = None,
        informal_statements: Optional[list[str]] = None,
        header: Optional[str] = None,
        batch_size: int = 10,
    ) -> list[str]:
        """
        Submit multiple formal or informal statements.

        Args:
            formal_statements: List of formal statements (provide this OR informal_statements)
            informal_statements: List of informal statements (provide this OR formal_statements)
            header: Optional custom header
            batch_size: Number of statements to submit at once

        Returns:
            List of statement/theorem IDs

        Raises:
            ValueError: If both or neither arguments are provided
        """
        # Validate inputs
        if formal_statements is not None and informal_statements is not None:
            raise ValueError(
                "Cannot provide both formal_statements and informal_statements. "
                "Please provide only one."
            )
        if formal_statements is None and informal_statements is None:
            raise ValueError(
                "Must provide either formal_statements or informal_statements"
            )

        # Determine which type of statements to process
        if formal_statements is not None:
            statements = formal_statements
            submit_func = self.submit_formal_statement
            statement_type = "formal"
        else:
            statements = informal_statements
            submit_func = self.submit_informal_statement
            statement_type = "informal"

        # Submit in batches
        statement_ids = []
        for i in range(0, len(statements), batch_size):
            batch = statements[i : i + batch_size]
            batch_ids = await asyncio.gather(
                *[submit_func(stmt, header) for stmt in batch]
            )
            statement_ids.extend(batch_ids)

        logger.info(
            f"Submitted {len(statement_ids)} {statement_type} statements in total"
        )
        return statement_ids

    async def get_stats(self) -> Dict[str, Any]:
        """Get current statistics."""
        async with self.context.record_lock:
            total_record_len = len(self.context.statement_record)

            async with self.context.root_state_ids_lock:
                root_states_count = len(
                    [
                        state_id
                        for state_id in self.context.statement_record.keys()
                        if state_id in self.context.root_state_ids
                    ]
                )

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
                if all(
                    size == 0 for size in stats["queue_sizes"].get("finish_agent", [])
                ):
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

    async def export_results(
        self,
        output_file: str,
        statement_ids: Optional[list[str]] = None,
        custom_ids: Optional[Dict[str, Any]] = None,
    ):
        """
        Export theorem proving results to JSONL format.

        Args:
            output_file: Path to the output JSONL file
            statement_ids: Optional list of statement IDs to export. If None, exports all root states.
            custom_ids: Optional dict mapping statement IDs to custom IDs
        """
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Determine which states to export
        if statement_ids is None:
            # Export all root states
            async with self.context.root_state_ids_lock:
                ids_to_export = list(self.context.root_state_ids)
        else:
            ids_to_export = statement_ids

        # Fetch states from database
        exported_count = 0
        with open(output_path, "w", encoding="utf-8") as f:
            for state_id in ids_to_export:
                try:
                    # Fetch from database
                    state_doc = await self.context.db.theoremforgestate.find_one(
                        {"id": state_id}
                    )

                    if state_doc:
                        # Prepare export data
                        export_data = {
                            "id": state_id,
                            "informal_statement": state_doc.get(
                                "informalStatement", ""
                            ),
                            "formal_statement": state_doc.get("formalStatement", ""),
                            "formal_proof": state_doc.get("formalProof", ""),
                            "header": state_doc.get("header", ""),
                            "success": state_doc.get("success", False),
                        }

                        # Add custom_id if provided
                        if custom_ids and state_id in custom_ids:
                            export_data["custom_id"] = custom_ids[state_id]

                        # Write as JSONL
                        f.write(json.dumps(export_data, ensure_ascii=False) + "\n")
                        exported_count += 1

                except Exception as e:
                    logger.error(f"Failed to export state {state_id}: {e}")

        logger.info(f"Exported {exported_count} results to {output_path}")
        return exported_count


# Convenience function for backward compatibility
async def run_theorem_forge(
    formal_statements: Optional[list[str]] = None,
    informal_statements: Optional[list[str]] = None,
    max_workers: int = 4,
    export_file: Optional[str] = None,
    custom_ids: Optional[list[Any]] = None,
):
    """
    Convenience function to run TheoremForge with a list of statements.

    Args:
        formal_statements: List of formal statements to prove (provide this OR informal_statements)
        informal_statements: List of informal statements to prove (provide this OR formal_statements)
        max_workers: Number of worker agents to create
        export_file: Optional path to export results in JSONL format
        custom_ids: Optional list of custom IDs (same length and order as statements)
    """
    manager = TheoremForgeStateManager(
        max_workers=max_workers,
    )

    await manager.start()

    try:
        # Submit all statements and track IDs
        statement_ids = await manager.submit_multiple(
            formal_statements=formal_statements,
            informal_statements=informal_statements,
        )

        # Wait for completion
        await manager.wait_for_completion()

        # Export results if requested
        if export_file:
            # Create mapping from statement_ids to custom_ids
            custom_id_mapping = None
            if custom_ids is not None:
                if len(custom_ids) != len(statement_ids):
                    logger.warning(
                        f"custom_ids length ({len(custom_ids)}) does not match "
                        f"statement_ids length ({len(statement_ids)}). Some custom_ids may be ignored."
                    )
                # Map statement IDs to custom IDs
                custom_id_mapping = {
                    statement_id: custom_id
                    for statement_id, custom_id in zip(statement_ids, custom_ids)
                }

            await manager.export_results(export_file, statement_ids, custom_id_mapping)

    finally:
        # Graceful shutdown
        await manager.stop()
