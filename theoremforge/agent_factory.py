"""
Agent Factory for TheoremForge.

This module provides a factory pattern for creating and managing agents,
promoting better dependency injection and modularity.
"""

from typing import Dict, Any, Optional
from theoremforge.agents.prover_agent import ProverAgent
from theoremforge.agents.informal_proof_agent import InformalProofAgent
from theoremforge.agents.theorem_retriever_agent import TheoremRetrieverAgent
from theoremforge.agents.verifier_agent import VerifierAgent
from theoremforge.agents.subgoal_extraction_agent import SubgoalExtractionAgent
from theoremforge.agents.decomposition_agent import DecompositionAgent
from theoremforge.agents.subgoal_solving_agent import SubgoalSolvingAgent
from theoremforge.agents.proof_assembly_agent import ProofAssemblyAgent
from theoremforge.agents.self_correction_agent import SelfCorrectionAgent
from theoremforge.config import config
from loguru import logger
import os


class AgentFactory:
    """
    Factory for creating and managing TheoremForge agents.

    Benefits:
    - Centralized agent creation
    - Dependency injection
    - Easy testing and mocking
    - Configuration management
    """

    def __init__(self, custom_config: Optional[Dict[str, Any]] = None):
        """
        Initialize the agent factory.

        Args:
            custom_config: Optional custom configuration to override defaults
        """
        self.config = custom_config or {}
        self._agents: Dict[str, Any] = {}
        self._initialized = False

    def initialize(self):
        """Initialize all agents with proper dependencies."""
        if self._initialized:
            logger.warning("Agent factory already initialized")
            return

        logger.info("Initializing agent factory...")

        # Get configurations
        lean_config = self.config.get("lean_server") or config.lean_server
        prover_config = self.config.get("prover_agent") or config.prover_agent
        informal_proof_config = (
            self.config.get("informal_proof_agent") or config.informal_proof_agent
        )
        decomp_config = (
            self.config.get("decomposition_agent") or config.decomposition_agent
        )
        theorem_retriever_config = (
            self.config.get("theorem_retriever_agent") or config.theorem_retriever_agent
        )

        # Create base agents first (no dependencies)
        self._agents["verifier"] = self._create_verifier_agent(lean_config)
        self._agents["theorem_retriever"] = self._create_theorem_retriever_agent(
            theorem_retriever_config
        )
        self._agents["subgoal_extraction"] = self._create_subgoal_extraction_agent(
            lean_config
        )
        self._agents["informal_proof"] = self._create_informal_proof_agent(
            informal_proof_config
        )
        self._agents["self_correction"] = self._create_self_correction_agent(
            prover_config
        )
        # Create agents with dependencies
        self._agents["prover"] = self._create_prover_agent(
            prover_config, self._agents["verifier"], self._agents["self_correction"]
        )
        self._agents["decomposition"] = self._create_decomposition_agent(
            decomp_config,
            self._agents["theorem_retriever"],
            self._agents["informal_proof"],
            self._agents["verifier"],
            self._agents["subgoal_extraction"],
            self._agents["self_correction"],
        )
        self._agents["subgoal_solving"] = self._create_subgoal_solving_agent(
            prover_config, self._agents["prover"]
        )
        self._agents["proof_assembly"] = self._create_proof_assembly_agent(
            decomp_config, self._agents["verifier"]
        )

        self._initialized = True
        logger.info(f"Agent factory initialized with {len(self._agents)} agents")

    def _create_verifier_agent(self, lean_config: Dict[str, Any]) -> VerifierAgent:
        """Create verifier agent."""
        url = f"http://localhost:{lean_config['LeanServerPort']}"
        logger.debug(f"Creating verifier agent with URL: {url}")
        return VerifierAgent(url)

    def _create_self_correction_agent(
        self, prover_config: Dict[str, Any]
    ) -> SelfCorrectionAgent:
        """Create self correction agent."""
        url = f"http://localhost:{prover_config['ProverPort']}/v1"
        model = prover_config.get("ProverModel", "model/Goedel-Prover-V2-32B")
        logger.debug(f"Creating self correction agent with URL: {url}, model: {model}")
        return SelfCorrectionAgent(url, "EMPTY", model)

    def _create_theorem_retriever_agent(
        self, theorem_retriever_config: Dict[str, Any]
    ) -> TheoremRetrieverAgent:
        """Create theorem retriever agent."""
        retriever_url = theorem_retriever_config.get(
            "RetrieverServerURL", "http://localhost:8002"
        )
        base_url = "https://api.openai-proxy.org/v1"
        api_key = os.getenv("CLOSEAI_API_KEY")
        if not api_key:
            logger.warning("CLOSEAI_API_KEY not found in environment")
        model = theorem_retriever_config.get("TheoremRetrieverAgentModel", "gpt-5-nano")
        logger.debug(f"Creating theorem retriever agent with URL: {base_url}")
        return TheoremRetrieverAgent(base_url, api_key, model, retriever_url)

    def _create_subgoal_extraction_agent(
        self, lean_config: Dict[str, Any]
    ) -> SubgoalExtractionAgent:
        """Create subgoal extraction agent."""
        url = f"http://localhost:{lean_config['LeanServerPort']}"
        logger.debug(f"Creating subgoal extraction agent with URL: {url}")
        return SubgoalExtractionAgent(url)

    def _create_informal_proof_agent(
        self, informal_proof_config: Dict[str, Any]
    ) -> InformalProofAgent:
        """Create informal proof agent."""
        api_key = os.getenv("CLOSEAI_API_KEY")
        if not api_key:
            logger.warning("CLOSEAI_API_KEY not found in environment")
        model = informal_proof_config.get("InformalProofAgentModel", "gpt-5-mini")

        url = "https://api.openai-proxy.org/v1"
        logger.debug(f"Creating informal proof agent with model: {model}")
        return InformalProofAgent(url, api_key, model)

    def _create_prover_agent(
        self,
        prover_config: Dict[str, Any],
        verifier_agent: VerifierAgent,
        self_correction_agent: SelfCorrectionAgent,
    ) -> ProverAgent:
        """Create prover agent."""
        url = f"http://localhost:{prover_config['ProverPort']}/v1"
        model = prover_config.get("ProverModel", "model/Goedel-Prover-V2-32B")
        logger.debug(f"Creating prover agent with URL: {url}, model: {model}")
        return ProverAgent(url, "EMPTY", model, verifier_agent, self_correction_agent)

    def _create_decomposition_agent(
        self,
        decomp_config: Dict[str, Any],
        theorem_retriever_agent: TheoremRetrieverAgent,
        informal_proof_agent: InformalProofAgent,
        verifier_agent: VerifierAgent,
        subgoal_extraction_agent: SubgoalExtractionAgent,
        self_correction_agent: SelfCorrectionAgent,
    ) -> DecompositionAgent:
        """Create decomposition agent."""
        url = "https://api.openai-proxy.org/v1"
        model = decomp_config.get("DecompositionAgentModel", "gpt-5-mini")
        api_key = os.getenv("CLOSEAI_API_KEY")
        logger.debug(f"Creating decomposition agent with URL: {url}, model: {model}")
        return DecompositionAgent(
            url,
            api_key,
            model,
            theorem_retriever_agent,
            informal_proof_agent,
            verifier_agent,
            subgoal_extraction_agent,
            self_correction_agent,
        )

    def _create_subgoal_solving_agent(
        self, prover_config: Dict[str, Any], prover_agent: ProverAgent
    ) -> SubgoalSolvingAgent:
        """Create subgoal solving agent."""
        logger.debug("Creating subgoal solving agent")
        url = f"http://localhost:{prover_config['ProverPort']}/v1"
        model = prover_config.get("ProverModel", "model/Goedel-Prover-V2-32B")
        return SubgoalSolvingAgent(url, "EMPTY", model, prover_agent)

    def _create_proof_assembly_agent(
        self, decomp_config: Dict[str, Any], verifier_agent: VerifierAgent
    ) -> ProofAssemblyAgent:
        """Create proof assembly agent."""
        api_key = os.getenv("CLOSEAI_API_KEY")
        if not api_key:
            logger.warning("CLOSEAI_API_KEY not found in environment")
        model = decomp_config.get("DecompositionAgentModel", "gpt-5-mini")
        url = "https://api.openai-proxy.org/v1"
        logger.debug(f"Creating proof assembly agent with model: {model}")
        return ProofAssemblyAgent(url, api_key, model, verifier_agent)

    def get_agent(self, agent_name: str) -> Any:
        """
        Get an agent by name.

        Args:
            agent_name: Name of the agent

        Returns:
            The requested agent

        Raises:
            ValueError: If agent not found or factory not initialized
        """
        if not self._initialized:
            raise ValueError("Agent factory not initialized. Call initialize() first.")

        if agent_name not in self._agents:
            raise ValueError(
                f"Agent '{agent_name}' not found. Available: {list(self._agents.keys())}"
            )

        return self._agents[agent_name]

    def get_all_agents(self) -> Dict[str, Any]:
        """Get all agents."""
        if not self._initialized:
            raise ValueError("Agent factory not initialized. Call initialize() first.")
        return self._agents.copy()
