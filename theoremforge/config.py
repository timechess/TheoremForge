"""
Configuration loader for TheoremForge.

This module loads configuration from config.yaml and provides easy access to configuration values.
"""
import os
import yaml
from pathlib import Path
from typing import Any, Dict


class Config:
    """Configuration loader that reads from config.yaml"""

    _instance = None
    _config: Dict[str, Any] = None

    def __new__(cls, config_path: str):
        if cls._instance is None:
            cls._instance = super(Config, cls).__new__(cls)
            cls._instance._load_config(config_path)
        return cls._instance

    def _load_config(self, config_path: str):
        """Load configuration from config.yaml"""
        if not Path(config_path).exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        with open(config_path, "r") as f:
            self._config = yaml.safe_load(f)   

    def get(self, section: str, key: str, default: Any = None) -> Any:
        """
        Get a configuration value.

        Args:
            section: The configuration section (e.g., 'LeanServerConfig')
            key: The configuration key (e.g., 'LeanServerPort')
            default: Default value if key not found

        Returns:
            The configuration value or default
        """
        if self._config is None:
            self._load_config()

        return self._config.get(section, {}).get(key, default)

    def get_api_key(self, config_dict: Dict[str, Any]) -> str:
        """
        Resolve API key from configuration.

        Args:
            config_dict: Configuration dictionary containing 'api_key' field

        Returns:
            The resolved API key value. Returns "EMPTY" for local models (api_key="LOCAL"),
            otherwise returns the value from the environment variable named in api_key field.
        """
        api_key_ref = config_dict.get("api_key", "LOCAL")

        if api_key_ref == "LOCAL":
            return "EMPTY"

        # Get the API key from environment variable
        return os.getenv(api_key_ref, "EMPTY")

    @property
    def lean_server(self):
        """Get Lean Server configuration"""
        return self._config.get("LeanServerConfig", {})

    @property
    def statement_normalization_agent(self):
        """Get Statement Normalization Agent configuration"""
        return self._config.get("StatementNormalizationAgentConfig", {})

    @property
    def autoformalization_agent(self):
        """Get Autoformalization Agent configuration"""
        return self._config.get("AutoformalizationAgentConfig", {})

    @property
    def semantic_check_agent(self):
        """Get Semantic Check Agent configuration"""
        return self._config.get("SemanticCheckAgentConfig", {})

    @property
    def statement_correction_agent(self):
        """Get Statement Correction Agent configuration"""
        return self._config.get("StatementCorrectionAgentConfig", {})

    @property
    def formalization_selection_agent(self):
        """Get Formalization Selection Agent configuration"""
        return self._config.get("FormalizationSelectionAgentConfig", {})

    @property
    def subgoal_extraction_agent(self):
        """Get Subgoal Extraction Agent configuration"""
        return self._config.get("SubgoalExtractionAgentConfig", {})

    @property
    def prover_agent(self):
        """Get Prover Agent configuration"""
        return self._config.get("ProverAgentConfig", {})

    @property
    def proof_correction_agent(self):
        """Get Proof Correction Agent configuration"""
        return self._config.get("ProofCorrectionAgentConfig", {})

    @property
    def sketch_correction_agent(self):
        """Get Sketch Correction Agent configuration"""
        return self._config.get("SketchCorrectionAgentConfig", {})

    @property
    def correctness_check_agent(self):
        """Get Correctness Check Agent configuration"""
        return self._config.get("CorrectnessCheckAgentConfig", {})

    @property
    def shallow_solve_agent(self):
        """Get Shallow Solve Agent configuration"""
        return self._config.get("ShallowSolveAgentConfig", {})

    @property
    def retrieval_agent(self):
        """Get Theorem Retrieval Agent configuration"""
        return self._config.get("RetrievalAgentConfig", {})

    @property
    def informal_proof_agent(self):
        """Get Informal Proof Agent configuration"""
        return self._config.get("InformalProofAgentConfig", {})

    @property
    def proof_sketch_agent(self):
        """Get Proof Sketch Agent configuration"""
        return self._config.get("ProofSketchAgentConfig", {})

    @property
    def proof_assembly_agent(self):
        """Get Proof Assembly Agent configuration"""
        return self._config.get("ProofAssemblyAgentConfig", {})

    @property
    def assembly_correction_agent(self):
        """Get Assembly Correction Agent configuration"""
        return self._config.get("AssemblyCorrectionAgentConfig", {})
