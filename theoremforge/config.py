"""
Configuration loader for TheoremForge.

This module loads configuration from config.yaml and provides easy access to configuration values.
"""
import yaml
from pathlib import Path
from typing import Any, Dict


class Config:
    """Configuration loader that reads from config.yaml"""
    
    _instance = None
    _config: Dict[str, Any] = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Config, cls).__new__(cls)
            cls._instance._load_config()
        return cls._instance
    
    def _load_config(self):
        """Load configuration from config.yaml"""
        config_path = Path(__file__).parent.parent / "config.yaml"
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        with open(config_path, 'r') as f:
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
    
    @property
    def lean_server(self):
        """Get Lean Server configuration"""
        return self._config.get('LeanServerConfig', {})
    
    @property
    def prover_agent(self):
        """Get Prover Agent configuration"""
        return self._config.get('ProverAgentConfig', {})
    
    @property
    def informal_proof_agent(self):
        """Get Informal Proof Agent configuration"""
        return self._config.get('InformalProofAgentConfig', {})
    
    @property
    def decomposition_agent(self):
        """Get Decomposition Agent configuration"""
        return self._config.get('DecompositionAgentConfig', {})

    
    @property
    def theorem_retriever_agent(self):
        """Get Theorem Retriever Agent configuration"""
        return self._config.get('TheoremRetrieverAgentConfig', {})


# Global config instance
config = Config()


