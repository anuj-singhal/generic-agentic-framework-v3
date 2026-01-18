"""
Configuration module for the Agentic Framework.
Handles environment variables and settings.
"""

import os
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ModelConfig:
    """Configuration for the LLM model (The Brain)."""
    model_name: str = "gpt-4o-mini"
    temperature: float = 0.7
    max_tokens: int = 4096
    api_key: Optional[str] = None
    
    def __post_init__(self):
        if self.api_key is None:
            self.api_key = os.getenv("OPENAI_API_KEY")


@dataclass
class FrameworkConfig:
    """Main configuration for the agentic framework."""
    model: ModelConfig = field(default_factory=ModelConfig)
    max_iterations: int = 10  # Maximum ReAct loops
    verbose: bool = True
    memory_enabled: bool = True
    

def get_config() -> FrameworkConfig:
    """Get the framework configuration."""
    return FrameworkConfig(
        model=ModelConfig(
            model_name=os.getenv("MODEL_NAME", "gpt-4o-mini"),
            temperature=float(os.getenv("TEMPERATURE", "0.7")),
            api_key=os.getenv("OPENAI_API_KEY")
        ),
        max_iterations=int(os.getenv("MAX_ITERATIONS", "10")),
        verbose=os.getenv("VERBOSE", "true").lower() == "true"
    )
