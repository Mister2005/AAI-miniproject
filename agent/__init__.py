"""
Agent package for ML Lab Agent.
"""

from agent.llm_client import LLMClient
from agent.prompt_templates import SYSTEM_PROMPT, build_agent_context
from agent.schemas import (
    ALLOWED_MODEL_FAMILIES,
    AgentAction,
    CampaignConfig,
    CampaignReport,
    CampaignState,
    DatasetProfile,
    ExperimentResult,
)


def __getattr__(name: str):
    """Lazy import for AgentOrchestrator to prevent circular imports."""
    if name == "AgentOrchestrator":
        from agent.orchestrator import AgentOrchestrator
        return AgentOrchestrator
    raise AttributeError(f"module 'agent' has no attribute '{name}'")


__all__ = [
    "AgentOrchestrator",
    "LLMClient",
    "CampaignConfig",
    "CampaignState",
    "CampaignReport",
    "AgentAction",
    "ExperimentResult",
    "DatasetProfile",
    "SYSTEM_PROMPT",
    "build_agent_context",
    "ALLOWED_MODEL_FAMILIES",
]
