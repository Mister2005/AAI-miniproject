"""
Natural Language Intent Parser for ML Lab Agent.

Translates conversational user instructions into concrete, validated
ML campaign configurations, task types, target columns, and metrics.
"""

from __future__ import annotations

import logging
from typing import Any, Literal
import pandas as pd
from pydantic import BaseModel, Field

from agent.llm_client import LLMClient
from agent.schemas import CampaignConfig

logger = logging.getLogger(__name__)


class ParsedUserIntent(BaseModel):
    """Structured understanding extracted from user's conversational prompt."""
    task_type: Literal["classification", "regression", "clustering"] = "classification"
    target_column: str | None = None
    objective_metric: str = "f1"
    max_experiments: int = Field(default=8, ge=2, le=30)
    max_time_minutes: int = Field(default=15, ge=1, le=120)
    user_guidance: str = ""
    agent_reply: str = Field(description="Conversational response acknowledging instructions and outlining strategy.")


def parse_user_prompt(
    user_prompt: str,
    df: pd.DataFrame | None = None,
    llm_client: LLMClient | None = None,
    default_config: CampaignConfig | None = None,
) -> tuple[CampaignConfig, str]:
    """
    Parse a user's natural language request in the context of an uploaded/selected dataset.

    Args:
        user_prompt: Freeform text prompt from the user
        df: Optional dataframe for column matching and task inference
        llm_client: LLM client instance
        default_config: Fallback default configuration

    Returns:
        tuple of (CampaignConfig, agent_reply_string)
    """
    if llm_client is None:
        llm_client = LLMClient()

    cols_info = ""
    target_candidate = None
    inferred_task = "classification"
    default_metric = "f1"

    if df is not None:
        columns = df.columns.tolist()
        cols_info = f"Columns available in dataset: {columns}\n"
        sample_types = {col: str(df[col].dtype) for col in columns[:15]}
        cols_info += f"Data types: {sample_types}\n"

        # Heuristic target detection
        for candidate in ["target", "y", "label", "price", "SalePrice", "churn", "class", "output", "survived"]:
            matches = [c for c in columns if c.lower() == candidate.lower()]
            if matches:
                target_candidate = matches[0]
                break
        if target_candidate is None and columns:
            target_candidate = columns[-1]

        # Heuristic task detection if target is found
        if target_candidate and target_candidate in df.columns:
            s = df[target_candidate].dropna()
            if pd.api.types.is_numeric_dtype(s) and s.nunique() > 20:
                inferred_task = "regression"
                default_metric = "rmse"
            else:
                inferred_task = "classification"
                default_metric = "f1"

    system_instruction = f"""You are the ML Lab Agent intent parser.
Your job is to read the user's natural language request and dataset columns, then output a JSON object with:
- "task_type": "classification", "regression", or "clustering"
- "target_column": exact column name from available columns (or null for clustering)
- "objective_metric": chosen evaluation metric (e.g. for classification: f1, pr_auc, roc_auc, accuracy, recall, precision; for regression: rmse, r2, mae; for clustering: silhouette)
- "max_experiments": number between 2 and 30 (default: 8, or whatever the user requested)
- "max_time_minutes": time budget in minutes (default: 15, or whatever user requested)
- "user_guidance": concise summary of user preferences (e.g., 'explore diverse models, prioritize recall')
- "agent_reply": a helpful, professional 2-3 sentence response summarizing your understanding and the strategy you are initiating.

Dataset Context:
{cols_info}
Inferred Target Default: {target_candidate}
Inferred Task Default: {inferred_task}
Inferred Metric Default: {default_metric}
"""

    messages = [
        {"role": "system", "content": system_instruction},
        {"role": "user", "content": f"User Request: {user_prompt}"},
    ]

    try:
        parsed, resp = llm_client.call_structured(messages, ParsedUserIntent, max_repair_attempts=2)
        target = parsed.target_column or target_candidate or ""
        metric = parsed.objective_metric or default_metric

        config = CampaignConfig(
            dataset_path=default_config.dataset_path if default_config else "custom_dataset",
            target_column=target,
            task_type=parsed.task_type,
            objective_metric=metric,
            max_experiments=parsed.max_experiments,
            max_time_minutes=parsed.max_time_minutes,
            user_prompt=user_prompt,
            user_guidance=parsed.user_guidance,
        )
        return config, parsed.agent_reply

    except Exception as e:
        logger.warning(f"NLP intent parse fallback triggered: {e}")
        target = target_candidate or "target"
        config = CampaignConfig(
            dataset_path=default_config.dataset_path if default_config else "custom_dataset",
            target_column=target,
            task_type=inferred_task,
            objective_metric=default_metric,
            max_experiments=8,
            max_time_minutes=15,
            user_prompt=user_prompt,
            user_guidance="Standard autonomous exploration",
        )
        reply = (
            f"Understood! I will analyze the dataset targeting `{target}` using a `{inferred_task}` workflow "
            f"optimized for `{default_metric}` across a budget of 8 experiments."
        )
        return config, reply
