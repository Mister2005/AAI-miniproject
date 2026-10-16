"""
Prompt templates for ML Lab Agent.

Defines the system instructions, structured output contract,
and dynamic context assembly for the LLM experiment planner.
Adapts dynamically to Classification, Regression, and Clustering.
"""

from __future__ import annotations

import json
from typing import Any

from agent.schemas import ALLOWED_MODEL_FAMILIES, CampaignState, DatasetProfile, ExperimentResult
from ml.model_registry import list_families


def get_system_prompt(task_type: str = "classification", objective_metric: str = "f1") -> str:
    """Generate dynamic system prompt tailored to the active task type and metric."""
    families = list_families(task_type)
    lower_better = objective_metric.lower() in ["rmse", "mse", "mae", "log_loss", "davies_bouldin"]
    direction = "MINIMIZE" if lower_better else "MAXIMIZE"

    return f"""You are the ML Lab Agent — an autonomous machine learning researcher.
Your task is to iteratively formulate scientific hypotheses, select diverse model families, tune hyperparameters, diagnose prediction errors, and optimize the objective metric on a given dataset within a fixed budget.

CURRENT TASK CONTEXT:
- Task Type: {task_type.upper()}
- Target Metric to {direction}: {objective_metric}
- Eligible Model Families: {families}

STRICT OPERATIONAL RULES:
1. You NEVER compute metrics or write raw ML code directly. Deterministic Python tools execute your decisions and return exact metrics.
2. Every proposal MUST specify:
   - "action": One of ["run_experiment", "analyze_errors", "compare_experiments", "finish_campaign"]
   - "model_family": One of {families} (required for run_experiment, null otherwise)
   - "hypothesis": What scientific question or improvement is being tested (e.g., "Exploring ExtraTrees to improve boundary robustness")
   - "parameters": Hyperparameter dictionary (optional, or empty dict to let Optuna search)
   - "reason": Why this step is justified based on past findings and data profile
3. Output MUST be ONLY a single valid JSON object matching the AgentAction schema. No markdown outside ```json```, no conversational greeting.
4. Stop when:
   - You have explored diverse model families and performance has plateaued
   - You reach the target metric
   - Or when budget is exhausted
"""


def build_agent_context(
    state: CampaignState,
    recent_error_analysis: dict[str, Any] | None = None,
) -> list[dict[str, str]]:
    """
    Construct prompt for the LLM based on current campaign state.

    Args:
        state: Active CampaignState
        recent_error_analysis: Optional latest error analysis observations

    Returns:
        List of message dicts formatted for chat completion
    """
    profile: DatasetProfile | None = state.dataset_profile
    best: ExperimentResult | None = state.current_best
    completed = state.get_completed_experiments()
    metric_name = state.config.objective_metric
    task_type = state.config.task_type if state.config.task_type != "auto" else "classification"

    lower_better = metric_name.lower() in ["rmse", "mse", "mae", "log_loss", "davies_bouldin"]
    direction = "minimize" if lower_better else "maximize"

    context_lines = [
        f"### Campaign Status",
        f"- Campaign ID: {state.campaign_id}",
        f"- Task Type: {task_type}",
        f"- Target Metric: {metric_name} ({direction})",
        f"- Budget: {len(state.experiments)} / {state.config.max_experiments} experiments used",
        f"- Time Remaining: {state.time_remaining_seconds() / 60:.1f} minutes",
    ]

    if state.config.user_guidance:
        context_lines.append(f"- User Guidance: {state.config.user_guidance}")

    if profile:
        context_lines.extend([
            "",
            "### Dataset Profile",
            f"- Rows: {profile.n_rows}, Columns: {profile.n_columns}",
            f"- Target Column: '{profile.target_column}'",
            f"- Feature Types: {len(profile.feature_types)} columns classified",
        ])
        if profile.class_imbalance_ratio > 1.2:
            context_lines.append(f"- Class Imbalance Ratio: {profile.class_imbalance_ratio:.2f}:1")
        if profile.potential_leakage_columns:
            context_lines.append(f"- Potential Leakage Columns: {profile.potential_leakage_columns}")
        if profile.warnings:
            context_lines.append(f"- Data Warnings: {'; '.join(profile.warnings[:3])}")

    if best:
        best_val = state.get_score(best)
        best_val_str = f"{best_val:.4f}" if best_val is not None else "N/A"
        context_lines.extend([
            "",
            f"### Current Champion Model",
            f"- Best Model: {best.model_family} (Run ID: {best.run_id})",
            f"- Champion {metric_name}: {best_val_str}",
            f"- Best Parameters: {best.parameters}",
        ])

    if completed:
        context_lines.extend([
            "",
            "### Completed Experiments History",
        ])
        for exp in completed:
            val = state.get_score(exp)
            val_str = f"{val:.4f}" if val is not None else "N/A"
            context_lines.append(
                f"- Exp #{exp.experiment_number} ({exp.model_family}): "
                f"{metric_name} = {val_str}. Hypothesis: '{exp.hypothesis}'"
            )

    if recent_error_analysis and "observations" in recent_error_analysis:
        context_lines.extend([
            "",
            "### Diagnostic Error Analysis",
            *[f"- {obs}" for obs in recent_error_analysis["observations"]],
        ])

    context_lines.extend([
        "",
        "### Your Task",
        f"Analyze findings above and decide the next optimal action to {direction} {metric_name}.",
        "Explore diverse model families (e.g. linear, tree, boosting, ensemble) or tune parameters.",
        "Respond with a valid JSON object matching the AgentAction schema.",
    ])

    user_prompt = "\n".join(context_lines)
    system_prompt = get_system_prompt(task_type, metric_name)

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


SYSTEM_PROMPT = get_system_prompt()
