"""
Experiment comparison tool for ML Lab Agent.

Compares completed experiment runs, identifies the current champion,
computes progress deltas against the baseline, and detects plateaus.
Supports both higher-is-better and lower-is-better metrics.
"""

from __future__ import annotations

from typing import Any
from agent.schemas import ExperimentComparison, ExperimentResult


def compare_experiments(
    experiments: list[ExperimentResult],
    metric_name: str = "f1",
) -> ExperimentComparison:
    """
    Compare a list of experiment results and summarize current standings.

    Args:
        experiments: List of ExperimentResult objects
        metric_name: Primary metric for ranking (e.g. 'f1', 'pr_auc', 'rmse', 'r2')

    Returns:
        ExperimentComparison schema with summary and champion run ID
    """
    completed = [e for e in experiments if e.status == "completed"]
    if not completed:
        return ExperimentComparison(
            experiments=[],
            best_run_id="",
            best_metric_value=0.0,
            metric_name=metric_name,
            summary="No completed experiments available to compare.",
        )

    clean_metric = metric_name.lower().replace("validation_", "").strip()
    lower_better = clean_metric in ["rmse", "mse", "mae", "log_loss", "davies_bouldin"]

    def get_metric(exp: ExperimentResult) -> float:
        if exp.primary_metric_value is not None and exp.primary_metric_name.lower() == clean_metric:
            return float(exp.primary_metric_value)
        if clean_metric in exp.metrics:
            val = exp.metrics[clean_metric]
            if val is not None:
                return float(val)
        val = getattr(exp, f"validation_{clean_metric}", None)
        if val is not None:
            return float(val)
        val = getattr(exp, clean_metric, None)
        if val is not None:
            return float(val)
        return float(exp.primary_metric_value or 0.0)

    sorted_exps = sorted(completed, key=get_metric, reverse=not lower_better)
    best_exp = sorted_exps[0]
    best_val = get_metric(best_exp)

    # Check baseline (first experiment)
    baseline_val = get_metric(completed[0])
    delta_vs_baseline = (baseline_val - best_val) if lower_better else (best_val - baseline_val)

    rows = []
    for exp in completed:
        m_val = get_metric(exp)
        rows.append({
            "run_id": exp.run_id,
            "experiment_number": exp.experiment_number,
            "model_family": exp.model_family,
            clean_metric: m_val,
            "duration_s": exp.duration_seconds,
            "is_best": exp.run_id == best_exp.run_id,
            "parameters": exp.parameters,
        })

    direction_str = "lower" if lower_better else "higher"
    summary_text = (
        f"Compared {len(completed)} experiments optimizing {clean_metric} ({direction_str} is better). "
        f"Champion is {best_exp.model_family} ({best_exp.run_id}) with {clean_metric}={best_val:.4f} "
        f"({'+' if delta_vs_baseline >= 0 else ''}{delta_vs_baseline:.4f} vs baseline)."
    )

    return ExperimentComparison(
        experiments=rows,
        best_run_id=best_exp.run_id,
        best_metric_value=best_val,
        metric_name=clean_metric,
        summary=summary_text,
    )
