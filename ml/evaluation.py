"""
Comprehensive metric computation for ML Lab Agent.

Computes exact metrics for Classification, Regression, and Clustering.
The LLM never computes metrics — this deterministic module does.
"""

from __future__ import annotations

import math
from typing import Any
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray | None = None,
    positive_label: int = 1,
) -> dict[str, Any]:
    """Compute all classification metrics."""
    metrics: dict[str, Any] = {}

    metrics["f1"] = float(f1_score(y_true, y_pred, pos_label=positive_label, zero_division=0))
    metrics["precision"] = float(precision_score(y_true, y_pred, pos_label=positive_label, zero_division=0))
    metrics["recall"] = float(recall_score(y_true, y_pred, pos_label=positive_label, zero_division=0))
    metrics["accuracy"] = float(accuracy_score(y_true, y_pred))

    # PR-AUC & ROC-AUC
    if y_proba is not None:
        try:
            metrics["pr_auc"] = float(average_precision_score(y_true, y_proba, pos_label=positive_label))
        except Exception:
            metrics["pr_auc"] = None
        try:
            metrics["roc_auc"] = float(roc_auc_score(y_true, y_proba))
        except Exception:
            metrics["roc_auc"] = None
    else:
        metrics["pr_auc"] = None
        metrics["roc_auc"] = None

    # Confusion matrix
    try:
        cm = confusion_matrix(y_true, y_pred).tolist()
        metrics["confusion_matrix"] = cm
        if len(cm) == 2:
            metrics["true_negatives"] = cm[0][0]
            metrics["false_positives"] = cm[0][1]
            metrics["false_negatives"] = cm[1][0]
            metrics["true_positives"] = cm[1][1]
    except Exception:
        metrics["confusion_matrix"] = None

    try:
        metrics["classification_report"] = classification_report(y_true, y_pred, zero_division=0)
    except Exception:
        metrics["classification_report"] = ""

    return metrics


def compute_regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, Any]:
    """Compute all regression metrics."""
    metrics: dict[str, Any] = {}

    mse = float(mean_squared_error(y_true, y_pred))
    rmse = float(math.sqrt(mse))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))

    metrics["rmse"] = rmse
    metrics["mse"] = mse
    metrics["mae"] = mae
    metrics["r2"] = r2

    # Residuals
    residuals = y_true - y_pred
    metrics["residual_mean"] = float(np.mean(residuals))
    metrics["residual_std"] = float(np.std(residuals))

    return metrics


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray | None = None,
    positive_label: int = 1,
    task_type: str = "classification",
) -> dict[str, Any]:
    """
    Unified metric computer. Dispatches to classification or regression
    based on task type.
    """
    if task_type == "regression":
        return compute_regression_metrics(y_true, y_pred)
    return compute_classification_metrics(y_true, y_pred, y_proba, positive_label)


def get_primary_metric(metrics: dict[str, Any], objective: str = "f1") -> float:
    """Extract the primary objective metric value with sensible fallbacks."""
    obj = objective.lower().strip()
    val = metrics.get(obj)
    if val is not None:
        return float(val)
    # Fallback search
    for k, v in metrics.items():
        if isinstance(v, (int, float)) and v is not None:
            return float(v)
    return 0.0
