"""
Error analysis tool for ML Lab Agent.

Inspects model prediction errors on the validation partition, computes
confusion matrix breakdown, identifies problematic sample indices, and
extracts patterns to inform subsequent agent hypotheses.
"""

from __future__ import annotations

from typing import Any
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import Pipeline

from agent.schemas import ErrorAnalysis
from ml.validation import DataPartitions


def analyze_errors(
    pipeline: Pipeline,
    partitions: DataPartitions,
    max_sample_indices: int = 10,
) -> ErrorAnalysis:
    """
    Perform deep error analysis on the validation partition.

    Args:
        pipeline: Trained sklearn Pipeline
        partitions: Clean train/val/test data partitions
        max_sample_indices: How many false positive/negative row indices to return

    Returns:
        ErrorAnalysis schema with actionable diagnosis
    """
    X_val = partitions.val_X
    y_val = partitions.val_y.to_numpy()

    y_pred = pipeline.predict(X_val)

    # Confusion matrix elements: [[TN, FP], [FN, TP]]
    cm = confusion_matrix(y_val, y_pred, labels=[0, 1])
    tn, fp, fn, tp = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])

    # Indices of errors
    fp_indices = np.where((y_pred == 1) & (y_val == 0))[0][:max_sample_indices].tolist()
    fn_indices = np.where((y_pred == 0) & (y_val == 1))[0][:max_sample_indices].tolist()

    report_str = classification_report(y_val, y_pred, labels=[0, 1], zero_division=0)

    observations: list[str] = []
    total = len(y_val)

    if fn > fp * 2:
        observations.append(
            f"High False Negative rate: {fn} missed positive cases vs {fp} false alarms. "
            "Model is too conservative. Consider class_weight='balanced' or lowering decision threshold."
        )
    elif fp > fn * 2:
        observations.append(
            f"High False Positive rate: {fp} false alarms vs {fn} missed positives. "
            "Model is over-predicting the positive class. Consider stronger regularization."
        )

    if fn == 0 and tp == 0:
        observations.append(
            "Model predicted all negative class (0 TP, 0 FP). The classifier collapsed to majority class."
        )

    # Check feature anomalies on False Negatives if numeric features exist
    numeric_cols = X_val.select_dtypes(include=[np.number]).columns.tolist()
    if numeric_cols and len(fn_indices) > 0:
        try:
            fn_rows = X_val.iloc[fn_indices]
            tp_indices = np.where((y_pred == 1) & (y_val == 1))[0]
            if len(tp_indices) > 0:
                tp_rows = X_val.iloc[tp_indices]
                for col in numeric_cols[:3]:
                    fn_mean = fn_rows[col].mean()
                    tp_mean = tp_rows[col].mean()
                    if abs(fn_mean - tp_mean) > 1e-3:
                        observations.append(
                            f"Feature '{col}' mean in FN is {fn_mean:.2f} vs {tp_mean:.2f} in TP."
                        )
        except Exception:
            pass

    return ErrorAnalysis(
        total_predictions=total,
        true_positives=tp,
        true_negatives=tn,
        false_positives=fp,
        false_negatives=fn,
        classification_report=report_str,
        confusion_matrix=cm.tolist(),
        fp_sample_indices=fp_indices,
        fn_sample_indices=fn_indices,
        observations=observations,
    )
