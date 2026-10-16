"""
Baseline training tool for ML Lab Agent.

Trains an un-tuned benchmark model (Dummy, linear model, or Ridge)
to establish the benchmark that subsequent experiments must outperform.
"""

from __future__ import annotations

import time
import uuid
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from agent.schemas import ExperimentResult
from ml.evaluation import compute_metrics
from ml.model_registry import instantiate_model
from ml.preprocessing import create_preprocessing_pipeline, identify_feature_types
from ml.validation import DataPartitions


def train_baseline(
    partitions: DataPartitions,
    model_family: str | None = None,
    task_type: str = "classification",
    objective_metric: str = "f1",
    hypothesis: str = "Establish initial performance baseline.",
    reason: str = "Benchmark against which all subsequent models will be measured.",
) -> tuple[ExperimentResult, Pipeline]:
    """
    Train an un-tuned baseline model and evaluate on the validation set.

    Args:
        partitions: Clean train/val/test data partitions
        model_family: Optional baseline family (defaults to LogisticRegression or Ridge)
        task_type: 'classification' or 'regression'
        objective_metric: Target metric to measure
        hypothesis: Why this baseline is being run
        reason: Justification

    Returns:
        tuple of (ExperimentResult, trained Pipeline)
    """
    t0 = time.perf_counter()
    run_id = f"baseline_{uuid.uuid4().hex[:6]}"

    if model_family is None:
        model_family = "Ridge" if task_type == "regression" else "LogisticRegression"

    X_train, y_train = partitions.train_X, partitions.train_y
    X_val, y_val = partitions.val_X, partitions.val_y

    feature_types = identify_feature_types(X_train)
    preprocessor = create_preprocessing_pipeline(feature_types, model_family)

    model = instantiate_model(model_family)
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("estimator", model),
    ])

    pipeline.fit(X_train, y_train)

    # Evaluate on validation
    val_pred = pipeline.predict(X_val)
    val_proba = None
    if task_type == "classification" and hasattr(pipeline, "predict_proba"):
        try:
            val_proba = pipeline.predict_proba(X_val)[:, 1]
        except Exception:
            val_proba = None

    y_val_np = y_val.to_numpy()
    val_metrics = compute_metrics(
        y_val_np, val_pred, val_proba, positive_label=1, task_type=task_type
    )

    # Evaluate on training (overfitting check)
    train_pred = pipeline.predict(X_train)
    y_train_np = y_train.to_numpy()
    train_metrics = compute_metrics(
        y_train_np, train_pred, None, positive_label=1, task_type=task_type
    )

    duration = time.perf_counter() - t0
    target_score = val_metrics.get(objective_metric)
    if target_score is None:
        target_score = val_metrics.get("f1", 0.0)

    result = ExperimentResult(
        run_id=run_id,
        experiment_number=0,
        status="completed",
        model_family=model_family,
        task_type=task_type,
        primary_metric_name=objective_metric,
        primary_metric_value=target_score,
        metrics=val_metrics,
        training_metrics=train_metrics,
        validation_f1=val_metrics.get("f1"),
        validation_precision=val_metrics.get("precision"),
        validation_recall=val_metrics.get("recall"),
        validation_accuracy=val_metrics.get("accuracy"),
        validation_pr_auc=val_metrics.get("pr_auc"),
        training_f1=train_metrics.get("f1"),
        training_accuracy=train_metrics.get("accuracy"),
        confusion_matrix=val_metrics.get("confusion_matrix"),
        duration_seconds=round(duration, 3),
        parameters=getattr(model, "get_params", lambda: {})(),
        hypothesis=hypothesis,
        reason=reason,
    )

    return result, pipeline
