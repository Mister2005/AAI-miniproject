"""
Hyperparameter optimization and model training tool for ML Lab Agent.

Trains a candidate model family within pre-approved search spaces.
Optuna or the parameter spec determines values; metrics are computed
on the validation set for either classification or regression tasks.
"""

from __future__ import annotations

import logging
import time
import uuid
from typing import Any
import numpy as np
import optuna
from sklearn.pipeline import Pipeline

from agent.schemas import ExperimentResult
from ml.evaluation import compute_metrics
from ml.model_registry import instantiate_model, validate_model_family
from ml.preprocessing import create_preprocessing_pipeline, identify_feature_types
from ml.search_spaces import suggest_params
from ml.validation import DataPartitions

logger = logging.getLogger(__name__)
optuna.logging.set_verbosity(optuna.logging.WARNING)


def run_hpo_iteration(
    partitions: DataPartitions,
    model_family: str,
    hypothesis: str,
    reason: str,
    experiment_number: int,
    parameters: dict[str, Any] | None = None,
    n_trials: int = 3,
    objective_metric: str = "f1",
    task_type: str = "classification",
) -> tuple[ExperimentResult, Pipeline | None]:
    """
    Train a candidate model on train set and evaluate on validation set.

    Args:
        partitions: Clean train/val/test data partitions
        model_family: Approved model family name
        hypothesis: What this experiment tests
        reason: Why this experiment is justified
        experiment_number: Sequential experiment index
        parameters: Optional explicit parameters
        n_trials: Number of Optuna trials if optimizing
        objective_metric: Primary metric to optimize
        task_type: 'classification' or 'regression'

    Returns:
        tuple of (ExperimentResult, fitted Pipeline or None if failed)
    """
    t0 = time.perf_counter()
    run_id = f"exp_{experiment_number:02d}_{uuid.uuid4().hex[:4]}"

    try:
        validate_model_family(model_family)
    except Exception as e:
        return ExperimentResult(
            run_id=run_id,
            experiment_number=experiment_number,
            status="failed",
            model_family=model_family,
            task_type=task_type,
            primary_metric_name=objective_metric,
            primary_metric_value=0.0,
            error_message=f"Invalid model family: {e}",
            duration_seconds=round(time.perf_counter() - t0, 3),
            hypothesis=hypothesis,
            reason=reason,
        ), None

    X_train, y_train = partitions.train_X, partitions.train_y
    X_val, y_val = partitions.val_X, partitions.val_y
    y_train_np = y_train.to_numpy()
    y_val_np = y_val.to_numpy()

    feature_types = identify_feature_types(X_train)

    best_pipeline: Pipeline | None = None
    best_params: dict[str, Any] = {}
    best_val_metrics: dict[str, Any] = {}
    best_train_metrics: dict[str, Any] = {}

    lower_better = objective_metric.lower() in [
        "rmse", "mse", "mae", "log_loss", "davies_bouldin"
    ]

    try:
        # If parameters provided directly by LLM, try direct execution
        if parameters and len(parameters) > 0:
            preprocessor = create_preprocessing_pipeline(feature_types, model_family)
            model = instantiate_model(model_family, **parameters)
            pipeline = Pipeline([
                ("preprocessor", preprocessor),
                ("estimator", model),
            ])
            pipeline.fit(X_train, y_train)

            val_pred = pipeline.predict(X_val)
            val_proba = None
            if task_type == "classification" and hasattr(pipeline, "predict_proba"):
                try:
                    val_proba = pipeline.predict_proba(X_val)[:, 1]
                except Exception:
                    pass
            val_metrics = compute_metrics(y_val_np, val_pred, val_proba, task_type=task_type)

            train_pred = pipeline.predict(X_train)
            train_metrics = compute_metrics(y_train_np, train_pred, None, task_type=task_type)

            best_pipeline = pipeline
            best_params = parameters
            best_val_metrics = val_metrics
            best_train_metrics = train_metrics

        else:
            # Optuna-guided search
            def objective(trial: optuna.Trial) -> float:
                suggested = suggest_params(trial, model_family)
                preprocessor = create_preprocessing_pipeline(feature_types, model_family)
                model = instantiate_model(model_family, **suggested)
                pipe = Pipeline([
                    ("preprocessor", preprocessor),
                    ("estimator", model),
                ])
                pipe.fit(X_train, y_train)

                val_pred = pipe.predict(X_val)
                val_proba = None
                if task_type == "classification" and hasattr(pipe, "predict_proba"):
                    try:
                        val_proba = pipe.predict_proba(X_val)[:, 1]
                    except Exception:
                        pass
                m = compute_metrics(y_val_np, val_pred, val_proba, task_type=task_type)
                score = m.get(objective_metric)
                if score is None:
                    score = m.get("f1") or m.get("rmse") or 0.0
                trial.set_user_attr("metrics", m)
                trial.set_user_attr("pipeline", pipe)
                return float(score)

            study = optuna.create_study(
                direction="minimize" if lower_better else "maximize"
            )
            study.optimize(objective, n_trials=n_trials, timeout=120)

            best_trial = study.best_trial
            best_params = best_trial.params
            best_val_metrics = best_trial.user_attrs.get("metrics", {})
            best_pipeline = best_trial.user_attrs.get("pipeline")

            if best_pipeline is None:
                preprocessor = create_preprocessing_pipeline(feature_types, model_family)
                model = instantiate_model(model_family, **best_params)
                best_pipeline = Pipeline([
                    ("preprocessor", preprocessor),
                    ("estimator", model),
                ])
                best_pipeline.fit(X_train, y_train)

            train_pred = best_pipeline.predict(X_train)
            best_train_metrics = compute_metrics(y_train_np, train_pred, None, task_type=task_type)

        duration = time.perf_counter() - t0
        primary_val = best_val_metrics.get(objective_metric)
        if primary_val is None:
            primary_val = best_val_metrics.get("f1") or best_val_metrics.get("rmse") or 0.0

        result = ExperimentResult(
            run_id=run_id,
            experiment_number=experiment_number,
            status="completed",
            model_family=model_family,
            task_type=task_type,
            primary_metric_name=objective_metric,
            primary_metric_value=float(primary_val),
            metrics=best_val_metrics,
            training_metrics=best_train_metrics,
            validation_f1=best_val_metrics.get("f1"),
            validation_precision=best_val_metrics.get("precision"),
            validation_recall=best_val_metrics.get("recall"),
            validation_accuracy=best_val_metrics.get("accuracy"),
            validation_pr_auc=best_val_metrics.get("pr_auc"),
            training_f1=best_train_metrics.get("f1"),
            training_accuracy=best_train_metrics.get("accuracy"),
            confusion_matrix=best_val_metrics.get("confusion_matrix"),
            duration_seconds=round(duration, 3),
            parameters=best_params,
            hypothesis=hypothesis,
            reason=reason,
        )
        return result, best_pipeline

    except Exception as e:
        logger.error(f"Experiment {experiment_number} failed: {e}", exc_info=True)
        return ExperimentResult(
            run_id=run_id,
            experiment_number=experiment_number,
            status="failed",
            model_family=model_family,
            task_type=task_type,
            primary_metric_name=objective_metric,
            primary_metric_value=0.0,
            metrics={},
            training_metrics={},
            error_message=str(e),
            duration_seconds=round(time.perf_counter() - t0, 3),
            hypothesis=hypothesis,
            reason=reason,
        ), None
