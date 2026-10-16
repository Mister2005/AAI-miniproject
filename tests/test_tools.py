"""
Unit tests for deterministic tools.
"""

import pandas as pd
from sklearn.datasets import load_breast_cancer

from ml.preprocessing import identify_feature_types
from ml.validation import create_partitions
from tools.compare_experiments import compare_experiments
from tools.error_analysis import analyze_errors
from tools.evaluate_on_test import evaluate_on_test
from tools.profile_dataset import profile_dataset
from tools.run_hpo_iteration import run_hpo_iteration
from tools.train_baseline import train_baseline


def get_sample_partitions():
    raw = load_breast_cancer(as_frame=True)
    df = raw.frame
    return create_partitions(df, target_column="target", random_seed=42), df


def test_profile_dataset():
    _, df = get_sample_partitions()
    profile = profile_dataset(df, target_column="target")
    assert profile.n_rows == len(df)
    assert profile.n_columns == len(df.columns)
    assert profile.class_imbalance_ratio >= 1.0


def test_train_baseline():
    partitions, _ = get_sample_partitions()
    result, pipeline = train_baseline(partitions, model_family="LogisticRegression")
    assert result.status == "completed"
    assert result.validation_f1 is not None
    assert result.validation_f1 > 0.8
    assert pipeline is not None


def test_run_hpo_iteration():
    partitions, _ = get_sample_partitions()
    result, pipeline = run_hpo_iteration(
        partitions=partitions,
        model_family="RandomForestClassifier",
        hypothesis="Testing RF depth",
        reason="Check ensemble",
        experiment_number=1,
        n_trials=2,
    )
    assert result.status == "completed"
    assert result.validation_f1 is not None
    assert pipeline is not None


def test_error_analysis():
    partitions, _ = get_sample_partitions()
    _, pipeline = train_baseline(partitions, model_family="LogisticRegression")
    analysis = analyze_errors(pipeline, partitions)
    assert analysis.total_predictions == len(partitions.val_y)
    assert analysis.true_positives >= 0


def test_evaluate_on_test():
    partitions, _ = get_sample_partitions()
    _, pipeline = train_baseline(partitions, model_family="LogisticRegression")
    eval_out = evaluate_on_test(pipeline, partitions, campaign_id="unit_test")
    assert "test_metrics" in eval_out
    assert eval_out["test_metrics"]["f1"] > 0.8
    assert "model_artifact_path" in eval_out
