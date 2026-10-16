"""
Integration tests for NLP parser, dynamic task detection, PR-AUC and RMSE metrics.
"""

import pytest
import pandas as pd
import numpy as np
from agent.schemas import CampaignConfig, CampaignState, ExperimentResult
from agent.nlp_parser import parse_user_prompt, ParsedUserIntent
from ml.evaluation import compute_classification_metrics, compute_regression_metrics
from ml.model_registry import sanitize_params


def test_nlp_parser_heuristic_fallback():
    # Test fallback without calling live API in unit test
    df = pd.DataFrame({
        "customer_id": [1, 2, 3, 4],
        "tenure": [10, 20, 30, 40],
        "churn": [0, 1, 0, 1]
    })
    
    # Passing an invalid LLM client to force the robust fallback
    class MockFailingClient:
        def call_structured(self, *args, **kwargs):
            raise ConnectionError("Mock offline")

    config, reply = parse_user_prompt(
        "Focus on churn and run 10 experiments",
        df=df,
        llm_client=MockFailingClient()
    )
    
    assert config.target_column == "churn"
    assert config.task_type == "classification"
    assert "churn" in reply


def test_sanitize_params_extra_trees_auto():
    # Verify that the scikit-learn 1.4+ max_features='auto' bug is intercepted and sanitized
    dirty = {"n_estimators": 100, "max_features": "auto", "max_depth": 10}
    clean = sanitize_params("ExtraTreesClassifier", dirty)
    assert clean["max_features"] == "sqrt"
    assert clean["n_estimators"] == 100


def test_classification_pr_auc_metric_computation():
    y_true = np.array([0, 1, 0, 1, 1, 0, 0, 1])
    y_pred = np.array([0, 1, 0, 1, 0, 0, 0, 1])
    y_prob = np.array([0.1, 0.9, 0.2, 0.8, 0.4, 0.1, 0.3, 0.7])
    
    metrics = compute_classification_metrics(y_true, y_pred, y_prob)
    assert "pr_auc" in metrics
    assert "f1" in metrics
    assert "roc_auc" in metrics
    assert 0.0 <= metrics["pr_auc"] <= 1.0


def test_regression_metric_computation():
    y_true = np.array([10.0, 20.0, 30.0, 40.0])
    y_pred = np.array([11.0, 19.0, 32.0, 39.0])
    
    metrics = compute_regression_metrics(y_true, y_pred)
    assert "rmse" in metrics
    assert "mae" in metrics
    assert "r2" in metrics
    assert metrics["rmse"] > 0.0
    assert metrics["r2"] > 0.9


def test_champion_tracking_direction_awareness():
    # For higher-is-better (pr_auc)
    cfg_prauc = CampaignConfig(dataset_path="test", target_column="y", objective_metric="pr_auc")
    state_prauc = CampaignState(campaign_id="test1", config=cfg_prauc)
    assert state_prauc.is_better(0.85, 0.80) is True
    assert state_prauc.is_better(0.75, 0.80) is False

    # For lower-is-better (rmse)
    cfg_rmse = CampaignConfig(dataset_path="test", target_column="y", objective_metric="rmse")
    state_rmse = CampaignState(campaign_id="test2", config=cfg_rmse)
    assert state_rmse.is_better(2.1, 3.5) is True
    assert state_rmse.is_better(4.0, 3.5) is False

