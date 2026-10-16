"""
Unit tests for data contracts and schemas.
"""

import pytest
from pydantic import ValidationError

from agent.schemas import (
    ALLOWED_MODEL_FAMILIES,
    AgentAction,
    CampaignConfig,
    ExperimentResult,
)


def test_valid_campaign_config():
    cfg = CampaignConfig(
        dataset_path="breast_cancer",
        target_column="target",
        objective_metric="f1",
        max_experiments=5,
    )
    assert cfg.max_experiments == 5
    assert cfg.objective_metric == "f1"


def test_invalid_campaign_metric():
    with pytest.raises(ValidationError):
        CampaignConfig(
            dataset_path="test",
            target_column="target",
            objective_metric="invalid_metric_xyz",
        )


def test_valid_agent_action():
    action = AgentAction(
        action="run_experiment",
        model_family="RandomForestClassifier",
        hypothesis="Testing tree ensemble depth",
        reason="Imbalance requires robust non-linear boundaries",
    )
    assert action.action == "run_experiment"
    assert action.model_family == "RandomForestClassifier"


def test_invalid_model_family_in_action():
    with pytest.raises(ValidationError):
        AgentAction(
            action="run_experiment",
            model_family="ArbitraryNeuralNetXYZ",
            hypothesis="Testing unapproved family",
            reason="Should fail validation",
        )


def test_experiment_result_champion_tracking():
    exp1 = ExperimentResult(
        run_id="trial_1",
        model_family="LogisticRegression",
        validation_f1=0.75,
        status="completed",
    )
    assert exp1.validation_f1 == 0.75
