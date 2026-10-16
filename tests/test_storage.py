"""
Unit tests for storage and persistence.
"""

import os
import shutil
from agent.schemas import CampaignConfig, CampaignReport, CampaignState, ExperimentResult
from storage.run_store import RunStore


def test_run_store_lifecycle(tmp_path):
    store_dir = str(tmp_path / "exp_test")
    store = RunStore(base_dir=store_dir)

    cfg = CampaignConfig(dataset_path="test_ds", target_column="y", campaign_id="store_test")
    state = CampaignState(campaign_id="store_test", config=cfg)
    exp = ExperimentResult(
        run_id="trial_01",
        experiment_number=1,
        model_family="LogisticRegression",
        validation_f1=0.88,
        status="completed",
    )
    state.add_experiment(exp)

    store.save_campaign_state(state)

    # Reload
    loaded = store.load_campaign_state("store_test")
    assert loaded is not None
    assert loaded.campaign_id == "store_test"
    assert len(loaded.experiments) == 1
    assert loaded.experiments[0].validation_f1 == 0.88

    # Check campaigns query
    campaigns = store.list_campaigns()
    assert len(campaigns) >= 1
    assert campaigns[0]["campaign_id"] == "store_test"
