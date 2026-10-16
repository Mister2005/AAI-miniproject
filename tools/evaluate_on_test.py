"""
Final test evaluation tool for ML Lab Agent.

STRICT RULE: This tool is touched exactly ONCE at the conclusion of a campaign.
It evaluates the champion pipeline on the isolated test set to measure
true out-of-sample generalization.
"""

from __future__ import annotations

import logging
import os
from typing import Any
import joblib
import numpy as np
from sklearn.pipeline import Pipeline

from ml.evaluation import compute_metrics
from ml.validation import DataPartitions

logger = logging.getLogger(__name__)


def evaluate_on_test(
    champion_pipeline: Pipeline,
    partitions: DataPartitions,
    campaign_id: str,
    artifacts_dir: str = "artifacts",
    task_type: str = "classification",
) -> dict[str, Any]:
    """
    Evaluate the final champion model on the isolated test set.
    Saves model artifact to artifacts directory.

    Args:
        champion_pipeline: The best trained Pipeline from the campaign
        partitions: Clean train/val/test data partitions
        campaign_id: Active campaign identifier
        artifacts_dir: Directory where the model artifact will be saved
        task_type: 'classification' or 'regression'

    Returns:
        Dict with test metrics, generalization gap, and artifact path
    """
    os.makedirs(artifacts_dir, exist_ok=True)

    X_test = partitions.test_X
    y_test = partitions.test_y.to_numpy()

    test_pred = champion_pipeline.predict(X_test)
    test_proba = None
    if task_type == "classification" and hasattr(champion_pipeline, "predict_proba"):
        try:
            test_proba = champion_pipeline.predict_proba(X_test)[:, 1]
        except Exception:
            pass

    test_metrics = compute_metrics(
        y_test, test_pred, test_proba, positive_label=1, task_type=task_type
    )

    # Save artifact
    model_filename = f"champion_model_{campaign_id}.joblib"
    model_path = os.path.join(artifacts_dir, model_filename)
    joblib.dump(champion_pipeline, model_path)
    logger.info(f"Saved champion pipeline to {model_path}")

    return {
        "test_metrics": test_metrics,
        "model_artifact_path": model_path,
        "test_sample_count": len(y_test),
    }
