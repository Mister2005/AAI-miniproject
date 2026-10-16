"""
ML core package for ML Lab Agent.
"""

from ml.datasets import get_dataset, load_bank_marketing, load_csv, load_dataset, load_sklearn_dataset
from ml.evaluation import compute_metrics
from ml.model_registry import MODEL_REGISTRY, instantiate_model, validate_model_family
from ml.preprocessing import create_preprocessing_pipeline, identify_feature_types
from ml.search_spaces import SEARCH_SPACES, suggest_params
from ml.validation import DataPartitions, DataSplit, check_no_leakage, create_partitions, create_splits

__all__ = [
    "load_dataset",
    "get_dataset",
    "load_bank_marketing",
    "load_csv",
    "load_sklearn_dataset",
    "compute_metrics",
    "MODEL_REGISTRY",
    "instantiate_model",
    "validate_model_family",
    "create_preprocessing_pipeline",
    "identify_feature_types",
    "SEARCH_SPACES",
    "suggest_params",
    "DataPartitions",
    "DataSplit",
    "create_partitions",
    "create_splits",
    "check_no_leakage",
]
