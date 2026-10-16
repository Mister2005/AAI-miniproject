"""
Comprehensive Model Registry for ML Lab Agent.

Supports Classification, Regression, and Clustering models.
Includes automatic parameter sanitization (e.g. converting deprecated 'max_features'='auto'
to 'sqrt', filtering invalid kwargs) so LLM exploration never crashes execution.
"""

from __future__ import annotations

import inspect
import logging
from typing import Any

# Classification
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import (
    AdaBoostClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

# Regression
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import (
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

# Clustering
from sklearn.cluster import AgglomerativeClustering, DBSCAN, KMeans

logger = logging.getLogger(__name__)

MODEL_REGISTRY: dict[str, dict[str, Any]] = {
    # ── Classification ────────────────────────────────────────────────────────
    "DummyClassifier": {
        "class": DummyClassifier,
        "type": "classification",
        "default_params": {"strategy": "most_frequent"},
        "needs_scaling": False,
        "supports_class_weight": False,
        "description": "Baseline predicting most frequent class.",
    },
    "LogisticRegression": {
        "class": LogisticRegression,
        "type": "classification",
        "default_params": {"max_iter": 1000, "solver": "lbfgs", "random_state": 42},
        "needs_scaling": True,
        "supports_class_weight": True,
        "description": "Linear model with L2 regularization.",
    },
    "RandomForestClassifier": {
        "class": RandomForestClassifier,
        "type": "classification",
        "default_params": {"n_estimators": 100, "random_state": 42, "n_jobs": -1},
        "needs_scaling": False,
        "supports_class_weight": True,
        "description": "Bagging ensemble of decision trees.",
    },
    "ExtraTreesClassifier": {
        "class": ExtraTreesClassifier,
        "type": "classification",
        "default_params": {"n_estimators": 100, "random_state": 42, "n_jobs": -1},
        "needs_scaling": False,
        "supports_class_weight": True,
        "description": "Extremely randomized tree ensemble.",
    },
    "HistGradientBoostingClassifier": {
        "class": HistGradientBoostingClassifier,
        "type": "classification",
        "default_params": {"max_iter": 200, "random_state": 42},
        "needs_scaling": False,
        "supports_class_weight": False,
        "description": "Fast histogram-based gradient boosting.",
    },
    "GradientBoostingClassifier": {
        "class": GradientBoostingClassifier,
        "type": "classification",
        "default_params": {"n_estimators": 100, "random_state": 42},
        "needs_scaling": False,
        "supports_class_weight": False,
        "description": "Standard gradient boosted decision trees.",
    },
    "AdaBoostClassifier": {
        "class": AdaBoostClassifier,
        "type": "classification",
        "default_params": {"n_estimators": 50, "random_state": 42},
        "needs_scaling": False,
        "supports_class_weight": False,
        "description": "Adaptive boosting ensemble.",
    },
    "KNeighborsClassifier": {
        "class": KNeighborsClassifier,
        "type": "classification",
        "default_params": {"n_neighbors": 5},
        "needs_scaling": True,
        "supports_class_weight": False,
        "description": "K-Nearest Neighbors classifier.",
    },
    "SVC": {
        "class": SVC,
        "type": "classification",
        "default_params": {"probability": True, "random_state": 42},
        "needs_scaling": True,
        "supports_class_weight": True,
        "description": "Support Vector Machine with RBF kernel.",
    },
    "GaussianNB": {
        "class": GaussianNB,
        "type": "classification",
        "default_params": {},
        "needs_scaling": False,
        "supports_class_weight": False,
        "description": "Gaussian Naive Bayes classifier.",
    },
    "DecisionTreeClassifier": {
        "class": DecisionTreeClassifier,
        "type": "classification",
        "default_params": {"max_depth": 10, "random_state": 42},
        "needs_scaling": False,
        "supports_class_weight": True,
        "description": "Single decision tree classifier.",
    },

    # ── Regression ───────────────────────────────────────────────────────────
    "DummyRegressor": {
        "class": DummyRegressor,
        "type": "regression",
        "default_params": {"strategy": "mean"},
        "needs_scaling": False,
        "supports_class_weight": False,
        "description": "Baseline predicting mean target.",
    },
    "LinearRegression": {
        "class": LinearRegression,
        "type": "regression",
        "default_params": {},
        "needs_scaling": True,
        "supports_class_weight": False,
        "description": "Ordinary least squares linear regression.",
    },
    "Ridge": {
        "class": Ridge,
        "type": "regression",
        "default_params": {"alpha": 1.0, "random_state": 42},
        "needs_scaling": True,
        "supports_class_weight": False,
        "description": "Linear regression with L2 regularization.",
    },
    "Lasso": {
        "class": Lasso,
        "type": "regression",
        "default_params": {"alpha": 1.0, "random_state": 42},
        "needs_scaling": True,
        "supports_class_weight": False,
        "description": "Linear regression with L1 regularization.",
    },
    "ElasticNet": {
        "class": ElasticNet,
        "type": "regression",
        "default_params": {"alpha": 1.0, "l1_ratio": 0.5, "random_state": 42},
        "needs_scaling": True,
        "supports_class_weight": False,
        "description": "Linear regression with combined L1 and L2 penalty.",
    },
    "RandomForestRegressor": {
        "class": RandomForestRegressor,
        "type": "regression",
        "default_params": {"n_estimators": 100, "random_state": 42, "n_jobs": -1},
        "needs_scaling": False,
        "supports_class_weight": False,
        "description": "Random forest ensemble for regression.",
    },
    "ExtraTreesRegressor": {
        "class": ExtraTreesRegressor,
        "type": "regression",
        "default_params": {"n_estimators": 100, "random_state": 42, "n_jobs": -1},
        "needs_scaling": False,
        "supports_class_weight": False,
        "description": "Extremely randomized trees regressor.",
    },
    "HistGradientBoostingRegressor": {
        "class": HistGradientBoostingRegressor,
        "type": "regression",
        "default_params": {"max_iter": 200, "random_state": 42},
        "needs_scaling": False,
        "supports_class_weight": False,
        "description": "Histogram-based gradient boosted regressor.",
    },
    "GradientBoostingRegressor": {
        "class": GradientBoostingRegressor,
        "type": "regression",
        "default_params": {"n_estimators": 100, "random_state": 42},
        "needs_scaling": False,
        "supports_class_weight": False,
        "description": "Gradient boosted decision trees for regression.",
    },
    "KNeighborsRegressor": {
        "class": KNeighborsRegressor,
        "type": "regression",
        "default_params": {"n_neighbors": 5},
        "needs_scaling": True,
        "supports_class_weight": False,
        "description": "K-Nearest Neighbors regressor.",
    },
    "SVR": {
        "class": SVR,
        "type": "regression",
        "default_params": {"kernel": "rbf"},
        "needs_scaling": True,
        "supports_class_weight": False,
        "description": "Epsilon-Support Vector Regression.",
    },
    "DecisionTreeRegressor": {
        "class": DecisionTreeRegressor,
        "type": "regression",
        "default_params": {"max_depth": 10, "random_state": 42},
        "needs_scaling": False,
        "supports_class_weight": False,
        "description": "Single decision tree regressor.",
    },

    # ── Clustering ───────────────────────────────────────────────────────────
    "KMeans": {
        "class": KMeans,
        "type": "clustering",
        "default_params": {"n_clusters": 5, "random_state": 42, "n_init": "auto"},
        "needs_scaling": True,
        "supports_class_weight": False,
        "description": "K-Means clustering.",
    },
    "DBSCAN": {
        "class": DBSCAN,
        "type": "clustering",
        "default_params": {"eps": 0.5, "min_samples": 5},
        "needs_scaling": True,
        "supports_class_weight": False,
        "description": "Density-based spatial clustering.",
    },
    "AgglomerativeClustering": {
        "class": AgglomerativeClustering,
        "type": "clustering",
        "default_params": {"n_clusters": 5},
        "needs_scaling": True,
        "supports_class_weight": False,
        "description": "Hierarchical agglomerative clustering.",
    },
}


def sanitize_params(family: str, params: dict[str, Any]) -> dict[str, Any]:
    """
    Clean and sanitize parameters for a given model family.
    - Fixes scikit-learn 1.4+ deprecation: 'auto' max_features -> 'sqrt'
    - Strips unsupported kwargs so model initialization never fails
    """
    clean = dict(params)

    # Fix deprecated max_features='auto'
    if clean.get("max_features") == "auto":
        clean["max_features"] = "sqrt"

    # Fix deprecated loss='deviance' for GradientBoostingClassifier
    if clean.get("loss") == "deviance":
        clean["loss"] = "log_loss"

    # Fix deprecated criterion 'mse' / 'mae' in tree models
    if clean.get("criterion") == "mse":
        clean["criterion"] = "squared_error"
    elif clean.get("criterion") == "mae":
        clean["criterion"] = "absolute_error"

    # Filter out parameters not accepted by the model's __init__
    model_cls = MODEL_REGISTRY[family]["class"]
    try:
        sig = inspect.signature(model_cls.__init__)
        valid_params = set(sig.parameters.keys()) - {"self"}
        # If __init__ accepts **kwargs, keep all
        has_var_keyword = any(
            p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
        )
        if not has_var_keyword:
            clean = {k: v for k, v in clean.items() if k in valid_params}
    except Exception:
        pass

    # Strip class_weight if not supported
    if "class_weight" in clean and not MODEL_REGISTRY[family]["supports_class_weight"]:
        clean.pop("class_weight", None)

    return clean


def get_model_class(family: str):
    """Get the sklearn class for a model family."""
    if family not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model family: '{family}'. Allowed: {list(MODEL_REGISTRY.keys())}"
        )
    return MODEL_REGISTRY[family]["class"]


def get_default_params(family: str) -> dict[str, Any]:
    """Get default parameters for a model family."""
    if family not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model family: '{family}'")
    return MODEL_REGISTRY[family]["default_params"].copy()


def create_model(family: str, params: dict[str, Any] | None = None):
    """Instantiate a sanitized model from registry."""
    if family not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model family '{family}'")

    model_class = get_model_class(family)
    default = get_default_params(family)

    if params:
        merged = {**default, **params}
    else:
        merged = default

    sanitized = sanitize_params(family, merged)
    return model_class(**sanitized)


def instantiate_model(family: str, **params: Any):
    """Convenience alias for create_model."""
    return create_model(family, params if params else None)


def validate_model_family(family: str) -> None:
    """Validate that family is in MODEL_REGISTRY."""
    if family not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model family: '{family}'. Allowed: {list(MODEL_REGISTRY.keys())}"
        )


def needs_scaling(family: str) -> bool:
    """Check if model needs feature scaling."""
    return MODEL_REGISTRY.get(family, {}).get("needs_scaling", False)


def list_families(task_type: str | None = None) -> list[str]:
    """List registered model families, optionally filtered by task type."""
    if task_type:
        return [f for f, v in MODEL_REGISTRY.items() if v["type"] == task_type]
    return list(MODEL_REGISTRY.keys())
