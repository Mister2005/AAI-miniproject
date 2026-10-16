"""
Optuna search spaces for diverse model families across Classification,
Regression, and Clustering in ML Lab Agent.
"""

from __future__ import annotations

from typing import Any
import optuna

SEARCH_SPACES: dict[str, dict[str, tuple]] = {
    # ── Classification ────────────────────────────────────────────────────────
    "LogisticRegression": {
        "C": ("log_float", 1e-4, 10.0),
        "penalty": ("categorical", ["l1", "l2"]),
        "solver": ("categorical", ["liblinear", "saga"]),
    },
    "RandomForestClassifier": {
        "n_estimators": ("int", 50, 300),
        "max_depth": ("int_or_none", 3, 20),
        "min_samples_split": ("int", 2, 20),
        "min_samples_leaf": ("int", 1, 10),
    },
    "ExtraTreesClassifier": {
        "n_estimators": ("int", 50, 300),
        "max_depth": ("int_or_none", 3, 20),
        "min_samples_split": ("int", 2, 20),
        "min_samples_leaf": ("int", 1, 10),
    },
    "HistGradientBoostingClassifier": {
        "learning_rate": ("log_float", 0.01, 0.3),
        "max_iter": ("int", 100, 500),
        "max_depth": ("int_or_none", 3, 12),
        "min_samples_leaf": ("int", 5, 50),
        "l2_regularization": ("log_float", 1e-6, 10.0),
    },
    "GradientBoostingClassifier": {
        "learning_rate": ("log_float", 0.01, 0.2),
        "n_estimators": ("int", 50, 250),
        "max_depth": ("int", 3, 8),
        "subsample": ("float", 0.6, 1.0),
    },
    "AdaBoostClassifier": {
        "n_estimators": ("int", 30, 200),
        "learning_rate": ("log_float", 0.01, 1.5),
    },
    "KNeighborsClassifier": {
        "n_neighbors": ("int", 3, 25),
        "weights": ("categorical", ["uniform", "distance"]),
    },
    "SVC": {
        "C": ("log_float", 0.1, 50.0),
        "gamma": ("categorical", ["scale", "auto"]),
    },
    "DecisionTreeClassifier": {
        "max_depth": ("int_or_none", 3, 20),
        "min_samples_split": ("int", 2, 20),
    },

    # ── Regression ───────────────────────────────────────────────────────────
    "Ridge": {
        "alpha": ("log_float", 1e-4, 100.0),
    },
    "Lasso": {
        "alpha": ("log_float", 1e-4, 50.0),
    },
    "ElasticNet": {
        "alpha": ("log_float", 1e-4, 50.0),
        "l1_ratio": ("float", 0.1, 0.9),
    },
    "RandomForestRegressor": {
        "n_estimators": ("int", 50, 300),
        "max_depth": ("int_or_none", 3, 20),
        "min_samples_split": ("int", 2, 20),
        "min_samples_leaf": ("int", 1, 10),
    },
    "ExtraTreesRegressor": {
        "n_estimators": ("int", 50, 300),
        "max_depth": ("int_or_none", 3, 20),
        "min_samples_split": ("int", 2, 20),
        "min_samples_leaf": ("int", 1, 10),
    },
    "HistGradientBoostingRegressor": {
        "learning_rate": ("log_float", 0.01, 0.3),
        "max_iter": ("int", 100, 500),
        "max_depth": ("int_or_none", 3, 12),
        "min_samples_leaf": ("int", 5, 50),
        "l2_regularization": ("log_float", 1e-6, 10.0),
    },
    "GradientBoostingRegressor": {
        "learning_rate": ("log_float", 0.01, 0.2),
        "n_estimators": ("int", 50, 250),
        "max_depth": ("int", 3, 8),
    },
    "KNeighborsRegressor": {
        "n_neighbors": ("int", 3, 25),
        "weights": ("categorical", ["uniform", "distance"]),
    },
    "SVR": {
        "C": ("log_float", 0.1, 50.0),
    },
    "DecisionTreeRegressor": {
        "max_depth": ("int_or_none", 3, 20),
    },

    # ── Clustering ───────────────────────────────────────────────────────────
    "KMeans": {
        "n_clusters": ("int", 2, 12),
    },
    "DBSCAN": {
        "eps": ("float", 0.1, 2.0),
        "min_samples": ("int", 3, 15),
    },
    "AgglomerativeClustering": {
        "n_clusters": ("int", 2, 12),
    },
}


def suggest_params(
    trial: optuna.Trial,
    model_family: str,
) -> dict[str, Any]:
    """Suggest hyperparameters from approved search space."""
    if model_family not in SEARCH_SPACES:
        return {}

    space = SEARCH_SPACES[model_family]
    params = {}

    for param_name, spec in space.items():
        param_type = spec[0]

        if param_type == "int":
            params[param_name] = trial.suggest_int(param_name, spec[1], spec[2])
        elif param_type == "float":
            params[param_name] = trial.suggest_float(param_name, spec[1], spec[2])
        elif param_type == "log_float":
            params[param_name] = trial.suggest_float(param_name, spec[1], spec[2], log=True)
        elif param_type == "categorical":
            params[param_name] = trial.suggest_categorical(param_name, spec[1])
        elif param_type == "int_or_none":
            use_none = trial.suggest_categorical(f"{param_name}_is_none", [True, False])
            if use_none:
                params[param_name] = None
            else:
                params[param_name] = trial.suggest_int(param_name, spec[1], spec[2])

    return params
