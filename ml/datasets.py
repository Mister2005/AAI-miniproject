"""
Dataset loading utilities for ML Lab Agent.

Supports:
- UCI Bank Marketing dataset (primary benchmark)
- CSV file upload
- scikit-learn built-in datasets for quick tests
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import pandas as pd


def load_bank_marketing(data_dir: str = "data") -> tuple[pd.DataFrame, dict]:
    """
    Load the UCI Bank Marketing dataset.

    Tries local cache first, then downloads via ucimlrepo.
    Returns (dataframe, metadata_dict).
    """
    cache_path = Path(data_dir) / "bank_marketing.csv"

    if cache_path.exists():
        df = pd.read_csv(cache_path)
        metadata = {
            "source": "UCI Machine Learning Repository",
            "name": "Bank Marketing",
            "url": "https://archive.ics.uci.edu/dataset/222/bank",
            "loaded_from": "local_cache",
            "file_path": str(cache_path),
        }
        return df, metadata

    # Download from UCI
    try:
        from ucimlrepo import fetch_ucirepo

        dataset = fetch_ucirepo(id=222)
        features = dataset.data.features
        targets = dataset.data.targets

        df = pd.concat([features, targets], axis=1)

        # Cache locally
        os.makedirs(data_dir, exist_ok=True)
        df.to_csv(cache_path, index=False)

        metadata = {
            "source": "UCI Machine Learning Repository",
            "name": "Bank Marketing",
            "url": "https://archive.ics.uci.edu/dataset/222/bank",
            "loaded_from": "ucimlrepo",
            "file_path": str(cache_path),
            "n_rows": len(df),
            "n_columns": len(df.columns),
        }
        return df, metadata

    except Exception as e:
        raise RuntimeError(
            f"Failed to load Bank Marketing dataset: {e}\n"
            f"Try downloading manually from "
            f"https://archive.ics.uci.edu/dataset/222/bank "
            f"and placing the CSV at {cache_path}"
        ) from e


def load_csv(file_path: str) -> tuple[pd.DataFrame, dict]:
    """
    Load a CSV file from disk.
    Returns (dataframe, metadata_dict).
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {file_path}")
    if not path.suffix.lower() == ".csv":
        raise ValueError(f"Expected .csv file, got: {path.suffix}")

    df = pd.read_csv(path)

    # Compute file checksum for reproducibility
    with open(path, "rb") as f:
        checksum = hashlib.md5(f.read()).hexdigest()

    metadata = {
        "source": "user_upload",
        "name": path.stem,
        "file_path": str(path.resolve()),
        "checksum_md5": checksum,
        "n_rows": len(df),
        "n_columns": len(df.columns),
    }
    return df, metadata


def load_sklearn_dataset(name: str) -> tuple[pd.DataFrame, dict]:
    """
    Load a scikit-learn built-in dataset for quick testing and demonstrations.
    Supports:
    - Classification: 'breast_cancer', 'iris', 'wine'
    - Regression: 'diabetes', 'california_housing'
    """
    clean_name = name.lower().strip()
    if clean_name == "breast_cancer":
        from sklearn.datasets import load_breast_cancer
        data = load_breast_cancer(as_frame=True)
        df = data.frame
        metadata = {
            "source": "sklearn",
            "name": "breast_cancer",
            "target_column": "target",
            "task_type": "classification",
            "n_rows": len(df),
            "n_columns": len(df.columns),
        }
        return df, metadata
    elif clean_name == "diabetes":
        from sklearn.datasets import load_diabetes
        data = load_diabetes(as_frame=True)
        df = data.frame
        metadata = {
            "source": "sklearn",
            "name": "diabetes",
            "target_column": "target",
            "task_type": "regression",
            "n_rows": len(df),
            "n_columns": len(df.columns),
        }
        return df, metadata
    elif clean_name in ["california_housing", "housing"]:
        from sklearn.datasets import fetch_california_housing
        data = fetch_california_housing(as_frame=True)
        df = data.frame
        metadata = {
            "source": "sklearn",
            "name": "california_housing",
            "target_column": "MedHouseVal",
            "task_type": "regression",
            "n_rows": len(df),
            "n_columns": len(df.columns),
        }
        return df, metadata
    elif clean_name == "iris":
        from sklearn.datasets import load_iris
        data = load_iris(as_frame=True)
        df = data.frame
        metadata = {
            "source": "sklearn",
            "name": "iris",
            "target_column": "target",
            "task_type": "classification",
            "n_rows": len(df),
            "n_columns": len(df.columns),
        }
        return df, metadata
    elif clean_name == "wine":
        from sklearn.datasets import load_wine
        data = load_wine(as_frame=True)
        df = data.frame
        metadata = {
            "source": "sklearn",
            "name": "wine",
            "target_column": "target",
            "task_type": "classification",
            "n_rows": len(df),
            "n_columns": len(df.columns),
        }
        return df, metadata
    else:
        raise ValueError(
            f"Unknown sklearn dataset: '{name}'. "
            f"Supported: 'breast_cancer', 'diabetes', 'california_housing', 'iris', 'wine'"
        )


def get_dataset(
    source: str,
    file_path: str | None = None,
    data_dir: str = "data",
) -> tuple[pd.DataFrame, dict]:
    """
    Unified dataset loader.

    Args:
        source: "bank_marketing", "csv", or "sklearn:<name>"
        file_path: Required when source is "csv"
        data_dir: Directory for cached datasets
    """
    if source == "bank_marketing":
        return load_bank_marketing(data_dir)
    elif source == "csv":
        if file_path is None:
            raise ValueError("file_path required when source is 'csv'")
        return load_csv(file_path)
    elif source.startswith("sklearn:"):
        name = source.split(":", 1)[1]
        return load_sklearn_dataset(name)
    else:
        raise ValueError(f"Unknown data source: {source}")


def load_dataset(path_or_source: str, data_dir: str = "data") -> pd.DataFrame:
    """
    Convenience function returning the DataFrame directly.
    Accepts:
    - Path to a CSV file (e.g. 'data/bank_marketing.csv')
    - 'bank_marketing'
    - 'breast_cancer', 'diabetes', 'california_housing', 'iris', 'wine'
    - 'sklearn:<name>'
    """
    clean = path_or_source.lower().strip()
    if clean == "bank_marketing":
        df, _ = load_bank_marketing(data_dir=data_dir)
        return df
    if clean in ["breast_cancer", "diabetes", "california_housing", "housing", "iris", "wine"]:
        df, _ = load_sklearn_dataset(clean)
        return df
    if clean.startswith("sklearn:"):
        df, _ = load_sklearn_dataset(clean.split(":", 1)[1])
        return df
    if os.path.exists(path_or_source):
        df, _ = load_csv(path_or_source)
        return df
    raise ValueError(f"Unable to load dataset from: {path_or_source}")
