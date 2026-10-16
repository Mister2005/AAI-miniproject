"""
Data splitting and integrity validation for ML Lab Agent.

Strict separation of train / validation / test partitions.
The test set is touched exactly once — during final evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


@dataclass
class DataSplit:
    """One partition of the data with its metadata."""
    X: pd.DataFrame
    y: pd.Series
    indices: np.ndarray
    name: str  # "train", "validation", "test"

    @property
    def n_samples(self) -> int:
        return len(self.y)

    @property
    def class_distribution(self) -> dict:
        return self.y.value_counts().to_dict()


@dataclass
class DataPartitions:
    """Convenient bundle of all partitions for training and evaluation tools."""
    train_split: DataSplit
    val_split: DataSplit
    test_split: DataSplit

    @property
    def train_X(self) -> pd.DataFrame:
        return self.train_split.X

    @property
    def train_y(self) -> pd.Series:
        return self.train_split.y

    @property
    def val_X(self) -> pd.DataFrame:
        return self.val_split.X

    @property
    def val_y(self) -> pd.Series:
        return self.val_split.y

    @property
    def test_X(self) -> pd.DataFrame:
        return self.test_split.X

    @property
    def test_y(self) -> pd.Series:
        return self.test_split.y


def create_partitions(
    df: pd.DataFrame,
    target_column: str,
    test_size: float = 0.15,
    validation_size: float = 0.15,
    random_seed: int = 42,
    stratify: bool = True,
) -> DataPartitions:
    """Create train, validation, and test partitions bundled in DataPartitions."""
    train, val, test = create_splits(
        df=df,
        target_column=target_column,
        test_size=test_size,
        val_size=validation_size,
        random_seed=random_seed,
        stratify=stratify,
    )
    return DataPartitions(train_split=train, val_split=val, test_split=test)


def create_splits(
    df: pd.DataFrame,
    target_column: str,
    test_size: float = 0.15,
    val_size: float = 0.15,
    random_seed: int = 42,
    stratify: bool = True,
) -> tuple[DataSplit, DataSplit, DataSplit]:
    """
    Create train/validation/test splits with strict separation.

    Strategy:
    1. Split off the test set first (never touched during experiments).
    2. Split the remainder into train and validation.
    3. Validate no index overlap.

    Args:
        df: Full dataset
        target_column: Name of the target column
        test_size: Fraction for final test set
        val_size: Fraction for validation set (of original data)
        random_seed: For reproducibility
        stratify: Stratify by target class

    Returns:
        (train_split, val_split, test_split)
    """
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not in dataframe")

    X = df.drop(columns=[target_column])
    y_raw = df[target_column]

    is_numeric = pd.api.types.is_numeric_dtype(y_raw)
    n_unique = y_raw.dropna().nunique()
    is_regression = is_numeric and n_unique > 20

    if is_regression:
        y = y_raw.astype(float)
        stratify_col = None
    else:
        # Ensure target is clean binary {0, 1} with minority class as 1
        if not (is_numeric and set(y_raw.dropna().unique()).issubset({0, 1})):
            from ml.preprocessing import encode_target
            y, le, pos_label = encode_target(y_raw)
            if pos_label == 0:
                y = 1 - y
        else:
            y = y_raw.astype(int)
        stratify_col = y if stratify else None

    # Step 1: Split off test set
    X_dev, X_test, y_dev, y_test, idx_dev, idx_test = train_test_split(
        X, y, np.arange(len(df)),
        test_size=test_size,
        random_state=random_seed,
        stratify=stratify_col,
    )

    # Step 2: Split dev into train and validation
    # Adjust val_size relative to dev size
    val_fraction_of_dev = val_size / (1 - test_size)
    stratify_dev = y_dev if (stratify and not is_regression) else None

    X_train, X_val, y_train, y_val, idx_train, idx_val = train_test_split(
        X_dev, y_dev, idx_dev,
        test_size=val_fraction_of_dev,
        random_state=random_seed,
        stratify=stratify_dev,
    )

    # Reset indices for clean DataFrames
    X_train = X_train.reset_index(drop=True)
    X_val = X_val.reset_index(drop=True)
    X_test = X_test.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)
    y_val = y_val.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)

    train_split = DataSplit(X=X_train, y=y_train, indices=idx_train, name="train")
    val_split = DataSplit(X=X_val, y=y_val, indices=idx_val, name="validation")
    test_split = DataSplit(X=X_test, y=y_test, indices=idx_test, name="test")

    # Validate no overlap
    check_no_leakage(train_split, val_split, test_split)

    return train_split, val_split, test_split


def check_no_leakage(
    train: DataSplit,
    val: DataSplit,
    test: DataSplit,
) -> None:
    """
    Verify no index overlap between any two partitions.
    Raises ValueError if overlap is detected.
    """
    train_set = set(train.indices.tolist())
    val_set = set(val.indices.tolist())
    test_set = set(test.indices.tolist())

    train_val = train_set & val_set
    train_test = train_set & test_set
    val_test = val_set & test_set

    errors = []
    if train_val:
        errors.append(f"Train-validation overlap: {len(train_val)} indices")
    if train_test:
        errors.append(f"Train-test overlap: {len(train_test)} indices")
    if val_test:
        errors.append(f"Validation-test overlap: {len(val_test)} indices")

    if errors:
        raise ValueError(
            "DATA LEAKAGE DETECTED:\n" + "\n".join(errors)
        )

    # Also verify total count
    total = len(train_set) + len(val_set) + len(test_set)
    all_unique = len(train_set | val_set | test_set)
    if total != all_unique:
        raise ValueError(
            f"Index count mismatch: {total} total vs {all_unique} unique. "
            "Possible duplicate indices."
        )


def get_split_info(
    train: DataSplit,
    val: DataSplit,
    test: DataSplit,
) -> dict:
    """Get summary of splits for logging and reproducibility."""
    return {
        "train_samples": train.n_samples,
        "validation_samples": val.n_samples,
        "test_samples": test.n_samples,
        "train_class_distribution": train.class_distribution,
        "validation_class_distribution": val.class_distribution,
        "test_class_distribution": test.class_distribution,
        "total_samples": train.n_samples + val.n_samples + test.n_samples,
    }
