"""
Dataset profiling tool for ML Lab Agent.

Returns structured information about the dataset:
schema, quality statistics, target distribution, and warnings.
The agent uses this to understand what it's working with.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from agent.schemas import DatasetProfile
from ml.preprocessing import identify_feature_types


def profile_dataset(
    df: pd.DataFrame,
    target_column: str,
) -> DatasetProfile:
    """
    Profile a dataset and return structured quality information.

    Detects:
    - Missing values per column
    - Duplicate rows
    - High-cardinality categorical columns (>50 unique)
    - Potential leakage columns (high correlation with target, ID-like)
    - Class imbalance
    - Numerical summaries

    Args:
        df: The full dataframe
        target_column: Name of the target column

    Returns:
        DatasetProfile with all profiling results
    """
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in dataset")

    # Basic shape
    n_rows, n_columns = df.shape

    # Target distribution
    target_dist = df[target_column].value_counts().to_dict()
    # Convert keys to strings for JSON serialization
    target_dist = {str(k): int(v) for k, v in target_dist.items()}

    # Class imbalance ratio (majority / minority)
    counts = list(target_dist.values())
    if len(counts) >= 2:
        class_imbalance_ratio = max(counts) / min(counts)
    else:
        class_imbalance_ratio = 1.0

    # Missing values
    missing = df.isnull().sum()
    missing_dict = {
        col: int(count) for col, count in missing.items() if count > 0
    }

    # Duplicate rows
    duplicate_rows = int(df.duplicated().sum())

    # Feature types (excluding target)
    features_df = df.drop(columns=[target_column])
    feature_types = identify_feature_types(features_df)

    # High cardinality columns (>50 unique values in categorical)
    high_cardinality = []
    for col in features_df.columns:
        if feature_types.get(col) == "categorical" and features_df[col].nunique() > 50:
            high_cardinality.append(col)

    # Potential leakage columns
    leakage_columns = _detect_potential_leakage(df, target_column, feature_types)

    # Numerical summary
    numeric_cols = [c for c, t in feature_types.items() if t == "numeric"]
    numerical_summary = {}
    if numeric_cols:
        desc = features_df[numeric_cols].describe()
        for col in numeric_cols:
            numerical_summary[col] = {
                "mean": float(desc[col]["mean"]),
                "std": float(desc[col]["std"]),
                "min": float(desc[col]["min"]),
                "max": float(desc[col]["max"]),
                "median": float(features_df[col].median()),
            }

    # Warnings
    warnings = _generate_warnings(
        n_rows=n_rows,
        target_dist=target_dist,
        class_imbalance_ratio=class_imbalance_ratio,
        missing_dict=missing_dict,
        duplicate_rows=duplicate_rows,
        high_cardinality=high_cardinality,
        leakage_columns=leakage_columns,
    )

    return DatasetProfile(
        n_rows=n_rows,
        n_columns=n_columns,
        target_column=target_column,
        target_distribution=target_dist,
        class_imbalance_ratio=round(class_imbalance_ratio, 2),
        missing_values=missing_dict,
        duplicate_rows=duplicate_rows,
        feature_types=feature_types,
        high_cardinality_columns=high_cardinality,
        potential_leakage_columns=leakage_columns,
        numerical_summary=numerical_summary,
        warnings=warnings,
    )


def _detect_potential_leakage(
    df: pd.DataFrame,
    target_column: str,
    feature_types: dict[str, str],
) -> list[str]:
    """
    Detect columns that may leak information about the target.

    Checks:
    1. ID-like columns (unique values == n_rows)
    2. Columns with suspiciously high correlation to target
    3. Known leakage-prone column names (e.g., 'duration' in Bank Marketing)
    """
    leakage_suspects = []
    features_df = df.drop(columns=[target_column])

    # 1. ID-like columns
    for col in features_df.columns:
        if features_df[col].nunique() == len(features_df):
            leakage_suspects.append(col)

    # 2. Known problematic names
    suspect_names = {"duration", "outcome", "result", "response", "target_"}
    for col in features_df.columns:
        col_lower = col.lower()
        for name in suspect_names:
            if name in col_lower and col not in leakage_suspects:
                leakage_suspects.append(col)

    # 3. High correlation with target (for numeric targets)
    try:
        # Encode target if categorical
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        y_encoded = le.fit_transform(df[target_column])

        for col in features_df.columns:
            if feature_types.get(col) == "numeric" and col not in leakage_suspects:
                try:
                    corr = abs(np.corrcoef(
                        features_df[col].fillna(0).values,
                        y_encoded
                    )[0, 1])
                    if corr > 0.9:
                        leakage_suspects.append(col)
                except (ValueError, TypeError):
                    pass
    except Exception:
        pass

    return leakage_suspects


def _generate_warnings(
    n_rows: int,
    target_dist: dict,
    class_imbalance_ratio: float,
    missing_dict: dict,
    duplicate_rows: int,
    high_cardinality: list,
    leakage_columns: list,
) -> list[str]:
    """Generate human-readable warnings for the agent and dashboard."""
    warnings = []

    if n_rows < 100:
        warnings.append(f"Small dataset ({n_rows} rows). Results may be unreliable.")

    if class_imbalance_ratio > 5:
        warnings.append(
            f"Significant class imbalance (ratio {class_imbalance_ratio:.1f}:1). "
            f"Consider class_weight='balanced' or threshold tuning."
        )
    elif class_imbalance_ratio > 2:
        warnings.append(
            f"Moderate class imbalance (ratio {class_imbalance_ratio:.1f}:1). "
            f"Monitor per-class metrics."
        )

    if missing_dict:
        total_missing = sum(missing_dict.values())
        cols_with_missing = len(missing_dict)
        warnings.append(
            f"{total_missing} missing values across {cols_with_missing} columns."
        )

    if duplicate_rows > 0:
        warnings.append(f"{duplicate_rows} duplicate rows found.")

    if high_cardinality:
        warnings.append(
            f"High-cardinality columns: {high_cardinality}. "
            f"Consider encoding strategy."
        )

    if leakage_columns:
        warnings.append(
            f"Potential leakage columns: {leakage_columns}. "
            f"Verify these are available at prediction time."
        )

    return warnings


def profile_to_summary(profile: DatasetProfile) -> str:
    """
    Convert a DatasetProfile to a concise text summary for the LLM context.
    """
    lines = [
        f"Dataset: {profile.n_rows} rows, {profile.n_columns} columns",
        f"Target: '{profile.target_column}'",
        f"Target distribution: {profile.target_distribution}",
        f"Class imbalance ratio: {profile.class_imbalance_ratio:.1f}:1",
    ]

    numeric_count = sum(1 for t in profile.feature_types.values() if t == "numeric")
    cat_count = sum(1 for t in profile.feature_types.values() if t == "categorical")
    lines.append(f"Features: {numeric_count} numeric, {cat_count} categorical")

    if profile.missing_values:
        lines.append(f"Missing values: {sum(profile.missing_values.values())} total")

    if profile.warnings:
        lines.append("Warnings:")
        for w in profile.warnings:
            lines.append(f"  - {w}")

    return "\n".join(lines)
