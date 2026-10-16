"""
Preprocessing pipeline factory for ML Lab Agent.

Builds sklearn ColumnTransformer pipelines appropriate for each model family.
Tree-based models skip scaling; linear models get it.
"""

from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    LabelEncoder,
    OneHotEncoder,
    OrdinalEncoder,
    StandardScaler,
)


def identify_feature_types(df: pd.DataFrame) -> dict[str, str]:
    """
    Classify each column as 'numeric' or 'categorical'.

    Heuristic:
    - Numeric dtypes with > 10 unique values → numeric
    - Everything else → categorical
    """
    feature_types = {}
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            if df[col].nunique() > 10:
                feature_types[col] = "numeric"
            else:
                feature_types[col] = "categorical"
        else:
            feature_types[col] = "categorical"
    return feature_types


def build_preprocessing_pipeline(
    feature_types: dict[str, str],
    model_family: str,
    use_onehot: bool = True,
) -> ColumnTransformer:
    """
    Build a sklearn ColumnTransformer based on feature types and model family.

    Args:
        feature_types: Mapping of column_name -> "numeric" | "categorical"
        model_family: Name from model registry (affects scaling choice)
        use_onehot: Use OneHotEncoder (True) or OrdinalEncoder (False) for categoricals

    Returns:
        ColumnTransformer that handles both numeric and categorical features.
    """
    numeric_cols = [c for c, t in feature_types.items() if t == "numeric"]
    categorical_cols = [c for c, t in feature_types.items() if t == "categorical"]

    # Numeric pipeline: impute missing values, optionally scale
    needs_scaling = model_family in ("LogisticRegression",)

    if needs_scaling:
        numeric_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ])
    else:
        numeric_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
        ])

    # Categorical pipeline: impute missing, encode
    if use_onehot:
        categorical_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False,
                drop="if_binary",
            )),
        ])
    else:
        categorical_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OrdinalEncoder(
                handle_unknown="use_encoded_value",
                unknown_value=-1,
            )),
        ])

    transformers = []
    if numeric_cols:
        transformers.append(("numeric", numeric_pipeline, numeric_cols))
    if categorical_cols:
        transformers.append(("categorical", categorical_pipeline, categorical_cols))

    return ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        verbose_feature_names_out=False,
    )


def encode_target(y: pd.Series, positive_class: str | None = None) -> tuple:
    """
    Encode target variable to 0/1 for binary classification.

    Returns:
        (encoded_series, label_encoder, positive_label_encoded)
    """
    le = LabelEncoder()
    y_encoded = pd.Series(le.fit_transform(y), name=y.name)

    if positive_class is not None:
        pos_encoded = le.transform([positive_class])[0]
    else:
        # Default: the minority class is positive
        value_counts = y_encoded.value_counts()
        pos_encoded = value_counts.idxmin()

    return y_encoded, le, pos_encoded


# Alias for consistent naming across tools
create_preprocessing_pipeline = build_preprocessing_pipeline
