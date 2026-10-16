"""
Pydantic schemas for every data contract in the ML Lab Agent.

Every action the LLM proposes, every result tools return, and every
configuration value flows through these models. If it doesn't validate,
it doesn't execute.
"""

from __future__ import annotations

import time
import uuid
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator


# ─────────────────────────────────────────────
# Configuration schemas
# ─────────────────────────────────────────────

ALLOWED_MODEL_FAMILIES = [
    # Classification
    "DummyClassifier",
    "LogisticRegression",
    "RandomForestClassifier",
    "ExtraTreesClassifier",
    "HistGradientBoostingClassifier",
    "GradientBoostingClassifier",
    "AdaBoostClassifier",
    "KNeighborsClassifier",
    "SVC",
    "GaussianNB",
    "DecisionTreeClassifier",
    # Regression
    "DummyRegressor",
    "Ridge",
    "Lasso",
    "ElasticNet",
    "LinearRegression",
    "RandomForestRegressor",
    "ExtraTreesRegressor",
    "HistGradientBoostingRegressor",
    "GradientBoostingRegressor",
    "KNeighborsRegressor",
    "SVR",
    "DecisionTreeRegressor",
    # Clustering
    "KMeans",
    "DBSCAN",
    "AgglomerativeClustering",
]

ALLOWED_METRICS = [
    "f1",
    "pr_auc",
    "roc_auc",
    "accuracy",
    "precision",
    "recall",
    "rmse",
    "r2",
    "mae",
    "mse",
    "silhouette",
    "davies_bouldin",
]

ALLOWED_ACTIONS = [
    "profile_dataset",
    "create_baseline",
    "run_experiment",
    "analyze_errors",
    "compare_experiments",
    "get_experiment_history",
    "get_remaining_budget",
    "finish_campaign",
]


class LLMProviderConfig(BaseModel):
    """Configuration for a single LLM provider."""
    name: str
    base_url: str
    api_key_env: str | None = None
    api_key_envs: list[str] = Field(default_factory=list)
    model: str
    max_tokens: int = 2048
    temperature: float = 0.3
    timeout_seconds: int = 30
    max_retries: int = 2


class CampaignConfig(BaseModel):
    """User-specified campaign parameters."""
    campaign_id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    dataset_path: str
    target_column: str = "target"
    task_type: Literal["classification", "regression", "clustering", "auto"] = "auto"
    positive_class: str | None = None
    objective_metric: str = "f1"
    max_experiments: int = 12
    max_time_minutes: int = 15
    quality_target: float | None = None
    random_seed: int = 42
    user_prompt: str | None = None
    user_guidance: str | None = None

    @field_validator("objective_metric")
    @classmethod
    def validate_metric(cls, v: str) -> str:
        clean = v.lower().strip()
        if clean not in ALLOWED_METRICS:
            raise ValueError(f"Invalid metric '{clean}'. Must be one of: {ALLOWED_METRICS}")
        return clean


# ─────────────────────────────────────────────
# Agent action schemas
# ─────────────────────────────────────────────

class AgentAction(BaseModel):
    """
    What the LLM wants to do next.
    Validated before any tool dispatch.
    """
    action: Literal[
        "run_experiment",
        "analyze_errors",
        "compare_experiments",
        "finish_campaign",
    ]
    model_family: str | None = None
    hypothesis: str = Field(default="", max_length=500)
    parameters: dict[str, Any] = Field(default_factory=dict)
    reason: str = Field(default="", max_length=500)

    @field_validator("model_family")
    @classmethod
    def validate_model_family(cls, v: str | None) -> str | None:
        if v is not None and v not in ALLOWED_MODEL_FAMILIES:
            raise ValueError(
                f"Model family '{v}' not allowed. "
                f"Choose from: {ALLOWED_MODEL_FAMILIES}"
            )
        return v


# ─────────────────────────────────────────────
# Tool result schemas
# ─────────────────────────────────────────────

class DatasetProfile(BaseModel):
    """Structured profiling output from profile_dataset tool."""
    n_rows: int
    n_columns: int
    target_column: str
    target_distribution: dict[str, int]
    class_imbalance_ratio: float
    missing_values: dict[str, int]
    duplicate_rows: int
    feature_types: dict[str, str]  # column_name -> "numeric" | "categorical"
    high_cardinality_columns: list[str]
    potential_leakage_columns: list[str]
    numerical_summary: dict[str, dict[str, float]] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)


class ExperimentResult(BaseModel):
    """Structured result from a completed experiment."""
    run_id: str = Field(default_factory=lambda: f"trial_{uuid.uuid4().hex[:6]}")
    experiment_number: int = 0
    status: Literal["completed", "failed", "timeout"] = "completed"
    model_family: str = ""
    task_type: str = "classification"
    primary_metric_name: str = "f1"
    primary_metric_value: float | None = None
    metrics: dict[str, Any] = Field(default_factory=dict)
    training_metrics: dict[str, Any] = Field(default_factory=dict)
    # Validation metrics
    validation_f1: float | None = None
    validation_precision: float | None = None
    validation_recall: float | None = None
    validation_accuracy: float | None = None
    validation_pr_auc: float | None = None
    # Training metrics (for overfitting detection)
    training_f1: float | None = None
    training_accuracy: float | None = None
    # Confusion matrix [TN, FP, FN, TP] or residual stats
    confusion_matrix: list[list[int]] | None = None
    # Metadata
    duration_seconds: float = 0.0
    parameters: dict[str, Any] = Field(default_factory=dict)
    error_message: str | None = None
    hypothesis: str = ""
    reason: str = ""
    mlflow_run_id: str | None = None


class ErrorAnalysis(BaseModel):
    """Result from error analysis tool."""
    total_predictions: int
    true_positives: int
    true_negatives: int
    false_positives: int
    false_negatives: int
    classification_report: str
    confusion_matrix: list[list[int]]
    fp_sample_indices: list[int] = Field(default_factory=list)
    fn_sample_indices: list[int] = Field(default_factory=list)
    observations: list[str] = Field(default_factory=list)


class ExperimentComparison(BaseModel):
    """Result from comparing experiments."""
    experiments: list[dict[str, Any]]
    best_run_id: str
    best_metric_value: float
    metric_name: str
    summary: str


class BudgetStatus(BaseModel):
    """Current budget status."""
    experiments_used: int
    experiments_remaining: int
    max_experiments: int
    elapsed_minutes: float
    remaining_minutes: float
    max_minutes: float


# ─────────────────────────────────────────────
# Campaign state
# ─────────────────────────────────────────────

class ActionRecord(BaseModel):
    """Record of one action taken by the agent."""
    action_number: int
    timestamp: float = Field(default_factory=time.time)
    proposed_action: dict[str, Any] = Field(default_factory=dict)
    validated: bool = False
    tool_name: str = ""
    tool_input: dict[str, Any] = Field(default_factory=dict)
    tool_result: dict[str, Any] = Field(default_factory=dict)
    provider_used: str = ""
    llm_latency_ms: float = 0.0
    execution_duration_seconds: float = 0.0
    error: str | None = None


class CampaignState(BaseModel):
    """Persistent state of an active campaign."""
    campaign_id: str
    config: CampaignConfig
    status: Literal[
        "initializing", "profiling", "baseline",
        "running", "stopping", "completed", "failed"
    ] = "initializing"
    dataset_profile: DatasetProfile | None = None
    experiments: list[ExperimentResult] = Field(default_factory=list)
    current_best: ExperimentResult | None = None
    actions: list[ActionRecord] = Field(default_factory=list)
    start_time: float = Field(default_factory=time.time)
    elapsed_seconds: float = 0.0
    stop_reason: str | None = None
    warnings: list[str] = Field(default_factory=list)
    # Data split info for reproducibility
    train_indices: list[int] | None = None
    val_indices: list[int] | None = None
    test_indices: list[int] | None = None

    def update_elapsed(self) -> None:
        """Refresh elapsed time."""
        self.elapsed_seconds = time.time() - self.start_time

    def get_score(self, exp: ExperimentResult | None) -> float | None:
        """Extract the campaign objective metric score for an experiment."""
        if exp is None:
            return None
        obj = self.config.objective_metric.lower().strip()
        if exp.primary_metric_value is not None and exp.primary_metric_name.lower().strip() == obj:
            return exp.primary_metric_value
        if obj in exp.metrics:
            return exp.metrics[obj]
        if f"validation_{obj}" in exp.metrics:
            return exp.metrics[f"validation_{obj}"]
        val = getattr(exp, f"validation_{obj}", None)
        if val is not None:
            return float(val)
        if exp.primary_metric_value is not None:
            return exp.primary_metric_value
        return exp.validation_f1

    def is_better(self, new_val: float, current_val: float) -> bool:
        """Determine if new_val improves upon current_val based on metric direction."""
        lower_better = self.config.objective_metric.lower() in [
            "rmse", "mse", "mae", "log_loss", "davies_bouldin"
        ]
        if lower_better:
            return new_val < current_val
        return new_val > current_val

    def add_experiment(self, result: ExperimentResult) -> None:
        """Add experiment and update current best if improved."""
        self.experiments.append(result)
        if result.status != "completed":
            return
        new_score = self.get_score(result)
        if new_score is None:
            return
        if self.current_best is None:
            self.current_best = result
        else:
            current_score = self.get_score(self.current_best)
            if current_score is None or self.is_better(new_score, current_score):
                self.current_best = result

    def get_completed_experiments(self) -> list[ExperimentResult]:
        """Return only successfully completed experiments."""
        return [e for e in self.experiments if e.status == "completed"]

    def experiments_remaining(self) -> int:
        """How many experiment slots remain."""
        return self.config.max_experiments - len(self.experiments)

    def time_remaining_seconds(self) -> float:
        """How many seconds remain in the budget."""
        self.update_elapsed()
        return max(0, self.config.max_time_minutes * 60 - self.elapsed_seconds)


# ─────────────────────────────────────────────
# LLM response wrapper
# ─────────────────────────────────────────────

class LLMResponse(BaseModel):
    """Response from the LLM client."""
    content: str
    provider_used: str
    model_used: str
    latency_ms: float
    prompt_tokens: int = 0
    completion_tokens: int = 0


# ─────────────────────────────────────────────
# Final report
# ─────────────────────────────────────────────

class CampaignReport(BaseModel):
    """Final output of a completed campaign."""
    campaign_id: str
    objective: str
    dataset_info: dict[str, Any]
    baseline_result: ExperimentResult | None = None
    best_result: ExperimentResult | None = None
    final_test_metrics: dict[str, Any] = Field(default_factory=dict)
    all_experiments: list[ExperimentResult] = Field(default_factory=list)
    total_experiments: int = 0
    successful_experiments: int = 0
    failed_experiments: int = 0
    total_duration_seconds: float = 0.0
    stop_reason: str = ""
    warnings: list[str] = Field(default_factory=list)
    model_artifact_path: str | None = None
    reproducibility: dict[str, Any] = Field(default_factory=dict)
