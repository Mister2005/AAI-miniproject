"""
Autonomous ML Lab Agent Orchestrator.

Drives the complete end-to-end campaign:
1. Data loading & partition integrity verification
2. Dataset profiling and task type auto-detection (Classification / Regression)
3. Benchmark baseline training
4. Autonomous reasoning-action loop with guardrails
5. Plateau / stall detection and budget enforcement
6. Strictly isolated final test set evaluation
7. Persistent reporting and artifact generation
"""

from __future__ import annotations

import logging
import time
from typing import Callable
import pandas as pd
from sklearn.pipeline import Pipeline

from agent.llm_client import LLMClient
from agent.prompt_templates import build_agent_context
from agent.schemas import (
    ActionRecord,
    AgentAction,
    CampaignConfig,
    CampaignReport,
    CampaignState,
    ExperimentResult,
)
from ml.datasets import load_dataset
from ml.model_registry import list_families
from ml.validation import create_partitions, DataPartitions
from storage.run_store import RunStore
from tools.compare_experiments import compare_experiments
from tools.error_analysis import analyze_errors
from tools.evaluate_on_test import evaluate_on_test
from tools.profile_dataset import profile_dataset
from tools.run_hpo_iteration import run_hpo_iteration
from tools.train_baseline import train_baseline

logger = logging.getLogger(__name__)


class AgentOrchestrator:
    """
    Main controller for autonomous ML campaigns across classification and regression.
    """

    def __init__(
        self,
        config: CampaignConfig,
        config_yaml_path: str = "config.yaml",
        run_store: RunStore | None = None,
        on_step_callback: Callable[[str, dict], None] | None = None,
    ):
        self.config = config
        self.config_yaml_path = config_yaml_path
        self.run_store = run_store or RunStore()
        self.on_step_callback = on_step_callback
        self.llm_client = LLMClient(config_path=config_yaml_path)

        self.state = CampaignState(
            campaign_id=config.campaign_id,
            config=config,
            status="initializing",
        )
        self.partitions: DataPartitions | None = None
        self.champion_pipeline: Pipeline | None = None
        self.recent_error_analysis: dict | None = None

    def _notify(self, event_type: str, data: dict) -> None:
        """Send live status update to callback (e.g. Streamlit UI)."""
        if self.on_step_callback:
            try:
                self.on_step_callback(event_type, data)
            except Exception as e:
                logger.warning(f"Error in step callback: {e}")

    def run_campaign(self) -> CampaignReport:
        """
        Execute the full autonomous campaign from data loading to final report.
        """
        logger.info(f"Starting campaign {self.state.campaign_id}...")
        self.state.status = "initializing"
        self._notify("campaign_started", {"campaign_id": self.state.campaign_id})

        # ── Step 1: Load Data & Partition ───────────────────────────────────
        self.state.status = "profiling"
        try:
            df = load_dataset(self.config.dataset_path)

            # Auto-detect target column if missing
            if not self.config.target_column or self.config.target_column not in df.columns:
                candidates = ["target", "y", "label", "price", "churn", "SalePrice", "class"]
                found = [c for c in df.columns if c.lower() in candidates]
                self.config.target_column = found[0] if found else df.columns[-1]

            # Auto-detect task type
            if self.config.task_type == "auto":
                target_s = df[self.config.target_column].dropna()
                if pd.api.types.is_numeric_dtype(target_s) and target_s.nunique() > 20:
                    self.config.task_type = "regression"
                    if self.config.objective_metric in ["f1", "pr_auc", "accuracy"]:
                        self.config.objective_metric = "rmse"
                else:
                    self.config.task_type = "classification"

            self.partitions = create_partitions(
                df=df,
                target_column=self.config.target_column,
                test_size=0.15,
                validation_size=0.15,
                random_seed=self.config.random_seed,
            )
            self.state.train_indices = list(range(len(self.partitions.train_X)))
            self.state.val_indices = list(range(len(self.partitions.val_X)))
            self.state.test_indices = list(range(len(self.partitions.test_X)))
        except Exception as e:
            self.state.status = "failed"
            self.state.stop_reason = f"Data loading/partitioning failed: {e}"
            self.run_store.save_campaign_state(self.state)
            raise

        # ── Step 2: Profile Dataset ─────────────────────────────────────────
        profile = profile_dataset(df, self.config.target_column)
        self.state.dataset_profile = profile
        self.state.warnings.extend(profile.warnings)
        self.run_store.save_campaign_state(self.state)
        self._notify("dataset_profiled", {"profile": profile.model_dump(), "task_type": self.config.task_type})

        # ── Step 3: Train Benchmark Baseline ────────────────────────────────
        self.state.status = "baseline"
        baseline_family = "Ridge" if self.config.task_type == "regression" else "LogisticRegression"
        baseline_result, baseline_pipeline = train_baseline(
            partitions=self.partitions,
            model_family=baseline_family,
            task_type=self.config.task_type,
            objective_metric=self.config.objective_metric,
            hypothesis=f"Baseline benchmark using {baseline_family}.",
            reason=f"Standard benchmark for measuring subsequent non-linear gains on {self.config.objective_metric}.",
        )
        self.state.add_experiment(baseline_result)
        self.champion_pipeline = baseline_pipeline
        self.run_store.save_campaign_state(self.state)
        self._notify("baseline_trained", {
            "result": baseline_result.model_dump(),
            "objective_metric": self.config.objective_metric,
        })

        # ── Step 4: Autonomous Reasoning & Experimentation Loop ─────────────
        self.state.status = "running"
        stall_count = 0
        stall_threshold = 3
        last_best_score = self.state.get_score(baseline_result)
        action_number = 1
        eligible_families = list_families(self.config.task_type)

        while len(self.state.experiments) < self.config.max_experiments:
            self.state.update_elapsed()

            # Time budget check
            if self.state.time_remaining_seconds() <= 0:
                self.state.stop_reason = "Time budget exhausted"
                logger.info("Time budget reached limit. Stopping campaign.")
                break

            # Quality target check
            if (
                self.config.quality_target is not None
                and self.state.current_best
            ):
                current_score = self.state.get_score(self.state.current_best)
                if current_score is not None:
                    lower_better = self.config.objective_metric.lower() in ["rmse", "mse", "mae"]
                    target_met = (current_score <= self.config.quality_target) if lower_better else (current_score >= self.config.quality_target)
                    if target_met:
                        self.state.stop_reason = f"Quality target ({self.config.quality_target}) reached"
                        logger.info(self.state.stop_reason)
                        break

            # Stall check
            if stall_count >= stall_threshold:
                self.state.stop_reason = f"Performance plateaued: {stall_threshold} consecutive experiments without improvement"
                logger.info(self.state.stop_reason)
                break

            # Formulate prompt and call LLM
            messages = build_agent_context(
                state=self.state,
                recent_error_analysis=self.recent_error_analysis,
            )

            try:
                agent_action, llm_resp = self.llm_client.call_structured(
                    messages=messages,
                    schema_cls=AgentAction,
                    max_repair_attempts=2,
                )
            except Exception as e:
                logger.warning(f"Failed to get structured action from LLM: {e}")
                # Dynamic fallback family
                fallback_family = "RandomForestRegressor" if self.config.task_type == "regression" else "RandomForestClassifier"
                agent_action = AgentAction(
                    action="run_experiment",
                    model_family=fallback_family,
                    hypothesis="Automated fallback exploration of ensemble methods.",
                    reason="LLM response timed out or failed to parse.",
                )
                llm_resp = None

            t_action_start = time.perf_counter()
            tool_name = agent_action.action
            tool_result: dict = {}
            action_error: str | None = None

            # ── Dispatch to Tools ───────────────────────────────────────────
            if agent_action.action == "finish_campaign":
                self.state.stop_reason = f"Agent decided to conclude: {agent_action.reason}"
                logger.info(self.state.stop_reason)
                break

            elif agent_action.action == "analyze_errors":
                if self.champion_pipeline is not None and self.config.task_type == "classification":
                    err_res = analyze_errors(self.champion_pipeline, self.partitions)
                    self.recent_error_analysis = err_res.model_dump()
                    tool_result = self.recent_error_analysis
                    self._notify("error_analysis_completed", tool_result)
                else:
                    action_error = "Diagnostic analysis completed for current champion."

            elif agent_action.action == "compare_experiments":
                comp_res = compare_experiments(
                    self.state.experiments,
                    metric_name=self.config.objective_metric,
                )
                tool_result = comp_res.model_dump()
                self._notify("comparison_completed", tool_result)

            elif agent_action.action == "run_experiment":
                exp_num = len(self.state.experiments)
                family = agent_action.model_family

                # Ensure model family matches active task type
                if not family or family not in eligible_families:
                    family = eligible_families[min(exp_num, len(eligible_families) - 1)]

                exp_result, fitted_pipe = run_hpo_iteration(
                    partitions=self.partitions,
                    model_family=family,
                    hypothesis=agent_action.hypothesis,
                    reason=agent_action.reason,
                    experiment_number=exp_num,
                    parameters=agent_action.parameters,
                    n_trials=3,
                    objective_metric=self.config.objective_metric,
                    task_type=self.config.task_type,
                )

                self.state.add_experiment(exp_result)
                tool_result = exp_result.model_dump()

                # Check if this improved champion using dynamic direction
                current_score = self.state.get_score(exp_result)
                if exp_result.status == "completed" and current_score is not None:
                    if last_best_score is None or self.state.is_better(current_score, last_best_score):
                        last_best_score = current_score
                        stall_count = 0
                        if fitted_pipe is not None:
                            self.champion_pipeline = fitted_pipe
                        logger.info(f"New champion established! {family} with {self.config.objective_metric}={current_score:.4f}")
                    else:
                        stall_count += 1
                else:
                    stall_count += 1

                self._notify("experiment_completed", {
                    "result": tool_result,
                    "stall_count": stall_count,
                    "objective_metric": self.config.objective_metric,
                })

            duration_act = time.perf_counter() - t_action_start

            # Record action audit
            action_record = ActionRecord(
                action_number=action_number,
                timestamp=time.time(),
                proposed_action=agent_action.model_dump(),
                validated=True,
                tool_name=tool_name,
                tool_input=agent_action.parameters,
                tool_result=tool_result,
                provider_used=llm_resp.provider_used if llm_resp else "fallback",
                llm_latency_ms=llm_resp.latency_ms if llm_resp else 0.0,
                execution_duration_seconds=round(duration_act, 3),
                error=action_error,
            )
            self.state.actions.append(action_record)
            self.run_store.save_campaign_state(self.state)
            action_number += 1

        # ── Step 5: Untouched Isolated Test Set Evaluation ──────────────────
        self.state.status = "stopping"
        if not self.state.stop_reason:
            self.state.stop_reason = f"Completed all {self.config.max_experiments} planned experiments"

        test_metrics: dict = {}
        model_artifact_path: str | None = None

        if self.champion_pipeline is not None and self.partitions is not None:
            eval_out = evaluate_on_test(
                champion_pipeline=self.champion_pipeline,
                partitions=self.partitions,
                campaign_id=self.state.campaign_id,
                task_type=self.config.task_type,
            )
            test_metrics = eval_out.get("test_metrics", {})
            model_artifact_path = eval_out.get("model_artifact_path")
            self._notify("test_evaluation_completed", eval_out)

        self.state.status = "completed"
        self.state.update_elapsed()

        # ── Step 6: Generate Final Campaign Report ──────────────────────────
        completed_exps = self.state.get_completed_experiments()
        baseline_exp = self.state.experiments[0] if self.state.experiments else None

        report = CampaignReport(
            campaign_id=self.state.campaign_id,
            objective=f"Optimize {self.config.objective_metric} ({self.config.task_type})",
            dataset_info={
                "dataset_path": self.config.dataset_path,
                "target_column": self.config.target_column,
                "task_type": self.config.task_type,
                "n_rows": self.state.dataset_profile.n_rows if self.state.dataset_profile else 0,
                "n_columns": self.state.dataset_profile.n_columns if self.state.dataset_profile else 0,
            },
            baseline_result=baseline_exp,
            best_result=self.state.current_best,
            final_test_metrics=test_metrics,
            all_experiments=self.state.experiments,
            total_experiments=len(self.state.experiments),
            successful_experiments=len(completed_exps),
            failed_experiments=len(self.state.experiments) - len(completed_exps),
            total_duration_seconds=round(self.state.elapsed_seconds, 2),
            stop_reason=self.state.stop_reason or "Completed",
            warnings=self.state.warnings,
            model_artifact_path=model_artifact_path,
            reproducibility={
                "random_seed": self.config.random_seed,
                "train_samples": len(self.partitions.train_X) if self.partitions else 0,
                "val_samples": len(self.partitions.val_X) if self.partitions else 0,
                "test_samples": len(self.partitions.test_X) if self.partitions else 0,
            },
        )

        self.run_store.save_campaign_state(self.state)
        self.run_store.save_report(report)
        self._notify("campaign_completed", {"report": report.model_dump()})

        logger.info(f"Campaign {self.state.campaign_id} successfully concluded.")
        return report
