"""
Experiment storage and persistence engine for ML Lab Agent.

Stores:
- SQLite database (`experiments/ml_lab_agent.db`) for fast queries and analytics
- JSON snapshots (`experiments/<campaign_id>/state.json`)
- Markdown final reports (`experiments/<campaign_id>/report.md`)
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import time
from typing import Any

from agent.schemas import CampaignReport, CampaignState, ExperimentResult

logger = logging.getLogger(__name__)


class RunStore:
    """
    Manages persistence of campaigns, experiments, and metrics.
    """

    def __init__(self, base_dir: str = "experiments", db_name: str = "ml_lab_agent.db"):
        self.base_dir = base_dir
        os.makedirs(self.base_dir, exist_ok=True)
        self.db_path = os.path.join(self.base_dir, db_name)
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        """Create database tables if they do not exist."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            # Campaigns table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS campaigns (
                    campaign_id TEXT PRIMARY KEY,
                    dataset_path TEXT,
                    target_column TEXT,
                    objective_metric TEXT,
                    status TEXT,
                    start_time REAL,
                    end_time REAL,
                    total_experiments INTEGER,
                    best_run_id TEXT,
                    best_metric REAL,
                    stop_reason TEXT,
                    config_json TEXT
                )
            """)
            # Experiments table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS experiments (
                    run_id TEXT PRIMARY KEY,
                    campaign_id TEXT,
                    experiment_number INTEGER,
                    model_family TEXT,
                    status TEXT,
                    val_f1 REAL,
                    val_precision REAL,
                    val_recall REAL,
                    val_accuracy REAL,
                    val_pr_auc REAL,
                    train_f1 REAL,
                    train_accuracy REAL,
                    duration_seconds REAL,
                    hypothesis TEXT,
                    reason TEXT,
                    parameters_json TEXT,
                    error_message TEXT,
                    created_at REAL,
                    FOREIGN KEY (campaign_id) REFERENCES campaigns (campaign_id)
                )
            """)
            # Actions audit log table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS audit_actions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    campaign_id TEXT,
                    action_number INTEGER,
                    timestamp REAL,
                    tool_name TEXT,
                    provider_used TEXT,
                    llm_latency_ms REAL,
                    execution_duration REAL,
                    proposed_action_json TEXT,
                    tool_result_json TEXT,
                    error TEXT,
                    FOREIGN KEY (campaign_id) REFERENCES campaigns (campaign_id)
                )
            """)
            conn.commit()

    def save_campaign_state(self, state: CampaignState) -> None:
        """Persist campaign state to SQLite and JSON snapshot."""
        # Save JSON snapshot
        campaign_dir = os.path.join(self.base_dir, state.campaign_id)
        os.makedirs(campaign_dir, exist_ok=True)
        json_path = os.path.join(campaign_dir, "state.json")
        with open(json_path, "w", encoding="utf-8") as f:
            f.write(state.model_dump_json(indent=2))

        # Save to SQLite
        best_run_id = state.current_best.run_id if state.current_best else None
        best_metric = getattr(state.current_best, "validation_f1", None) if state.current_best else None

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO campaigns (
                    campaign_id, dataset_path, target_column, objective_metric,
                    status, start_time, end_time, total_experiments,
                    best_run_id, best_metric, stop_reason, config_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(campaign_id) DO UPDATE SET
                    status=excluded.status,
                    end_time=excluded.end_time,
                    total_experiments=excluded.total_experiments,
                    best_run_id=excluded.best_run_id,
                    best_metric=excluded.best_metric,
                    stop_reason=excluded.stop_reason
            """, (
                state.campaign_id,
                state.config.dataset_path,
                state.config.target_column,
                state.config.objective_metric,
                state.status,
                state.start_time,
                time.time() if state.status in ["completed", "failed"] else None,
                len(state.experiments),
                best_run_id,
                best_metric,
                state.stop_reason,
                state.config.model_dump_json(),
            ))

            # Upsert experiments
            for exp in state.experiments:
                cursor.execute("""
                    INSERT INTO experiments (
                        run_id, campaign_id, experiment_number, model_family,
                        status, val_f1, val_precision, val_recall,
                        val_accuracy, val_pr_auc, train_f1, train_accuracy,
                        duration_seconds, hypothesis, reason, parameters_json,
                        error_message, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(run_id) DO UPDATE SET
                        status=excluded.status,
                        val_f1=excluded.val_f1,
                        val_precision=excluded.val_precision,
                        val_recall=excluded.val_recall,
                        val_accuracy=excluded.val_accuracy,
                        val_pr_auc=excluded.val_pr_auc,
                        train_f1=excluded.train_f1,
                        train_accuracy=excluded.train_accuracy,
                        duration_seconds=excluded.duration_seconds
                """, (
                    exp.run_id,
                    state.campaign_id,
                    exp.experiment_number,
                    exp.model_family,
                    exp.status,
                    exp.validation_f1,
                    exp.validation_precision,
                    exp.validation_recall,
                    exp.validation_accuracy,
                    exp.validation_pr_auc,
                    exp.training_f1,
                    exp.training_accuracy,
                    exp.duration_seconds,
                    exp.hypothesis,
                    exp.reason,
                    json.dumps(exp.parameters),
                    exp.error_message,
                    time.time(),
                ))

            # Log any new audit actions
            for action in state.actions:
                cursor.execute("""
                    INSERT INTO audit_actions (
                        campaign_id, action_number, timestamp, tool_name,
                        provider_used, llm_latency_ms, execution_duration,
                        proposed_action_json, tool_result_json, error
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    state.campaign_id,
                    action.action_number,
                    action.timestamp,
                    action.tool_name,
                    action.provider_used,
                    action.llm_latency_ms,
                    action.execution_duration_seconds,
                    json.dumps(action.proposed_action),
                    json.dumps(action.tool_result),
                    action.error,
                ))

            conn.commit()

    def save_report(self, report: CampaignReport) -> str:
        """Save final campaign report in JSON and Markdown formats."""
        campaign_dir = os.path.join(self.base_dir, report.campaign_id)
        os.makedirs(campaign_dir, exist_ok=True)

        json_path = os.path.join(campaign_dir, "report.json")
        with open(json_path, "w", encoding="utf-8") as f:
            f.write(report.model_dump_json(indent=2))

        md_path = os.path.join(campaign_dir, "report.md")
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(self.generate_markdown_report(report))

        logger.info(f"Report saved to {md_path}")
        return md_path

    @staticmethod
    def generate_markdown_report(report: CampaignReport) -> str:
        """Render a readable Markdown document summarizing the campaign."""
        best = report.best_result
        base = report.baseline_result

        base_score = base.primary_metric_value if base and base.primary_metric_value is not None else getattr(base, "validation_f1", None)
        best_score = best.primary_metric_value if best and best.primary_metric_value is not None else getattr(best, "validation_f1", None)

        base_str = f"{base_score:.4f}" if base_score is not None else "N/A"
        best_str = f"{best_score:.4f}" if best_score is not None else "N/A"
        test_metric_val = None
        for k, v in report.final_test_metrics.items():
            if isinstance(v, (int, float)):
                test_metric_val = v
                break
        test_str = f"{test_metric_val:.4f}" if test_metric_val is not None else "N/A"

        lines = [
            f"# ML Lab Agent — Campaign Summary Report",
            f"**Campaign ID:** `{report.campaign_id}`  ",
            f"**Objective:** {report.objective}  ",
            f"**Stop Reason:** {report.stop_reason}  ",
            f"**Total Duration:** {report.total_duration_seconds:.1f}s  ",
            "",
            "## Key Results",
            f"- **Baseline Score:** {base_str}",
            f"- **Best Model ({best.model_family if best else 'None'}):** Validation Score = {best_str}",
            f"- **Final Test Set Score (Isolated):** {test_str}",
            f"- **Saved Model Artifact:** `{report.model_artifact_path}`",
            "",
            "## Experiment History",
            "| # | Model | Score | Status | Duration (s) | Hypothesis |",
            "|---|---|---|---|---|---|",
        ]

        for e in report.all_experiments:
            score = e.primary_metric_value if e.primary_metric_value is not None else e.validation_f1
            score_str = f"{score:.4f}" if score is not None else "-"
            lines.append(
                f"| {e.experiment_number} | {e.model_family} | {score_str} | {e.status} | {e.duration_seconds:.1f} | {e.hypothesis} |"
            )

        if report.warnings:
            lines.extend([
                "",
                "## Warnings & Diagnoses",
                *[f"- {w}" for w in report.warnings],
            ])

        return "\n".join(lines)

    def load_campaign_state(self, campaign_id: str) -> CampaignState | None:
        """Load campaign state from JSON snapshot."""
        json_path = os.path.join(self.base_dir, campaign_id, "state.json")
        if not os.path.exists(json_path):
            return None
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return CampaignState.model_validate(data)

    def get_campaign_experiments(self, campaign_id: str) -> list[dict[str, Any]]:
        """Fetch all experiments for a campaign from SQLite."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT * FROM experiments WHERE campaign_id = ? ORDER BY experiment_number ASC",
                (campaign_id,)
            )
            rows = cursor.fetchall()
            return [dict(r) for r in rows]

    def list_campaigns(self) -> list[dict[str, Any]]:
        """Fetch summary of all campaigns from SQLite."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT * FROM campaigns ORDER BY start_time DESC"
            )
            rows = cursor.fetchall()
            return [dict(r) for r in rows]
