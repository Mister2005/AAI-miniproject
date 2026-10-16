"""
ML Lab Agent — Conversational Autonomous AI Data Scientist & Experiment Studio.

Features:
- Conversational Chat Interface: Give high-level NLP instructions (e.g. "Focus on target column 'y',
  run for 30 minutes, explore 15-20 models, optimize for PR-AUC")
- Dynamic Task Detection: Auto-detects Classification (Binary/Multiclass) vs Regression
- Dynamic Metric Architecture: Accurately optimizes & displays F1, PR-AUC, ROC-AUC, Accuracy, RMSE, or R2
- CSV File Upload with instant column profiling and preview
- Live multi-provider API status indicator (Groq 3-key pool, NVIDIA NIM, OpenRouter, Ollama)
- Interactive Plotly metric charts, confusion matrices, audit logs, and .joblib model export
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Ensure project root is in sys.path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from dotenv import load_dotenv
env_path = root_dir / ".env"
load_dotenv(dotenv_path=env_path)

from agent.llm_client import LLMClient
from agent.nlp_parser import parse_user_prompt
from agent.orchestrator import AgentOrchestrator
from agent.schemas import CampaignConfig, CampaignReport
from ml.datasets import load_dataset
from storage.run_store import RunStore

st.set_page_config(
    page_title="ML Lab Agent — Autonomous AI Data Scientist",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom modern CSS
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        background: linear-gradient(90deg, #6366f1, #a855f7, #ec4899);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.1rem;
    }
    .sub-title {
        color: #94a3b8;
        font-size: 0.95rem;
        margin-bottom: 1.2rem;
    }
    .kpi-card {
        background: rgba(30, 41, 59, 0.7);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 14px;
        text-align: center;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    .kpi-value {
        font-size: 1.7rem;
        font-weight: 700;
        color: #38bdf8;
    }
    .kpi-label {
        font-size: 0.8rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-top: 4px;
    }
    .chat-bubble {
        padding: 12px 16px;
        border-radius: 12px;
        margin-bottom: 10px;
        line-height: 1.5;
    }
</style>
""", unsafe_allow_html=True)

run_store = RunStore()
llm_client = LLMClient()

# Session State Initialization
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": (
                "👋 Hello! I am your **ML Lab Agent**.\n\n"
                "Upload a CSV dataset or pick one from the sidebar, then tell me what you'd like to achieve in plain English. "
                "For example:\n"
                "- *'Focus on the target column, run for 20 minutes, explore 12 different options, and optimize for PR-AUC.'*\n"
                "- *'Predict SalePrice with RMSE across diverse linear and boosting models.'*"
            ),
        }
    ]
if "last_report" not in st.session_state:
    st.session_state.last_report = None
if "current_df" not in st.session_state:
    st.session_state.current_df = None
if "active_dataset_path" not in st.session_state:
    st.session_state.active_dataset_path = "breast_cancer"

# ─────────────────────────────────────────────
# Sidebar: System Status & Data Setup
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🧪 ML Lab Agent")
    st.caption("Autonomous tabular modeling & HPO")

    st.markdown("---")
    st.markdown("#### ⚡ LLM Key Pool Status")

    groq_keys = [k for k in [os.getenv("GROQ_API_KEY"), os.getenv("GROQ_API_KEY2"), os.getenv("GROQ_API_KEY3")] if k]
    nv_key = os.getenv("NVIDIA_API_KEY")
    or_key = os.getenv("OPENROUTER_API_KEY")

    st.markdown(f"**Groq:** 🟢 {len(groq_keys)} Active Keys (Pooled & Rotating)")
    st.markdown(f"**NVIDIA NIM:** {'🟢 1 Key Active' if nv_key else '⚪ Inactive'}")
    st.markdown(f"**OpenRouter:** {'🟢 1 Key Active' if or_key else '⚪ Inactive'}")
    st.markdown("**Ollama:** ⚪ Local Fallback")

    st.markdown("---")
    st.markdown("#### 📁 Dataset Selection")

    data_option = st.radio(
        "Choose Data Source:",
        options=["Upload Custom CSV", "bank_marketing (UCI Imbalanced Tabular)", "breast_cancer (Classification Demo)"],
        index=0,
    )

    df_loaded = None
    resolved_path = ""

    if data_option == "Upload Custom CSV":
        uploaded_file = st.file_uploader("Upload your CSV dataset", type=["csv"])
        if uploaded_file is not None:
            save_dir = Path("data")
            save_dir.mkdir(exist_ok=True)
            resolved_path = str(save_dir / uploaded_file.name)
            with open(resolved_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            try:
                df_loaded = pd.read_csv(resolved_path)
                st.session_state.current_df = df_loaded
                st.session_state.active_dataset_path = resolved_path
                st.success(f"Loaded `{uploaded_file.name}` ({len(df_loaded)} rows, {len(df_loaded.columns)} cols)")
            except Exception as e:
                st.error(f"Error reading CSV: {e}")
    elif data_option == "bank_marketing (UCI Imbalanced Tabular)":
        resolved_path = "bank_marketing"
        st.session_state.active_dataset_path = resolved_path
        try:
            df_loaded = load_dataset("bank_marketing")
            st.session_state.current_df = df_loaded
            st.info("Loaded Bank Marketing (45,211 rows, target: 'y')")
        except Exception as e:
            st.warning(f"Fetching bank marketing: {e}")
    else:
        resolved_path = "breast_cancer"
        st.session_state.active_dataset_path = resolved_path
        try:
            df_loaded = load_dataset("breast_cancer")
            st.session_state.current_df = df_loaded
            st.info("Loaded Breast Cancer (569 rows, target: 'target')")
        except Exception as e:
            st.warning(f"Error: {e}")

    # Dataset column preview
    if st.session_state.current_df is not None:
        with st.expander("👀 View Dataset Preview"):
            st.dataframe(st.session_state.current_df.head(4))

# ─────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────
st.markdown('<div class="main-title">ML Lab Agent</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Conversational Autonomous Machine Learning with Multi-Key Pool & Closed-Loop Reasoning</div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Primary Tabs: Chat Assistant & Experiment Studio
# ─────────────────────────────────────────────
tab_chat, tab_studio = st.tabs(["💬 AI Agent Chat", "🔬 Experiment Studio & Analytics"])


def execute_campaign_from_config(config: CampaignConfig, progress_placeholder, status_placeholder, log_container):
    """Run campaign with live callback updates."""
    logs = []

    def ui_callback(event_type: str, data: dict):
        metric_name = config.objective_metric
        if event_type == "campaign_started":
            logs.append(f"🏁 Campaign `{data.get('campaign_id')}` initiated.")
            progress_placeholder.progress(5, text="Loading and partitioning dataset...")
        elif event_type == "dataset_profiled":
            prof = data.get("profile", {})
            task_type = data.get("task_type", "classification")
            logs.append(
                f"📊 Profiled dataset: {prof.get('n_rows')} rows, {prof.get('n_columns')} columns. "
                f"Task detected: **{task_type.upper()}**."
            )
            progress_placeholder.progress(15, text="Training benchmark baseline...")
        elif event_type == "baseline_trained":
            res = data.get("result", {})
            score = res.get("primary_metric_value")
            if score is None:
                score = res.get(f"validation_{metric_name}") or res.get("validation_f1")
            score_str = f"{score:.4f}" if score is not None else "N/A"
            logs.append(
                f"🎯 Baseline ({res.get('model_family')}): Validation {metric_name} = **{score_str}**"
            )
            progress_placeholder.progress(25, text="Entering autonomous agent loop...")
        elif event_type == "experiment_completed":
            res = data.get("result", {})
            exp_num = res.get("experiment_number", 0)
            score = res.get("primary_metric_value")
            if score is None:
                score = res.get(f"validation_{metric_name}") or res.get("validation_f1")
            score_str = f"{score:.4f}" if score is not None else "N/A"
            pct = min(90, int(25 + (exp_num / config.max_experiments) * 65))
            logs.append(
                f"🧪 Exp #{exp_num} ({res.get('model_family')}): {metric_name} = **{score_str}** | "
                f"Hypothesis: *'{res.get('hypothesis', '')}'*"
            )
            progress_placeholder.progress(pct, text=f"Experiment #{exp_num} done. Agent reasoning next action...")
        elif event_type == "error_analysis_completed":
            obs = data.get("observations", [])
            logs.append(f"🔎 Diagnostic error analysis: {len(obs)} patterns fed into next hypothesis.")
        elif event_type == "test_evaluation_completed":
            metrics = data.get("test_metrics", {})
            t_score = metrics.get(metric_name) or metrics.get("f1") or 0.0
            logs.append(f"🏆 Final Isolated Test Evaluation: {metric_name} = **{t_score:.4f}**")
            progress_placeholder.progress(100, text="Campaign completed!")

        log_container.markdown("\n\n".join(f"- {msg}" for msg in logs[-6:]))

    orchestrator = AgentOrchestrator(
        config=config,
        run_store=run_store,
        on_step_callback=ui_callback,
    )
    report = orchestrator.run_campaign()
    st.session_state.last_report = report
    return report


# ─────────────────────────────────────────────
# TAB 1: Conversational Chat Interface
# ─────────────────────────────────────────────
with tab_chat:
    st.markdown("#### Talk to your ML Agent")
    st.caption("Instruct the agent using plain English. It understands target columns, metrics, time limits, and model exploration strategies.")

    # Render Chat History
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Quick prompt shortcuts
    st.markdown("##### 💡 Suggested Instructions:")
    col_q1, col_q2, col_q3 = st.columns(3)
    quick_prompt = None
    with col_q1:
        if st.button("⚡ Auto-detect and run 8 diverse models", key="q1"):
            quick_prompt = "Auto-detect the task type and run 8 diverse experiments to find the best model."
    with col_q2:
        if st.button("🎯 Focus on target, optimize for PR-AUC", key="q2"):
            quick_prompt = "Focus on the target column, run for 15 minutes with 10 experiments, and optimize for PR-AUC."
    with col_q3:
        if st.button("🌲 Explore Tree Ensembles & Boosting", key="q3"):
            quick_prompt = "Focus on tree-based models and gradient boosting. Run 12 experiments."

    user_input = st.chat_input("Enter instructions for your ML agent (e.g. 'Focus on target column y, run 10 experiments, maximize PR-AUC')...")
    prompt_to_process = user_input or quick_prompt

    if prompt_to_process:
        # Display user message
        st.session_state.messages.append({"role": "user", "content": prompt_to_process})
        with st.chat_message("user"):
            st.markdown(prompt_to_process)

        # Parse with NLP parser
        df_for_nlp = st.session_state.current_df
        default_ds_path = st.session_state.active_dataset_path

        with st.chat_message("assistant"):
            with st.spinner("Analyzing your request and profiling dataset..."):
                campaign_cfg, agent_reply = parse_user_prompt(
                    user_prompt=prompt_to_process,
                    df=df_for_nlp,
                    llm_client=llm_client,
                    default_config=CampaignConfig(
                        dataset_path=default_ds_path,
                        target_column="target",
                    ),
                )
                campaign_cfg.dataset_path = default_ds_path

            # Show agent response breakdown
            st.markdown(f"**🤖 Agent Strategy & Understanding:**\n\n{agent_reply}")
            st.markdown(
                f"**Task Plan:** `{campaign_cfg.task_type.upper()}` | "
                f"**Target:** `{campaign_cfg.target_column}` | "
                f"**Metric:** `{campaign_cfg.objective_metric}` | "
                f"**Budget:** `{campaign_cfg.max_experiments}` experiments / `{campaign_cfg.max_time_minutes}` min"
            )

            # Execution container
            progress_bar = st.progress(0, text="Starting autonomous run...")
            status_box = st.empty()
            log_box = st.container()

            try:
                report = execute_campaign_from_config(
                    config=campaign_cfg,
                    progress_placeholder=progress_bar,
                    status_placeholder=status_box,
                    log_container=log_box,
                )

                best_m = report.best_result
                metric_name = campaign_cfg.objective_metric
                best_val = best_m.primary_metric_value if best_m and best_m.primary_metric_value is not None else getattr(best_m, "validation_f1", 0.0)
                test_val = report.final_test_metrics.get(metric_name) or report.final_test_metrics.get("f1") or 0.0

                conclusion_msg = (
                    f"✅ **Campaign Concluded!**\n\n"
                    f"- **Stop Reason:** {report.stop_reason}\n"
                    f"- **Winning Model:** `{best_m.model_family if best_m else 'None'}`\n"
                    f"- **Validation {metric_name.upper()}:** `{best_val:.4f}`\n"
                    f"- **Isolated Test {metric_name.upper()}:** `{test_val:.4f}`\n"
                    f"- **Model Artifact:** `{report.model_artifact_path}`\n\n"
                    f"👉 *Check the **Experiment Studio & Analytics** tab for the full progression curves, leaderboard, and one-click model download!*"
                )
                st.markdown(conclusion_msg)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": f"{agent_reply}\n\n{conclusion_msg}",
                })

            except Exception as e:
                st.error(f"Error during campaign execution: {e}")
                st.exception(e)

# ─────────────────────────────────────────────
# TAB 2: Experiment Studio & Analytics
# ─────────────────────────────────────────────
with tab_studio:
    report: CampaignReport | None = st.session_state.last_report

    # If no report in session, load the most recent from RunStore
    if report is None:
        campaigns = run_store.list_campaigns()
        if campaigns:
            most_recent_id = campaigns[0]["campaign_id"]
            state = run_store.load_campaign_state(most_recent_id)
            if state:
                metric_name = state.config.objective_metric
                report = CampaignReport(
                    campaign_id=state.campaign_id,
                    objective=f"Optimize {metric_name}",
                    dataset_info={"dataset_path": state.config.dataset_path, "target_column": state.config.target_column},
                    baseline_result=state.experiments[0] if state.experiments else None,
                    best_result=state.current_best,
                    all_experiments=state.experiments,
                    total_experiments=len(state.experiments),
                    successful_experiments=len(state.get_completed_experiments()),
                    failed_experiments=len([e for e in state.experiments if e.status != "completed"]),
                    total_duration_seconds=round(state.elapsed_seconds, 1),
                    stop_reason=state.stop_reason or "Completed",
                    warnings=state.warnings,
                )

    if report is not None:
        best = report.best_result
        base = report.baseline_result

        # Identify active metric name
        metric_name = "f1"
        if best and best.primary_metric_name:
            metric_name = best.primary_metric_name

        base_val = base.primary_metric_value if base and base.primary_metric_value is not None else getattr(base, "validation_f1", None)
        best_val = best.primary_metric_value if best and best.primary_metric_value is not None else getattr(best, "validation_f1", None)

        test_val = report.final_test_metrics.get(metric_name)
        if test_val is None:
            for k, v in report.final_test_metrics.items():
                if isinstance(v, (int, float)):
                    test_val = v
                    break

        base_str = f"{base_val:.4f}" if base_val is not None else "N/A"
        best_str = f"{best_val:.4f}" if best_val is not None else "N/A"
        test_str = f"{test_val:.4f}" if test_val is not None else "N/A"

        st.markdown(f"### 📊 Campaign Performance (`{report.campaign_id}`)")
        st.caption(f"Objective: **{report.objective}** | Stop Reason: {report.stop_reason}")

        # ── Dynamic KPI Cards ────────────────────────────────────────────────────
        c1, c2, c3, c4, c5 = st.columns(5)
        with c1:
            st.markdown(f'<div class="kpi-card"><div class="kpi-value">{base_str}</div><div class="kpi-label">Baseline {metric_name.upper()}</div></div>', unsafe_allow_html=True)
        with c2:
            st.markdown(f'<div class="kpi-card"><div class="kpi-value" style="color: #4ade80;">{best_str}</div><div class="kpi-label">Champion {metric_name.upper()}</div></div>', unsafe_allow_html=True)
        with c3:
            st.markdown(f'<div class="kpi-card"><div class="kpi-value" style="color: #ec4899;">{test_str}</div><div class="kpi-label">Test {metric_name.upper()} (Isolated)</div></div>', unsafe_allow_html=True)
        with c4:
            champ_name = best.model_family if best else "N/A"
            st.markdown(f'<div class="kpi-card"><div class="kpi-value" style="font-size: 1.1rem; padding-top: 8px;">{champ_name}</div><div class="kpi-label">Champion Model</div></div>', unsafe_allow_html=True)
        with c5:
            st.markdown(f'<div class="kpi-card"><div class="kpi-value" style="color: #fbbf24;">{report.total_duration_seconds:.1f}s</div><div class="kpi-label">Runtime</div></div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Studio Sub-Tabs ──────────────────────────────────────────────────────
        sub_chart, sub_lead, sub_diag, sub_audit, sub_export = st.tabs([
            "📈 Metric Progression",
            "🏆 Model Leaderboard",
            "🔍 Diagnostics",
            "📜 Agent Audit Trail",
            "💾 Download Model",
        ])

        with sub_chart:
            if report.all_experiments:
                exp_data = []
                for e in report.all_experiments:
                    if e.status == "completed":
                        score = e.primary_metric_value if e.primary_metric_value is not None else e.validation_f1
                        train_score = e.training_metrics.get(metric_name) or e.training_f1
                        exp_data.append({
                            "Experiment": f"#{e.experiment_number} ({e.model_family})",
                            "Experiment Number": e.experiment_number,
                            f"Validation {metric_name.upper()}": score,
                            f"Training {metric_name.upper()}": train_score,
                            "Model Family": e.model_family,
                        })
                df_plot = pd.DataFrame(exp_data)
                if not df_plot.empty:
                    fig = px.line(
                        df_plot,
                        x="Experiment Number",
                        y=[f"Validation {metric_name.upper()}", f"Training {metric_name.upper()}"],
                        markers=True,
                        title=f"Validation vs Training {metric_name.upper()} Across Experiments (Overfitting Detection)",
                        color_discrete_sequence=["#38bdf8", "#f43f5e"],
                    )
                    fig.update_layout(template="plotly_dark", hovermode="x unified")
                    st.plotly_chart(fig, use_container_width=True)

        with sub_lead:
            if report.all_experiments:
                rows = []
                for e in report.all_experiments:
                    score = e.primary_metric_value if e.primary_metric_value is not None else e.validation_f1
                    rows.append({
                        "Exp #": e.experiment_number,
                        "Model": e.model_family,
                        f"{metric_name.upper()} Score": round(score, 4) if score is not None else None,
                        "Precision": round(e.validation_precision, 4) if e.validation_precision is not None else None,
                        "Recall": round(e.validation_recall, 4) if e.validation_recall is not None else None,
                        "Accuracy": round(e.validation_accuracy, 4) if e.validation_accuracy is not None else None,
                        "Status": e.status,
                        "Duration (s)": e.duration_seconds,
                        "Hypothesis": e.hypothesis,
                    })
                lower_better = metric_name.lower() in ["rmse", "mse", "mae"]
                df_lead = pd.DataFrame(rows).sort_values(by=f"{metric_name.upper()} Score", ascending=lower_better)
                st.dataframe(df_lead, use_container_width=True)

        with sub_diag:
            cm = report.final_test_metrics.get("confusion_matrix")
            if cm is None and best and best.confusion_matrix:
                cm = best.confusion_matrix

            if cm:
                st.markdown("#### Test Set Confusion Matrix")
                fig_cm = px.imshow(
                    cm,
                    x=["Predicted Negative", "Predicted Positive"],
                    y=["Actual Negative", "Actual Positive"],
                    color_continuous_scale="Blues",
                    text_auto=True,
                    title="Confusion Matrix Breakdown",
                )
                fig_cm.update_layout(template="plotly_dark")
                st.plotly_chart(fig_cm, use_container_width=True)
            elif "residual_mean" in report.final_test_metrics:
                st.markdown("#### Regression Residual Diagnostics")
                st.metric("Residual Mean", f"{report.final_test_metrics.get('residual_mean', 0.0):.4f}")
                st.metric("Residual Std Dev", f"{report.final_test_metrics.get('residual_std', 0.0):.4f}")
            else:
                st.info("No confusion matrix or residual diagnostics available.")

        with sub_audit:
            state_for_audit = run_store.load_campaign_state(report.campaign_id)
            if state_for_audit and state_for_audit.actions:
                st.markdown(f"#### Audit Log ({len(state_for_audit.actions)} Actions Recorded)")
                for act in state_for_audit.actions:
                    with st.expander(f"Action #{act.action_number}: `{act.tool_name}` via `{act.provider_used}` ({act.llm_latency_ms:.0f}ms)"):
                        c_a1, c_a2 = st.columns(2)
                        with c_a1:
                            st.markdown("**Proposed Action by LLM:**")
                            st.json(act.proposed_action)
                        with c_a2:
                            st.markdown("**Deterministic Tool Execution Result:**")
                            st.json(act.tool_result)

        with sub_export:
            st.markdown("#### Export Winning Champion Pipeline")
            art_path = report.model_artifact_path
            if art_path and os.path.exists(art_path):
                with open(art_path, "rb") as f:
                    model_bytes = f.read()
                st.download_button(
                    label=f"⬇️ Download {os.path.basename(art_path)}",
                    data=model_bytes,
                    file_name=os.path.basename(art_path),
                    mime="application/octet-stream",
                )
                st.caption(f"Full trained pipeline artifact saved at: `{art_path}`")
            else:
                st.info("No saved model artifact found on disk.")
    else:
        st.info("No campaign has been run yet. Use the **💬 AI Agent Chat** tab to instruct the agent and launch your first experiment!")

# ─────────────────────────────────────────────
# Past Campaigns Browser
# ─────────────────────────────────────────────
st.markdown("---")
with st.expander("📚 Past Campaign History"):
    past = run_store.list_campaigns()
    if past:
        st.dataframe(pd.DataFrame(past)[["campaign_id", "dataset_path", "objective_metric", "best_metric", "total_experiments", "status", "stop_reason"]], use_container_width=True)
    else:
        st.caption("No historical campaigns recorded yet.")
