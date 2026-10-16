# 🧪 Autonomous ML Lab Agent

An autonomous Machine Learning Agent that translates conversational natural language prompts into end-to-end data science workflows. It profiles tabular datasets, dynamically detects task types (Classification, Regression, Clustering), formulates scientific hypotheses, tunes diverse algorithm families with Optuna, diagnoses prediction errors, and tracks experiments without human intervention.

---

## 🌟 Key Features & Capabilities

1. **💬 Conversational AI Chat Interface**
   - Interact in plain English: *"Focus on churn_flag target column, run for 20 minutes, explore 15 different options, and optimize for PR-AUC."*
   - Powered by an NLP Intent Parser ([`agent/nlp_parser.py`](file:///agent/nlp_parser.py)) that extracts task types, targets, metrics, and trial budgets with robust heuristic fallbacks.

2. **🧠 Dynamic Task & Metric Categorization**
   - **Classification**: Binary & Multiclass. Optimizes for `pr_auc`, `roc_auc`, `f1`, `accuracy`, `precision`, `recall`.
   - **Regression**: Continuous target prediction. Optimizes for `rmse`, `mae`, `mse`, `r2` with residual error diagnostics.
   - **Clustering**: Unsupervised grouping with `silhouette` and `davies_bouldin` metrics.
   - **Direction-Aware Champion Tracking**: Intelligently handles lower-is-better (e.g. RMSE, MAE) and higher-is-better (e.g. F1, PR-AUC).

3. **🌲 Broad Algorithm Exploration (Beyond Tree Ensembles)**
   - Explores across:
     - Linear models: `LogisticRegression`, `Ridge`, `Lasso`, `ElasticNet`, `LinearRegression`
     - Tree ensembles: `RandomForest`, `ExtraTrees`, `DecisionTree`
     - Boosting: `HistGradientBoosting`, `GradientBoosting`, `AdaBoost`
     - Distance & Kernel models: `KNeighbors`, `SVC`, `SVR`, `GaussianNB`
     - Clustering: `KMeans`, `DBSCAN`, `AgglomerativeClustering`
   - Automated parameter sanitization preventing framework deprecation crashes (e.g. `loss='deviance'` $\rightarrow$ `'log_loss'`, `max_features='auto'` $\rightarrow$ `'sqrt'`).

4. **🔄 Multi-Provider Resilience & Rotating LLM Pool**
   - **Groq Pool:** Automatically rotates across **3 Groq API keys** (`GROQ_API_KEY`, `GROQ_API_KEY2`, `GROQ_API_KEY3`) using round-robin and instant HTTP 429 rate-limit failover.
   - **NVIDIA NIM Fallback:** Automatic cascade to NVIDIA NIM API (`NVIDIA_API_KEY`) using Meta Llama models (`meta/llama-3.2-11b-vision-instruct`).
   - **OpenRouter Fallback:** Free model cascade (`google/gemma-4-31b-it:free`).
   - **Local Ollama Fallback:** Offline safety net (`localhost:11434`).

5. **🛡️ Strict LLM / Deterministic Boundary & Leakage-Proof Splits**
   - The LLM **never** touches or fabricates metrics. All metrics are deterministically computed by scikit-learn from actual predictions and ground-truth labels.
   - Guardrails: All proposed actions must validate against strict Pydantic schemas before any tool execution.
   - **Isolated Test Set:** Evaluated strictly once at the conclusion of the campaign on the champion model to prevent data leakage.

6. **🖥️ Interactive Streamlit Web Studio**
   - **Tab 1: 💬 AI Agent Chat**: Conversational interface with quick prompts, live progress, and reasoning logs.
   - **Tab 2: 🔬 Studio & Deep Analytics**: Dynamic KPI cards reflecting your selected metric, interactive Plotly learning curves, model leaderboard with CSV export, confusion matrix / regression residual plots, and downloadable trained `.joblib` model artifacts.

---

## 🚀 Quickstart

### 1. Environment & API Keys
Copy the example environment file and add your keys (at minimum one API key or local Ollama):
```bash
cp .env.example .env
```

```env
GROQ_API_KEY=gsk_...
GROQ_API_KEY2=gsk_...
GROQ_API_KEY3=gsk_...
NVIDIA_API_KEY=nvapi-...
OPENROUTER_API_KEY=sk-or-v1-...
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Launch the Streamlit Web Studio
```bash
streamlit run ui/app.py
```
Open [http://localhost:8501](http://localhost:8501) in your browser.

### 4. Run a Campaign via Python CLI
```python
from agent.orchestrator import AgentOrchestrator
from agent.schemas import CampaignConfig

config = CampaignConfig(
    dataset_path="breast_cancer",  # or "data/bank_marketing.csv", "diabetes", or custom CSV
    target_column="target",
    task_type="classification",    # or "regression", "auto"
    objective_metric="pr_auc",     # or "f1", "rmse", "r2", etc.
    max_experiments=8,
    max_time_minutes=15,
    random_seed=42,
)

orchestrator = AgentOrchestrator(config)
report = orchestrator.run_campaign()

print(f"Champion: {report.best_result.model_family}")
print(f"Primary Metric: {report.best_result.primary_metric_name} = {report.best_result.primary_metric_value:.4f}")
print(f"Final Test Metrics: {report.final_test_metrics}")
```

### 5. Run Automated Tests
```bash
pytest -v
```
All 21 unit and integration tests run in under 4 seconds.

