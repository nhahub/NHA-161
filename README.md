# Customer Churn Analysis Project

End-to-end workflow for understanding and predicting customer churn: data exploration, balancing, multi-model training (Python + R), SQL analytics, and a FastAPI web demo for quick scoring.

## At a Glance
- **Dataset:** 202,611 rows × 13 columns (no critical nulls) sourced from `data/raw_data.xlsx`.
- **Pipeline:** Explore → Clean → Balance → Model → Serve → Visualize.
- **Top Model:** Gradient Boosting (accuracy ≈ 0.799, F1 ≈ 0.888) edges out Logistic Regression & Random Forest by <0.001.
- **Key Drivers:** Higher churn in Electronics/Home categories, slightly younger cohorts, and customers with recent returns.

## Tech Stack
- **Python (3.11):** Notebooks, feature prep, FastAPI service, SHAP/visual helpers.
- **R:** `scripts/` for SMOTE balancing, modeling, and reporting.
- **SQL Server dialect:** Schema + analysis queries in `sql/SQLQuery.sql`.
- **FastAPI + Vanilla JS:** REST API (`web/api_server.py`) and UI (`web/customer.html`).

## Repository Map
| Folder | Purpose |
| --- | --- |
| `notebooks/` | `EDA Notebook with visualizations.ipynb` (plots + TL;DR) and `training_churn_models.ipynb` (scikit-learn bakeoff). |
| `scripts/` | `data_cleaning.R`, `train_churn_models.R`, and vendored `.r-lib/` packages for reproducible R installs. |
| `data/` | Excel sources (`raw_data.xlsx`, `train.xlsx`, balanced variants, tests). Not committed publicly. |
| `sql/` | Schema creation, analysis queries, and BI-ready `CustomerChurnReport` definition. |
| `web/` | FastAPI app, HTML client, payload samples, and cached model artifacts. |
| `docs/` | `project_brief.md` summarizing planning, stakeholders, DB design, and screenshot guidance. |

## Workflow Overview
1. **Explore & Validate** — Run `notebooks/EDA Notebook with visualizations.ipynb` to inspect imbalance, demographic trends, and product/payment signals.
2. **Clean & Balance** — Execute `scripts/data_cleaning.R` then leverage SMOTE-balanced files (`data/train_balanced.xlsx`) for stable modeling.
3. **Model & Compare** — Use `scripts/train_churn_models.R` or `notebooks/training_churn_models.ipynb` to benchmark Logistic Regression, Random Forest, and Gradient Boosting.
4. **Persist & Analyze** — Load curated tables via `sql/SQLQuery.sql` for BI dashboards (spend by churn status, payment preferences, etc.).
5. **Serve & Demo** — FastAPI service + `web/customer.html` provide real-time predictions with cached model weights (`model_cache.joblib`).

## Quick Start
### 1. Clone & Configure
```cmd
git clone <repo-url>
cd project\1
```

### 2. Python Environment
```cmd
pip install -r requirements.txt
```

### 3. R Environment (optional but recommended)
```cmd
Rscript scripts\data_cleaning.R --help
```
The first run installs dependencies into `scripts/.r-lib/` for repeatable use.

### 4. Reproduce the Pipeline
1. **EDA:** Open the notebooks in VS Code/Jupyter; update the `data_path` if your dataset lives elsewhere.
2. **Training (Python):** Run `notebooks/training_churn_models.ipynb` to regenerate metrics and identify the current champion model.
3. **Training (R CLI):**
   ```cmd
   Rscript scripts\train_churn_models.R --prob-threshold=0.4
   ```
4. **SQL layer:** Execute the statements in `sql/SQLQuery.sql` on SQL Server (or adapt to your warehouse) to materialize reporting tables.

### 5. Serve the FastAPI Demo
```cmd
python web\api_server.py
```
Server listens on `http://127.0.0.1:8000` with `/predict` and `/health`. Open `web/customer.html`, prefill sample payloads (`web/payload_*.json`), and click **Predict churn** to view probabilities.

**Environment helpers**
- `CHURN_SAMPLE_FRAC` (default `0.1`) trims dataset size for faster iterations.
- `CHURN_FORCE_RETRAIN=1` forces retraining instead of loading `model_cache.joblib`.

## Key Insights & Next Steps
- The retained class dominates; always stratify splits or use balanced files before fitting models.
- Gender and age trends matter but product category + return behavior have the strongest lift for targeting interventions.
- Model performance currently plateaus around 0.80 accuracy—focus on fresh features (e.g., recency/frequency scores) and threshold tuning to improve precision/recall trade-offs.
- See `docs/project_brief.md` for stakeholder expectations, DB architecture, and recommended screenshots to include in presentations.

## Contributing / Extending
- Add new models or hyperparameter sweeps by extending `models` dict inside `training_churn_models.ipynb` or wiring additional learners in `scripts/train_churn_models.R`.
- Expose more metadata through the API (e.g., top SHAP features) so CX teams can understand predictions.
- Update `sql/SQLQuery.sql` with incremental load logic if deploying beyond prototypes.

## Team
- Kholoud Khaled Al-Arabi
- Shaimaa Fouad Ismail Sabri
- Farah Hossam Eldin
- Shahd Ahmed Mohamed Samir
- Abdelrahman Sami Ibrahim
- Ahmed Mostafa Erfan Mahfouz

## License
MIT License (see `LICENSE`).
