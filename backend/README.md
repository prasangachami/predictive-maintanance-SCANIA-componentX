# Backend — Model Training, API & Experiments

This folder contains the full ML pipeline: data preprocessing, model training, experiment management, threshold optimisation, and a FastAPI REST API for serving predictions.

---

## Structure

```
backend/
├── app/                        ← FastAPI REST API
│   ├── main.py                 ← API entry point and routes
│   ├── predict.py              ← prediction logic using saved models
│   └── schemas.py              ← request and response schemas
│
├── src/                        ← all reusable source code
│   ├── data/
│   │   └── pipeline.py         ← SCANIAPipeline — full preprocessing class
│   ├── model.py                ← LightGBM wrapper with custom loss support
│   ├── experiments.py          ← experiment runner (all four experiments)
│   ├── cost_matrix.py          ← 5×5 industrial cost matrix and evaluation
│   ├── data_loader.py          ← data loading utilities
│   ├── registry.py             ← experiment registry and configuration
│   ├── results_tracker.py      ← saves and loads experiment results
│   └── utils.py                ← shared helpers
│
├── configs/                    ← per-experiment YAML configurations
│   ├── exp1.yaml
│   ├── exp2.yaml
│   ├── exp3.yaml
│   └── exp4.yaml
│
├── notebooks/                  ← analysis and result generation notebooks
│   ├── 00_exploration.ipynb
│   ├── 05_diagnostic.ipynb
│   ├── 06_merge_results.ipynb  ← generates all figures and result CSVs
│   └── Data_analysis.ipynb
│
├── outputs/                    ← generated results (committed)
│   ├── figures/                ← all thesis figures (PNG)
│   ├── models/                 ← saved model .pkl files
│   └── results/                ← CSV result tables
│
├── tests/                      ← pytest test suite
│   ├── test_scania_pipeline.py
│   ├── test_cost_matrix.py
│   ├── test_experiments.py
│   ├── test_model.py
│   ├── test_results_tracker.py
│   └── test_api.py
│
├── Dockerfile                  ← containerised backend
├── requirements.txt            ← Python dependencies
├── conftest.py                 ← pytest path configuration
└── pytest.ini                  ← pytest settings
```


## Industrial cost matrix

```
         Pred 0  Pred 1  Pred 2  Pred 3  Pred 4
Actual 0    0       7       8       9      10
Actual 1   200      0       7       8       9
Actual 2   300    200       0       7       8
Actual 3   400    300     200       0       7
Actual 4   500    400     300     200       0
```

**Class definitions:**
- Class 0 = > 48 steps before failure (healthy)
- Class 1 = 12–24 steps before failure
- Class 2 = 24–48 steps before failure
- Class 3 = 6–12 steps before failure
- Class 4 = 0–6 steps before failure (imminent)

**Two label systems:**
- Binary labels `{0,1}` → model training only
- 5-class `temporal_class` `{0,1,2,3,4}` → cost evaluation only
- These two systems never mix.

---

## Installation

From the project root:

```bash
pip install -r backend/requirements.txt
```

Or inside the `backend/` folder:

```bash
pip install -r requirements.txt
```

---

## Running Experiments

From the **project root** (recommended):

```bash
# All four experiments
python run_experiments.py --exp all

# Individual experiments
python run_experiments.py --exp exp1   # cross-entropy baseline
python run_experiments.py --exp exp2   # focal loss
python run_experiments.py --exp exp3   # cost-aware focal, fn_weight=1.0
python run_experiments.py --exp exp4   # cost-aware focal, fn_weight=0.5

# Sensitivity analysis (fn_weight sweep: 0.25 → 3.0)
python run_experiments.py --exp sensitivity
```

Results are saved to `backend/outputs/results/` and model files to `backend/outputs/models/`.

---

## Preprocessing Pipeline

The `SCANIAPipeline` class in `src/data/pipeline.py` handles:

- Loading raw sensor readouts from `data/raw/`
- Histogram feature aggregation (sum, mean, max per variable)
- Counter feature aggregation (last value, delta, rate of change)
- Stress ratio features (top-quartile ratio, entropy, concentration)
- Counter trend features (linear slope, recent slope, slope acceleration)
- Feature filtering (constant removal, correlation filtering)
- Writing processed feature vectors to `data/processed/`

```python
from backend.src.data.pipeline import SCANIAPipeline

pipeline = SCANIAPipeline(
    data_dir           = "data/raw/",   # defaults to project root data/raw/
    variance_threshold = 0.01,
    corr_threshold     = 0.95,
    cost_matrix        = COST_MATRIX,
    plot               = True,
)
pipeline.run()
```

---

## Experiment Design

All four experiments share the same setup:

| Setting | Value |
|---|---|
| Model | LightGBM |
| Hyperparameter tuning | Optuna TPE — 100 trials per experiment |
| Tuning objective | Minimise total validation cost |
| Threshold search | 50 candidates per position (~230,300 combinations) |
| Evaluation | Test set evaluated once, after all tuning is complete |

| Experiment | Loss Function | fn_weight | α | Test Cost | AUC |
|---|---|---|---|---|---|
| Exp 1 | Cross-entropy | — | — | 46,393 | 0.634 |
| Exp 2 | Focal loss | — | — | 46,345 | 0.633 |
| Exp 3 | Cost-aware focal | 1.0 | 0.976 | 49,099 | 0.665 |
| Exp 4 | Cost-aware focal | 0.5 | 0.954 | 28,050 | 0.500 |

---

## FastAPI — REST API

The API serves predictions from saved models.

### Start the API

```bash
cd backend
uvicorn app.main:app --reload --port 8000
```

Or with Docker:

```bash
docker build -t scania-backend .
docker run -p 8000:8000 scania-backend
```

### Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | Health check |
| GET | `/experiments` | List all experiments and results |
| POST | `/predict` | Predict failure class for a vehicle |

### Example request

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"experiment": "exp3", "features": {...}}'
```

API documentation available at: `http://localhost:8000/docs`

---

## Running Tests

```bash
cd backend
python -m pytest tests/ -v
```

Run a specific test module:

```bash
python -m pytest tests/test_scania_pipeline.py -v
python -m pytest tests/test_cost_matrix.py -v
python -m pytest tests/test_api.py -v

---

## Generated Figures

Running `notebooks/06_merge_results.ipynb` generates all thesis figures in `outputs/figures/`:

| Figure | Description |
|---|---|
| `fig4_1_cost_auc.png` | Cost and AUC comparison across experiments |
| `fig4_2_precision_recall.png` | Precision, recall, F1 by experiment |
| `fig4_3_sensitivity.png` | Sensitivity analysis — cost and AUC vs fn_weight |
| `roc_comparison_test.png` | ROC curves for all experiments |
| `feature_importance_all.png` | LightGBM feature importance (gain) |
| `threshold_comparison.png` | Threshold configuration per experiment |
| `sensitivity_curve.png` | fn_weight sensitivity curve |
| `cost_breakdown_exp3.png` | FN vs FP cost breakdown for Exp 3 |

---

## Key Design Decisions

**Cost-aware focal loss** — the class weight α is derived analytically from the cost matrix:
```
α = μ_FN / (μ_FN + μ_FP) = 350 / 358.5 ≈ 0.976
```
This gives failure instances 40× more gradient weight than healthy instances during training, without manual tuning.

**Threshold optimisation** — four decision thresholds (t1–t4) map the model's output probability to one of five prediction classes. The optimiser searches ~230,300 combinations on the validation set only. The test set is never used during optimisation.

**Sensitivity analysis** — fn_weight is varied across seven values (0.25, 0.50, 0.75, 1.00, 1.50, 2.00, 3.00). Below fn_weight = 1.0, AUC collapses. Above fn_weight = 1.0, cost improvement is flat due to alpha saturation.
