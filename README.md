# SCANIA ComponentX — Predictive Maintenance Thesis
## Cost-Aware Learning for Failure Prediction in Industrial Predictive Maintenance

---

## Project structure

```
scania_predictive_maintenance/
│
├── data/
│   ├── raw/                         ← original CSV files (never modified, gitignored)
│   │   ├── train_operational_readouts.csv
│   │   ├── train_tte.csv
│   │   ├── train_specifications.csv
│   │   ├── validation_operational_readouts.csv
│   │   ├── validation_labels.csv
│   │   ├── validation_specifications.csv
│   │   ├── test_operational_readouts.csv
│   │   ├── test_labels.csv
│   │   └── test_specifications.csv
│   └── processed/                   ← flat tables saved as .parquet (gitignored)
│
├── src/                             ← all importable Python source code
│   ├── __init__.py
│   ├── cost_matrix.py               ← Step 1: industrial cost matrix + threshold optimiser
│   ├── model.py                     ← Step 2: LightGBM trainer + 3 loss functions
│   ├── experiments.py               ← Step 3: 3 experiment runners + sensitivity analysis
│   ├── results_tracker.py           ← Step 4: logging, figures, report
│   └── utils.py
│
├── tests/                           ← pytest test suite — mirrors src/
│   ├── test_scania_pipeline.py      ← 57 tests (preprocessing)
│   ├── test_cost_matrix.py          ← 60 tests
│   ├── test_model.py                ← 65 tests
│   ├── test_experiments.py          ← 45 tests
│   └── test_results_tracker.py      ← 65 tests
│
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_experiments.ipynb
│   └── 04_results_figures.ipynb
│
├── outputs/                         ← generated artefacts (gitignored)
│   ├── models/                      ← saved .pkl pipeline + 3 trained models
│   ├── figures/                     ← all .png plots for thesis
│   └── results/                     ← metrics CSVs + experiment_report.txt
│
├── scania_pipeline.py               ← preprocessing pipeline (existing)
├── scania_labels.py                 ← label handling (existing)
├── run_experiments.py               ← Step 5: single entry point
├── pytest.ini
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Research questions

- **RQ1**: Does cost-aware training reduce total maintenance cost vs standard loss?
- **RQ2**: How does varying failure-class weight affect the FN/FP trade-off?
- **RQ3**: How sensitive is performance to the choice of class weights?

---

## Experiment design

| | Experiment | Loss function | Tuned param | Answers |
|--|--|--|--|--|
| Exp 1 | Log-loss baseline | Binary cross-entropy | LGBM hparams | RQ1 baseline |
| Exp 2 | Focal loss | FocalLoss(gamma) | gamma + LGBM hparams | RQ1, RQ2 |
| Exp 3 | Cost-aware focal (proposed) | CostAwareFocalLoss | gamma + fn_weight | RQ1, RQ2, RQ3 |

---

## How to run

### Install dependencies
```bash
pip install -r requirements.txt
```

### Place raw data files
```
data/raw/train_operational_readouts.csv
data/raw/train_tte.csv
data/raw/train_specifications.csv
data/raw/validation_operational_readouts.csv
data/raw/validation_labels.csv
data/raw/validation_specifications.csv
data/raw/test_operational_readouts.csv
data/raw/test_labels.csv
data/raw/test_specifications.csv
```

### Run commands

```bash
# Full run — all 3 experiments + sensitivity + figures + report
python run_experiments.py --all

# Quick development run (3 Optuna trials per experiment)
python run_experiments.py --all --fast

# Pipeline preprocessing only
python run_experiments.py --pipeline-only

# Run specific experiments (skip pipeline if already processed)
python run_experiments.py --exp1 --skip-pipeline
python run_experiments.py --exp2 --skip-pipeline
python run_experiments.py --exp3 --skip-pipeline
python run_experiments.py --exp1 --exp2 --exp3 --skip-pipeline

# Sensitivity analysis only
python run_experiments.py --sensitivity --skip-pipeline

# Custom fn_weights for sensitivity analysis
python run_experiments.py --sensitivity --fn-weights 0.5 1.0 2.0 3.0 --skip-pipeline

# Regenerate figures from saved results (no re-training)
python run_experiments.py --figures-only

# Override data directory
python run_experiments.py --all --data-dir /path/to/data

# Skip figure/report generation
python run_experiments.py --all --no-figures --no-report

# Set random seed
python run_experiments.py --all --random-state 123
```

### Run tests
```bash
# All tests
pytest tests/ -v --tb=short

# Specific test file
pytest tests/test_cost_matrix.py   -v
pytest tests/test_model.py         -v
pytest tests/test_experiments.py   -v
pytest tests/test_results_tracker.py -v

# Skip slow tests (default)
pytest tests/ -v -m "not slow"

# Run slow tests (requires real data, takes ~10 min)
pytest tests/ -v -m slow
```

---

## Output files

After a full run:

```
outputs/
├── models/
│   ├── pipeline_state.pkl           ← fitted preprocessing pipeline
│   ├── exp1_log_loss.pkl            ← trained Exp1 model + thresholds
│   ├── exp2_focal_loss.pkl          ← trained Exp2 model + thresholds
│   └── exp3_cost_aware_focal_*.pkl  ← trained Exp3 models
│
├── results/
│   ├── results.csv                  ← all experiment metrics
│   ├── comparison_table.csv         ← clean thesis comparison table
│   ├── sensitivity_results.csv      ← fn_weight sweep results
│   └── experiment_report.txt        ← plain-text thesis report
│
└── figures/
    ├── cost_comparison.png          ← RQ1: bar chart Exp1 vs Exp2 vs Exp3
    ├── roc_comparison_test.png      ← ROC curves
    ├── confusion_matrices.png       ← 5x5 cost-weighted confusion matrices
    ├── prob_distributions.png       ← P(failure) histograms
    ├── threshold_comparison.png     ← RQ2: optimal thresholds per experiment
    ├── sensitivity_curve.png        ← RQ3: cost vs fn_weight
    ├── cost_breakdown_exp3.png      ← cost heatmap for proposed method
    └── optuna_history_*.png         ← hyperparameter tuning history
```

---

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
