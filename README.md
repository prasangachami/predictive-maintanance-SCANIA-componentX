# Cost-Aware Learning for Failure Prediction
## A Case Study on the SCANIA Component X Dataset

> Master's Thesis — Predictive Maintenance with Asymmetric Cost-Aware Loss Functions

[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/Tests-pytest-orange.svg)](backend/tests/)

---

## Overview

This repository contains the full implementation of a cost-aware predictive maintenance system trained on the SCANIA Component X dataset. The study compares four LightGBM models trained with different loss functions, evaluated under a real 5×5 industrial cost matrix where missing a failure costs up to 40× more than a false alarm.

### Industrial Cost Matrix (5×5)

|  | Pred 0 | Pred 1 | Pred 2 | Pred 3 | Pred 4 |
|---|---|---|---|---|---|
| **Actual 0** | 0 | 7 | 8 | 9 | 10 |
| **Actual 1** | 200 | 0 | 7 | 8 | 9 |
| **Actual 2** | 300 | 200 | 0 | 7 | 8 |
| **Actual 3** | 400 | 300 | 200 | 0 | 7 |
| **Actual 4** | 500 | 400 | 300 | 200 | 0 |

### Cost-Aware Focal Loss Formula

The class weight α is derived analytically from the cost matrix:

```
α = μ_FN / (μ_FN + μ_FP) = 350 / 358.5 ≈ 0.976
```

---

## Repository Structure

```
├── run_preprocessing.py      ← Step 1: run full preprocessing pipeline
├── run_experiments.py        ← Step 2: run one or all experiments
├── data/
│   └── raw/                  ← place dataset files here (not committed)
├── backend/                  ← model training, API, experiments
│   └── README.md             ← backend setup and usage
└── frontend/                 ← interactive results dashboard (React)
    └── README.md             ← dashboard setup and live URL
```

---

## Quick Start

### 1. Clone and install

```bash
git clone https://github.com/prasangachami/predictive-maintanance-SCANIA-componentX.git
cd predictive-maintanance-SCANIA-componentX
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r backend/requirements.txt
```

### 2. Add the dataset

The SCANIA Component X dataset requires a data-sharing agreement and cannot be distributed in this repository. Once you have access, place the files here:

```
data/
└── raw/
    ├── train_operational_readouts.csv
    ├── train_labels.csv
    ├── validation_operational_readouts.csv
    ├── validation_labels.csv
    ├── test_operational_readouts.csv
    └── test_labels.csv
```

Dataset access: [IDA 2024 Industrial Challenge](https://ida2024.dsv.su.se/)  
Dataset paper: Kharazian et al. (2025), *Scientific Data* — [arXiv:2401.15199](https://arxiv.org/abs/2401.15199)

### 3. Run preprocessing

```bash
python run_experiments.py --pipeline-only
```

This reads from `data/raw/` and writes processed feature vectors to `data/processed/`.

### 4. Run experiments

```bash
# Run all four experiments
python run_experiments.py --exp all

# Run a single experiment
python run_experiments.py --exp exp1
python run_experiments.py --exp exp2
python run_experiments.py --exp exp3
python run_experiments.py --exp exp4

# Run sensitivity analysis
python run_experiments.py --exp sensitivity
```

Results are saved to `backend/outputs/results/` and model files to `backend/outputs/models/`.

---

## Resources

| Resource | Link |
|---|---|
| Interactive Dashboard | [https://scania-dashboard-185324232016.europe-north1.run.app/]
| Thesis Paper | [University repository] |
| Dataset Paper | [arXiv:2401.15199](https://arxiv.org/abs/2401.15199) |
| Backend API | See [backend/README.md](backend/README.md) |
| Frontend Dashboard | See [frontend/README.md](frontend/README.md) |

---

## Key Results

| Experiment | Loss Function | Test Cost | AUC-ROC | Recall |
|---|---|---|---|---|
| Exp 1 | Cross-entropy (baseline) | 46,393 | 0.634 | 0.796 |
| Exp 2 | Focal loss | 46,345 | 0.633 | 0.810 |
| Exp 3 | Cost-aware focal (w=1.0) | 49,099 | 0.665 | 1.000 |
| Exp 4 | Cost-aware focal (w=0.5) | 28,050 | 0.500 | 0.000 |

**Key finding:** Cost-aware training improved recall to 1.000 but increased total cost under the original cost matrix. A minimum recall constraint on the threshold optimiser is identified as the critical missing component for genuine cost reduction.

This gives failure instances 40× more gradient weight than healthy instances during training.

### Sensitivity Analysis

| fn_weight | α | Test Cost | Test AUC |
|---|---|---|---|
| 0.25 | 0.911 | 14,025 | 0.373 |
| 0.50 | 0.954 | 28,050 | 0.500 |
| 0.75 | 0.969 | 42,075 | 0.500 |
| **1.00** | **0.976** | **49,032** | **0.710** |
| 1.50 | 0.984 | 49,695 | 0.671 |
| 2.00 | 0.988 | 49,449 | 0.694 |
| 3.00 | 0.992 | 50,092 | 0.679 |

fn_weight = 1.0 (derived from the cost matrix) produces the highest AUC of 0.710 in the entire study.

---

## Running Tests

```bash
cd backend
python -m pytest tests/ -v
```

Six test modules cover the pipeline, cost matrix, experiments, model, API, and results tracker.

---

## Docker

Both services can be run with Docker:

```bash
# Backend API
docker build -t scania-backend ./backend
docker run -p 8000:8000 scania-backend

# Frontend dashboard
docker build -t scania-frontend ./frontend
docker run -p 80:80 scania-frontend
```

---

## Dataset Availability

The SCANIA Component X dataset is subject to a data-sharing agreement and is not included in this repository. Researchers wishing to reproduce the experiments should obtain access through the IDA 2024 Industrial Challenge organisers.

---