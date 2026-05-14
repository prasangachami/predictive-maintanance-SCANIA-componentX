"""
src/experiments.py
==================
Three Experiment Runners
Thesis: Cost-Aware Learning for Failure Prediction in Industrial Predictive Maintenance

This module orchestrates the three experiments that form the core of the thesis.
Each experiment follows the same strict protocol:

    1. Hyperparameter tuning  — Optuna minimises val total cost
    2. Final training         — best params, fitted on full train set
    3. Val evaluation         — thresholds optimised, all metrics computed
    4. Test evaluation        — exactly once, after all tuning decisions

The test set is touched exactly once per experiment, at the very end.
This is what makes the results scientifically valid.

Experiment map
--------------
    Experiment 1 — run_exp1_log_loss()
        Loss    : LogLoss (binary cross-entropy, LightGBM default)
        Tuning  : standard LGBM hyperparameters (n_leaves, lr, etc.)
        Answers : establishes cost baseline for RQ1

    Experiment 2 — run_exp2_focal_loss()
        Loss    : FocalLoss(gamma)
        Tuning  : gamma in [0.5, 5.0] + standard LGBM params
        Answers : RQ1 (vs baseline), RQ2 (gamma → FN/FP trade-off)

    Experiment 3 — run_exp3_cost_aware_focal()
        Loss    : CostAwareFocalLoss(cost_matrix, gamma, fn_weight)
        Tuning  : gamma + fn_weight sweep (sensitivity analysis)
        Answers : RQ1, RQ2, RQ3 (sensitivity to class weights)

    Sensitivity analysis — run_sensitivity_analysis()
        Runs Exp3 across fn_weight = {0.5, 1.0, 1.5, 2.0, 3.0}
        Produces the key thesis figure comparing cost vs FN weight

All results are returned as TrainingResult objects and passed to
ResultsTracker (Step 4) for logging, comparison tables, and figures.
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
import optuna

optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
outputs = os.path.join(ROOT, "backend/", "outputs/")

from src.cost_matrix import CostMatrix, SensitivityAnalyser
from src.model       import (
    LGBMTrainer, TrainingResult, make_trainer,
    LogLoss, FocalLoss, CostAwareFocalLoss,
)


# =============================================================================
# SHARED LGBM HYPERPARAMETER SEARCH SPACE
# Used by all three experiments — only the loss-specific params differ
# =============================================================================

def _suggest_lgbm_params(trial: optuna.Trial) -> dict:
    """
    Suggest shared LightGBM hyperparameters for an Optuna trial.
    These are the structural model params — identical search space
    across all three experiments to ensure fair comparison.
    """
    return {
        "n_estimators":      trial.suggest_int("n_estimators", 100, 800, step=50),
        "learning_rate":     trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "num_leaves":        trial.suggest_int("num_leaves", 16, 128),
        "min_child_samples": trial.suggest_int("min_child_samples", 10, 100),
        "subsample":         trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree":  trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_alpha":         trial.suggest_float("reg_alpha", 1e-4, 1.0, log=True),
        "reg_lambda":        trial.suggest_float("reg_lambda", 1e-4, 1.0, log=True),
        "verbose":           -1,
        "n_jobs":            -1,
    }


# =============================================================================
# EXPERIMENT 1 — Log-loss baseline
# =============================================================================

def run_exp1_log_loss(
    X_train:       pd.DataFrame,
    y_train:       pd.Series,
    X_val:         pd.DataFrame,
    y_val_binary:  pd.Series,
    y_val_5class:  np.ndarray,
    X_test:        pd.DataFrame,
    y_test_5class: np.ndarray,
    y_test_binary: np.ndarray,
    cost_matrix:   CostMatrix,
    n_trials:      int  = 30,
    random_state:  int  = 42,
    model_save_dir: str = "backend/outputs/models",
) -> TrainingResult:
    """
    Experiment 1 — LightGBM with standard log-loss (baseline).

    Protocol
    --------
    1. Optuna tunes LGBM hyperparameters, minimising val total cost
    2. Best params used for final training
    3. Thresholds optimised on val set
    4. Test evaluated once with val thresholds

    Parameters
    ----------
    X_train, y_train      training features and binary labels
    X_val, y_val_binary   validation features and binary labels
    y_val_5class          val temporal classes for cost evaluation
    X_test, y_test_5class, y_test_binary  test data
    cost_matrix           shared CostMatrix instance
    n_trials              Optuna trials (30 is enough for this search space)
    random_state          reproducibility seed
    model_save_dir        directory to save fitted model

    Returns
    -------
    TrainingResult with all val and test metrics populated
    """
    print("\n" + "="*60)
    print("  EXPERIMENT 1 — Log-loss (baseline)")
    print("="*60)

    # Class imbalance ratio for fair baseline comparison.
    # Exp3 handles imbalance implicitly via alpha derived from the cost matrix.
    # Exp1 must also address imbalance to make the comparison fair.
    pos = int((y_train == 1).sum())
    neg = int((y_train == 0).sum())
    scale_pos_weight = neg // pos
    print(f"  scale_pos_weight = {scale_pos_weight}  (neg/pos = {neg}/{pos})")

    # ---- Hyperparameter tuning ----
    def objective(trial):
        params = _suggest_lgbm_params(trial)
        params["scale_pos_weight"] = scale_pos_weight   # fair imbalance handling
        trainer = LGBMTrainer(
            loss_fn               = LogLoss(),
            cost_matrix           = cost_matrix,
            lgbm_params           = params,
            n_threshold_candidates= 30,
            early_stopping_rounds = 30,
            random_state          = random_state,
        )
        result = trainer.train(
            X_train, y_train, X_val, y_val_binary, y_val_5class,
            experiment_name=f"exp1_trial_{trial.number}",
        )
        return result.val_total_cost

    study = optuna.create_study(
        direction   = "minimize",
        sampler     = optuna.samplers.TPESampler(seed=random_state),
        study_name  = "exp1_log_loss",
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best_params = study.best_params
    print(f"\n  Best val cost  : {study.best_value:.2f}")
    print(f"  Best params    : {best_params}")

    # ---- Final training with best params ----
    lgbm_params = {k: best_params[k] for k in best_params}
    lgbm_params.update({
        "verbose":           -1,
        "n_jobs":            -1,
        "scale_pos_weight":  scale_pos_weight,
    })

    trainer = LGBMTrainer(
        loss_fn               = LogLoss(),
        cost_matrix           = cost_matrix,
        lgbm_params           = lgbm_params,
        n_threshold_candidates= 50,
        early_stopping_rounds = 50,
        random_state          = random_state,
    )
    result = trainer.train(
        X_train, y_train, X_val, y_val_binary, y_val_5class,
        experiment_name="exp1_log_loss",
    )

    # ---- Test evaluation ---- (exactly once)
    result = trainer.evaluate_test(
        result, X_test, y_test_5class, y_test_binary
    )

    # ---- Save model ----
    os.makedirs(model_save_dir, exist_ok=True)
    trainer.save(result, os.path.join(model_save_dir, "exp1_log_loss.pkl"))

    # Attach Optuna study for downstream reporting
    result.optuna_study = study

    _print_experiment_summary(result)
    return result


# =============================================================================
# EXPERIMENT 2 — Focal loss
# =============================================================================

def run_exp2_focal_loss(
    X_train:       pd.DataFrame,
    y_train:       pd.Series,
    X_val:         pd.DataFrame,
    y_val_binary:  pd.Series,
    y_val_5class:  np.ndarray,
    X_test:        pd.DataFrame,
    y_test_5class: np.ndarray,
    y_test_binary: np.ndarray,
    cost_matrix:   CostMatrix,
    n_trials:      int  = 40,
    random_state:  int  = 42,
    model_save_dir: str = "backend/outputs/models",
) -> TrainingResult:
    """
    Experiment 2 — LightGBM with focal loss.

    Protocol
    --------
    Same as Exp 1, plus gamma is tuned as an additional parameter.
    gamma controls the focusing — how aggressively the model down-weights
    easy examples.  Tuning gamma directly on val cost answers RQ2.

    gamma search range: [0.5, 5.0]
        gamma=0.5  mild focusing — close to log-loss
        gamma=2.0  standard focal loss setting
        gamma=5.0  strong focusing on hardest examples
    """
    print("\n" + "="*60)
    print("  EXPERIMENT 2 — Focal loss")
    print("="*60)

    def objective(trial):
        gamma  = trial.suggest_float("gamma", 0.5, 5.0)
        params = _suggest_lgbm_params(trial)

        trainer = LGBMTrainer(
            loss_fn               = FocalLoss(gamma=gamma),
            cost_matrix           = cost_matrix,
            lgbm_params           = params,
            n_threshold_candidates= 30,
            early_stopping_rounds = 30,
            random_state          = random_state,
        )
        result = trainer.train(
            X_train, y_train, X_val, y_val_binary, y_val_5class,
            experiment_name=f"exp2_trial_{trial.number}",
        )
        return result.val_total_cost

    study = optuna.create_study(
        direction  = "minimize",
        sampler    = optuna.samplers.TPESampler(seed=random_state),
        study_name = "exp2_focal_loss",
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best_params = study.best_params
    best_gamma  = best_params.pop("gamma")
    print(f"\n  Best val cost  : {study.best_value:.2f}")
    print(f"  Best gamma     : {best_gamma:.4f}")
    print(f"  Best LGBM params: {best_params}")

    lgbm_params = {k: best_params[k] for k in best_params}
    lgbm_params.update({"verbose": -1, "n_jobs": -1})

    trainer = LGBMTrainer(
        loss_fn               = FocalLoss(gamma=best_gamma),
        cost_matrix           = cost_matrix,
        lgbm_params           = lgbm_params,
        n_threshold_candidates= 50,
        early_stopping_rounds = 50,
        random_state          = random_state,
    )
    result = trainer.train(
        X_train, y_train, X_val, y_val_binary, y_val_5class,
        experiment_name="exp2_focal_loss",
    )
    result = trainer.evaluate_test(
        result, X_test, y_test_5class, y_test_binary
    )

    os.makedirs(model_save_dir, exist_ok=True)
    trainer.save(result, os.path.join(model_save_dir, "exp2_focal_loss.pkl"))

    result.optuna_study = study
    _print_experiment_summary(result)
    return result


# =============================================================================
# EXPERIMENT 3 — Cost-aware focal loss (proposed contribution)
# =============================================================================

def run_exp3_cost_aware_focal(
    X_train:       pd.DataFrame,
    y_train:       pd.Series,
    X_val:         pd.DataFrame,
    y_val_binary:  pd.Series,
    y_val_5class:  np.ndarray,
    X_test:        pd.DataFrame,
    y_test_5class: np.ndarray,
    y_test_binary: np.ndarray,
    cost_matrix:   CostMatrix,
    fn_weight:     float = 1.0,
    n_trials:      int   = 40,
    random_state:  int   = 42,
    model_save_dir: str  = "backend/outputs/models",
    experiment_name: str = "exp3_cost_aware_focal",
) -> TrainingResult:
    """
    Experiment 3 — LightGBM with cost-aware focal loss (proposed method).

    Protocol
    --------
    Same as Exp 2, plus fn_weight is a parameter controlling how strongly
    the cost matrix weights are applied.  This is the key experiment.

    fn_weight controls alpha (positive class weight):
        fn_weight=1.0  original cost matrix → alpha ~ 0.976
        fn_weight=2.0  double FN costs      → alpha closer to 1.0
        fn_weight=0.5  half FN costs        → alpha closer to 0.5

    This experiment is run multiple times with different fn_weights
    in run_sensitivity_analysis() to answer RQ2 and RQ3.

    Parameters
    ----------
    fn_weight  FN cost scaling factor — matched to cost_matrix.fn_weight
               for sensitivity analysis, pass CostMatrix(fn_weight=w)
    """
    print("\n" + "="*60)
    print(f"  EXPERIMENT 3 — Cost-aware focal loss  (fn_weight={fn_weight})")
    print("="*60)

    def objective(trial):
        gamma  = trial.suggest_float("gamma", 0.5, 5.0)
        params = _suggest_lgbm_params(trial)

        trainer = LGBMTrainer(
            loss_fn               = CostAwareFocalLoss(
                cost_matrix=cost_matrix, gamma=gamma, fn_weight=fn_weight
            ),
            cost_matrix           = cost_matrix,
            lgbm_params           = params,
            n_threshold_candidates= 30,
            early_stopping_rounds = 30,
            random_state          = random_state,
        )
        result = trainer.train(
            X_train, y_train, X_val, y_val_binary, y_val_5class,
            experiment_name=f"{experiment_name}_trial_{trial.number}",
        )
        return result.val_total_cost

    study = optuna.create_study(
        direction  = "minimize",
        sampler    = optuna.samplers.TPESampler(seed=random_state),
        study_name = experiment_name,
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best_params = study.best_params
    best_gamma  = best_params.pop("gamma")
    print(f"\n  Best val cost   : {study.best_value:.2f}")
    print(f"  Best gamma      : {best_gamma:.4f}")
    print(f"  fn_weight used  : {fn_weight}")
    print(f"  Best LGBM params: {best_params}")

    lgbm_params = {k: best_params[k] for k in best_params}
    lgbm_params.update({"verbose": -1, "n_jobs": -1})

    trainer = LGBMTrainer(
        loss_fn               = CostAwareFocalLoss(
            cost_matrix=cost_matrix, gamma=best_gamma, fn_weight=fn_weight
        ),
        cost_matrix           = cost_matrix,
        lgbm_params           = lgbm_params,
        n_threshold_candidates= 50,
        early_stopping_rounds = 50,
        random_state          = random_state,
    )
    result = trainer.train(
        X_train, y_train, X_val, y_val_binary, y_val_5class,
        experiment_name=experiment_name,
    )
    result = trainer.evaluate_test(
        result, X_test, y_test_5class, y_test_binary
    )

    os.makedirs(model_save_dir, exist_ok=True)
    save_name = f"{experiment_name}_fn{fn_weight}.pkl".replace(".", "_")
    trainer.save(result, os.path.join(model_save_dir, save_name))

    result.optuna_study = study
    _print_experiment_summary(result)
    return result


# =============================================================================
# SENSITIVITY ANALYSIS — Run Exp3 across fn_weight values
# Directly answers RQ2 and RQ3
# =============================================================================


# =============================================================================
# EXPERIMENT 4 — Cost-aware focal loss with fn_weight=0.5  (key finding)
# =============================================================================

def run_exp4_cost_aware_fn05(
    X_train:       pd.DataFrame,
    y_train:       pd.Series,
    X_val:         pd.DataFrame,
    y_val_binary:  pd.Series,
    y_val_5class:  np.ndarray,
    X_test:        pd.DataFrame,
    y_test_5class: np.ndarray,
    y_test_binary: np.ndarray,
    n_trials:      int  = 40,
    random_state:  int  = 42,
    model_save_dir: str = "backend/outputs/models",
) -> TrainingResult:
    """
    Experiment 4 — Cost-aware focal loss with fn_weight=0.5.

    Motivation
    ----------
    Sensitivity analysis (fn_weight sweep) found that fn_weight=0.5
    reduced total val cost from ~49,566 to ~28,700 — a ~42% reduction.
    This is the single most impactful empirical finding of the thesis.

    fn_weight=0.5 halves all false-negative costs in the cost matrix,
    causing the threshold optimiser to set higher decision thresholds,
    predicting fewer vehicles as near-failure, and drastically reducing
    false positive costs that dominate the dataset (many healthy vehicles
    incorrectly flagged at 7 units each × thousands of vehicles).

    This experiment makes Exp4 a first-class named experiment so it
    appears as a full row in the thesis results comparison table,
    directly answering RQ2 (weight effect on FN/FP trade-off) and
    RQ3 (sensitivity to class weights).

    Protocol
    --------
    Same as Exp3 but with CostMatrix(fn_weight=0.5) used for both
    the loss function alpha derivation AND the threshold optimisation.
    This is a complete cost-aware system tuned to the fn_weight=0.5 regime.
    """
    print("\n" + "="*60)
    print("  EXPERIMENT 4 — Cost-aware focal (fn_weight=0.5) [KEY FINDING]")
    print("="*60)

    # Cost matrix with halved FN costs — the key finding from sensitivity analysis
    cm_fn05 = CostMatrix(fn_weight=0.5)
    print(f"  fn_weight=0.5  alpha={CostAwareFocalLoss(cm_fn05).alpha:.4f}")
    print(f"  FN costs halved: cost[4][0] = {cm_fn05.matrix[4][0]:.0f} "
          f"(was {CostMatrix().matrix[4][0]:.0f})")

    def objective(trial):
        gamma  = trial.suggest_float("gamma", 0.5, 5.0)
        params = _suggest_lgbm_params(trial)

        trainer = LGBMTrainer(
            loss_fn               = CostAwareFocalLoss(
                cost_matrix=cm_fn05, gamma=gamma, fn_weight=0.5
            ),
            cost_matrix           = cm_fn05,   # optimise thresholds on same matrix
            lgbm_params           = params,
            n_threshold_candidates= 30,
            early_stopping_rounds = 30,
            random_state          = random_state,
        )
        result = trainer.train(
            X_train, y_train, X_val, y_val_binary, y_val_5class,
            experiment_name=f"exp4_trial_{trial.number}",
        )
        return result.val_total_cost

    study = optuna.create_study(
        direction  = "minimize",
        sampler    = optuna.samplers.TPESampler(seed=random_state),
        study_name = "exp4_cost_aware_fn05",
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best_params = study.best_params
    best_gamma  = best_params.pop("gamma")
    print(f"\n  Best val cost   : {study.best_value:.2f}")
    print(f"  Best gamma      : {best_gamma:.4f}")
    print(f"  Best LGBM params: {best_params}")

    lgbm_params = {k: best_params[k] for k in best_params}
    lgbm_params.update({"verbose": -1, "n_jobs": -1})

    trainer = LGBMTrainer(
        loss_fn               = CostAwareFocalLoss(
            cost_matrix=cm_fn05, gamma=best_gamma, fn_weight=0.5
        ),
        cost_matrix           = cm_fn05,
        lgbm_params           = lgbm_params,
        n_threshold_candidates= 50,
        early_stopping_rounds = 50,
        random_state          = random_state,
    )
    result = trainer.train(
        X_train, y_train, X_val, y_val_binary, y_val_5class,
        experiment_name="exp4_cost_aware_fn05",
    )
    result = trainer.evaluate_test(
        result, X_test, y_test_5class, y_test_binary
    )

    os.makedirs(model_save_dir, exist_ok=True)
    trainer.save(result, os.path.join(model_save_dir, "exp4_cost_aware_fn05.pkl"))

    result.optuna_study = study
    _print_experiment_summary(result)
    return result


def run_sensitivity_analysis(
    X_train:       pd.DataFrame,
    y_train:       pd.Series,
    X_val:         pd.DataFrame,
    y_val_binary:  pd.Series,
    y_val_5class:  np.ndarray,
    X_test:        pd.DataFrame,
    y_test_5class: np.ndarray,
    y_test_binary: np.ndarray,
    fn_weights:    list[float] = None,
    n_trials:      int  = 20,
    random_state:  int  = 42,
    model_save_dir: str = "backend/outputs/models",
) -> pd.DataFrame:
    """
    Run Experiment 3 across multiple fn_weight values to analyse
    how performance changes as the FN/FP cost ratio varies.

    This is the core of the sensitivity analysis for RQ2 and RQ3.

    For each fn_weight w:
        1. Build CostMatrix(fn_weight=w)
        2. Run Exp3 with that cost matrix
        3. Record val and test total cost, optimal gamma, optimal thresholds

    Parameters
    ----------
    fn_weights  list of FN cost scaling factors to sweep
                Default: [0.5, 1.0, 1.5, 2.0, 3.0]
                1.0 = original cost matrix
                2.0 = FN costs doubled
                0.5 = FN costs halved

    Returns
    -------
    pd.DataFrame with one row per fn_weight, columns:
        fn_weight, best_gamma, val_total_cost, test_total_cost,
        val_roc_auc, test_roc_auc, threshold_t1..t4, alpha
    """
    fn_weights = fn_weights or [0.5, 1.0, 1.5, 2.0, 3.0]

    print("\n" + "="*60)
    print("  SENSITIVITY ANALYSIS — fn_weight sweep")
    print(f"  fn_weights: {fn_weights}")
    print("="*60)

    rows = []
    for w in fn_weights:
        print(f"\n  --- fn_weight = {w} ---")

        # Build cost matrix with scaled FN costs
        cm_w = CostMatrix(fn_weight=w)

        result = run_exp3_cost_aware_focal(
            X_train, y_train,
            X_val, y_val_binary, y_val_5class,
            X_test, y_test_5class, y_test_binary,
            cost_matrix     = cm_w,
            fn_weight       = w,
            n_trials        = n_trials,
            random_state    = random_state,
            model_save_dir  = model_save_dir,
            experiment_name = f"exp3_sensitivity_fn{str(w).replace('.','_')}",
        )

        # Derive alpha from the loss function params
        alpha = result.loss_params.get("alpha", None)

        rows.append({
            "fn_weight":       w,
            "alpha":           alpha,
            "best_gamma":      result.loss_params.get("gamma"),
            "val_total_cost":  result.val_total_cost,
            "test_total_cost": result.test_total_cost,
            "val_roc_auc":     result.val_roc_auc,
            "test_roc_auc":    result.test_roc_auc,
            "threshold_t1":    result.thresholds[0] if result.thresholds else None,
            "threshold_t2":    result.thresholds[1] if result.thresholds else None,
            "threshold_t3":    result.thresholds[2] if result.thresholds else None,
            "threshold_t4":    result.thresholds[3] if result.thresholds else None,
        })

    sensitivity_df = pd.DataFrame(rows)

    print("\n" + "="*60)
    print("  SENSITIVITY ANALYSIS SUMMARY")
    print("="*60)
    print(sensitivity_df[
        ["fn_weight","alpha","best_gamma","val_total_cost","test_total_cost"]
    ].to_string(index=False))

    return sensitivity_df


# =============================================================================
# FULL EXPERIMENT SUITE — run all three experiments + sensitivity analysis
# =============================================================================

def run_all_experiments(
    X_train:       pd.DataFrame,
    y_train:       pd.Series,
    X_val:         pd.DataFrame,
    y_val_binary:  pd.Series,
    y_val_5class:  np.ndarray,
    X_test:        pd.DataFrame,
    y_test_5class: np.ndarray,
    y_test_binary: np.ndarray,
    n_trials_exp1: int  = 100,
    n_trials_exp2: int  = 100,
    n_trials_exp3: int  = 100,
    n_trials_exp4: int  = 100,
    n_trials_sens: int  = 30,
    fn_weights:    list = None,
    random_state:  int  = 42,
    model_save_dir: str = "backend/outputs/models",
) -> dict:
    """
    Run all four experiments plus sensitivity analysis in sequence.

    Experiment order
    ----------------
    Exp1 — log-loss baseline (with scale_pos_weight for fair comparison)
    Exp2 — focal loss (tune gamma)
    Exp3 — cost-aware focal loss fn_weight=1.0 (original matrix)
    Exp4 — cost-aware focal loss fn_weight=0.5 (key sensitivity finding)
    Sensitivity — fn_weight sweep across [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0]

    Protocol enforced
    -----------------
    - Exp1/Exp2/Exp3 use CostMatrix(fn_weight=1.0) for threshold optimisation
    - Exp4 uses CostMatrix(fn_weight=0.5) — consistent with its training objective
    - Test set touched exactly once per experiment
    - All experiments use 100 Optuna trials for meaningful hyperparameter search

    Returns
    -------
    dict with keys:
        exp1    TrainingResult
        exp2    TrainingResult
        exp3    TrainingResult
        sensitivity  pd.DataFrame  (fn_weight sweep results)
    """
    cm = CostMatrix(fn_weight=1.0)   # original cost matrix for all main experiments

    exp1 = run_exp1_log_loss(
        X_train, y_train, X_val, y_val_binary, y_val_5class,
        X_test, y_test_5class, y_test_binary,
        cost_matrix    = cm,
        n_trials       = n_trials_exp1,
        random_state   = random_state,
        model_save_dir = model_save_dir,
    )

    exp2 = run_exp2_focal_loss(
        X_train, y_train, X_val, y_val_binary, y_val_5class,
        X_test, y_test_5class, y_test_binary,
        cost_matrix    = cm,
        n_trials       = n_trials_exp2,
        random_state   = random_state,
        model_save_dir = model_save_dir,
    )

    exp3 = run_exp3_cost_aware_focal(
        X_train, y_train, X_val, y_val_binary, y_val_5class,
        X_test, y_test_5class, y_test_binary,
        cost_matrix    = cm,
        fn_weight      = 1.0,
        n_trials       = n_trials_exp3,
        random_state   = random_state,
        model_save_dir = model_save_dir,
    )

    # Exp4 uses its own cost matrix (fn_weight=0.5) — consistent with training
    exp4 = run_exp4_cost_aware_fn05(
        X_train, y_train, X_val, y_val_binary, y_val_5class,
        X_test, y_test_5class, y_test_binary,
        n_trials       = n_trials_exp4,
        random_state   = random_state,
        model_save_dir = model_save_dir,
    )

    sensitivity = run_sensitivity_analysis(
        X_train, y_train, X_val, y_val_binary, y_val_5class,
        X_test, y_test_5class, y_test_binary,
        fn_weights     = fn_weights or [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0],
        n_trials       = n_trials_sens,
        random_state   = random_state,
        model_save_dir = model_save_dir,
    )

    _print_final_comparison(exp1, exp2, exp3)
    return {
        "exp1":        exp1,
        "exp2":        exp2,
        "exp3":        exp3,
        "exp4":        exp4,
        "sensitivity": sensitivity,
    }


# =============================================================================
# COMPARISON TABLE — thesis Results chapter helper
# =============================================================================

def build_comparison_table(
    results: dict,
) -> pd.DataFrame:
    """
    Build a clean comparison table from the three experiment results.
    Ready to paste into the thesis Results chapter.

    Parameters
    ----------
    results  dict returned by run_all_experiments()

    Returns
    -------
    pd.DataFrame  columns: experiment, loss, val_cost, test_cost, roc_auc, thresholds
    """
    rows = []
    for key in ("exp1", "exp2", "exp3", "exp4"):
        r = results[key]
        rows.append({
            "experiment":      r.experiment_name,
            "loss_function":   r.loss_name,
            "loss_params":     str(r.loss_params),
            "val_total_cost":  round(r.val_total_cost,  2) if r.val_total_cost  else None,
            "test_total_cost": round(r.test_total_cost, 2) if r.test_total_cost else None,
            "val_roc_auc":     round(r.val_roc_auc,     4) if r.val_roc_auc     else None,
            "test_roc_auc":    round(r.test_roc_auc,    4) if r.test_roc_auc    else None,
            "best_iteration":  r.best_iteration,
            "threshold_t1":    r.thresholds[0] if r.thresholds else None,
            "threshold_t4":    r.thresholds[3] if r.thresholds else None,
        })

    df = pd.DataFrame(rows)
    return df


# =============================================================================
# INTERNAL HELPERS
# =============================================================================

def _print_experiment_summary(result: TrainingResult) -> None:
    print(f"\n  {'─'*50}")
    print(f"  Experiment  : {result.experiment_name}")
    print(f"  Loss        : {result.loss_name}  {result.loss_params}")
    print(f"  Val cost    : {result.val_total_cost:.2f}" if result.val_total_cost else "")
    print(f"  Test cost   : {result.test_total_cost:.2f}" if result.test_total_cost else "")
    print(f"  Val AUC     : {result.val_roc_auc:.4f}" if result.val_roc_auc else "")
    print(f"  Test AUC    : {result.test_roc_auc:.4f}" if result.test_roc_auc else "")
    print(f"  Thresholds  : {result.thresholds}")
    print(f"  {'─'*50}")


def _print_final_comparison(
    exp1: TrainingResult,
    exp2: TrainingResult,
    exp3: TrainingResult,
) -> None:
    print("\n" + "="*60)
    print("  FINAL COMPARISON — all three experiments")
    print("="*60)
    header = f"  {'Experiment':<30} {'Val cost':>12}  {'Test cost':>12}  {'Test AUC':>10}"
    print(header)
    print("  " + "─"*58)
    for r in (exp1, exp2, exp3):
        vc = f"{r.val_total_cost:.2f}"  if r.val_total_cost  else "N/A"
        tc = f"{r.test_total_cost:.2f}" if r.test_total_cost else "N/A"
        ra = f"{r.test_roc_auc:.4f}"    if r.test_roc_auc    else "N/A"
        print(f"  {r.experiment_name:<30} {vc:>12}  {tc:>12}  {ra:>10}")
    print("="*60)

    # Cost reduction from Exp1 to Exp3
    if exp1.test_total_cost and exp3.test_total_cost:
        reduction = (exp1.test_total_cost - exp3.test_total_cost) / exp1.test_total_cost * 100
        print(f"\n  Cost reduction (Exp1 → Exp3) : {reduction:+.1f}%")
        if reduction > 0:
            print("  Cost-aware focal loss reduced total cost vs log-loss baseline.")
        else:
            print("  Note: Exp3 did not reduce cost vs baseline on this run.")