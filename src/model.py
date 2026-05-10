"""
src/model.py
============
LightGBM Trainer — Three Loss Functions
Thesis: Cost-Aware Learning for Failure Prediction in Industrial Predictive Maintenance

Architecture — Strategy pattern
LGBMTrainer: shared training loop. Loss function is the only variable per experiment.

Loss functions:
    LogLoss               Experiment 1 — standard binary cross-entropy (baseline)
    FocalLoss             Experiment 2 — focuses training on hard examples via gamma
    CostAwareFocalLoss    Experiment 3 — focal loss + alpha from industrial cost matrix
"""

import os
import pickle
import warnings
import numpy as np
import pandas as pd
import lightgbm as lgb
import matplotlib.pyplot as plt

from abc         import ABC, abstractmethod
from dataclasses import dataclass, field
from typing      import Optional

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.cost_matrix import CostMatrix, ThresholdOptimiser, N_CLASSES

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Base loss interface
# ---------------------------------------------------------------------------

class BaseLoss(ABC):
    @abstractmethod
    def gradients(self, y_true: np.ndarray, y_pred: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        pass

    @property
    @abstractmethod
    def name(self) -> str: pass

    @property
    @abstractmethod
    def params(self) -> dict: pass

    @staticmethod
    def _sigmoid(x: np.ndarray) -> np.ndarray:
        return np.where(x >= 0, 1.0/(1.0+np.exp(-x)), np.exp(x)/(1.0+np.exp(x)))


# ---------------------------------------------------------------------------
# Loss 1: Log-loss  (Experiment 1 baseline)
# ---------------------------------------------------------------------------

class LogLoss(BaseLoss):
    """Standard binary cross-entropy. grad = p - y.  hess = p*(1-p)."""

    @property
    def name(self): return "log_loss"

    @property
    def params(self): return {"loss": "log_loss"}

    def gradients(self, y_true, y_pred):
        p    = self._sigmoid(y_pred)
        grad = p - y_true
        hess = np.maximum(p * (1.0 - p), 1e-6)
        return grad, hess


# ---------------------------------------------------------------------------
# Loss 2: Focal loss  (Experiment 2)
# ---------------------------------------------------------------------------

class FocalLoss(BaseLoss):
    """
    Focal loss (Lin et al. 2017). Focuses training on hard misclassifications.
    gamma=0 reduces to log-loss.  Typical range: 0.5 to 5.0.

    p_t  = p if y=1, else 1-p
    FL   = -(1-p_t)^gamma * log(p_t)
    grad = (1-p_t)^gamma * (p-y) * [1 + gamma*log(p_t)/(1-p_t)]
    hess = (1-p_t)^gamma * p*(1-p)
    """

    def __init__(self, gamma: float = 2.0):
        if gamma < 0:
            raise ValueError(f"gamma must be >= 0, got {gamma}")
        self.gamma = gamma

    @property
    def name(self): return "focal_loss"

    @property
    def params(self): return {"loss": "focal_loss", "gamma": self.gamma}

    def gradients(self, y_true, y_pred):
        eps    = 1e-7
        p      = self._sigmoid(y_pred)
        p_t    = np.where(y_true == 1, p, 1.0 - p)
        mod    = (1.0 - p_t) ** self.gamma
        log_pt = np.log(np.clip(p_t, eps, 1.0))
        grad   = mod * (p - y_true) * (1.0 + self.gamma * log_pt / (1.0 - p_t + eps))
        hess   = np.maximum(mod * p * (1.0 - p), 1e-6)
        return grad, hess


# ---------------------------------------------------------------------------
# Loss 3: Cost-aware focal loss  (Experiment 3 — proposed contribution)
# ---------------------------------------------------------------------------

class CostAwareFocalLoss(BaseLoss):
    """
    Cost-aware focal loss — the proposed method of this thesis.

    Extends focal loss with per-sample weight alpha derived from the
    industrial cost matrix.  The model is penalised differently for
    FN vs FP errors during training, not just at evaluation time.

    Alpha derivation (binary training labels only):
        fn_mean = mean(cost_matrix[1:, 0]) = mean([200,300,400,500]) = 350
        fp_mean = mean(cost_matrix[0, 1:]) = mean([7,8,9,10])        = 8.5
        alpha   = fn_mean / (fn_mean + fp_mean)                      ~ 0.976

    Positive samples (y=1) receive weight alpha.
    Negative samples (y=0) receive weight (1 - alpha).

    fn_weight scales FN costs before computing alpha — used in sensitivity
    analysis to answer RQ2 and RQ3.

    Parameters
    ----------
    cost_matrix  CostMatrix instance
    gamma        focal focusing parameter
    fn_weight    scales FN costs — matches cost_matrix.fn_weight for consistency
    """

    def __init__(self, cost_matrix: CostMatrix, gamma: float = 2.0, fn_weight: float = 1.0):
        if gamma < 0:
            raise ValueError(f"gamma must be >= 0, got {gamma}")
        self.cost_matrix = cost_matrix
        self.gamma       = gamma
        self.fn_weight   = fn_weight
        self.alpha       = self._derive_alpha()

    def _derive_alpha(self) -> float:
        m        = self.cost_matrix.matrix
        fn_mean  = np.mean([m[a][0] for a in range(1, N_CLASSES)])
        fp_mean  = np.mean([m[0][p] for p in range(1, N_CLASSES)])
        return float(fn_mean / (fn_mean + fp_mean + 1e-9))

    @property
    def name(self): return "cost_aware_focal_loss"

    @property
    def params(self):
        return {
            "loss":      "cost_aware_focal_loss",
            "gamma":     self.gamma,
            "alpha":     round(self.alpha, 4),
            "fn_weight": self.fn_weight,
        }

    def gradients(self, y_true, y_pred):
        eps     = 1e-7
        p       = self._sigmoid(y_pred)
        p_t     = np.where(y_true == 1, p, 1.0 - p)
        alpha_t = np.where(y_true == 1, self.alpha, 1.0 - self.alpha)
        mod     = (1.0 - p_t) ** self.gamma
        log_pt  = np.log(np.clip(p_t, eps, 1.0))
        grad    = alpha_t * mod * (p - y_true) * (1.0 + self.gamma * log_pt / (1.0 - p_t + eps))
        hess    = np.maximum(alpha_t * mod * p * (1.0 - p), 1e-6)
        return grad, hess


# ---------------------------------------------------------------------------
# Training result dataclass
# ---------------------------------------------------------------------------

@dataclass
class TrainingResult:
    """All outputs from one training run. Passed to ResultsTracker."""
    experiment_name:     str
    loss_name:           str
    loss_params:         dict
    model:               lgb.Booster        = field(repr=False)
    thresholds:          tuple              = None
    best_iteration:      int                = None
    train_logloss:       float              = None
    val_logloss:         float              = None
    val_total_cost:      float              = None
    val_cost_breakdown:  pd.DataFrame       = field(default=None, repr=False)
    test_total_cost:     float              = None
    test_cost_breakdown: pd.DataFrame       = field(default=None, repr=False)
    val_probs:           np.ndarray         = field(default=None, repr=False)
    test_probs:          np.ndarray         = field(default=None, repr=False)
    val_roc_auc:         float              = None
    test_roc_auc:        float              = None
    # Binary classification metrics computed on test set at threshold 0.5
    test_precision:      float              = None
    test_recall:         float              = None
    test_f1:             float              = None
    test_pr_auc:         float              = None

    def summary(self) -> dict:
        return {
            "experiment_name": self.experiment_name,
            "loss_name":       self.loss_name,
            **self.loss_params,
            "best_iteration":  self.best_iteration,
            "train_logloss":   self.train_logloss,
            "val_logloss":     self.val_logloss,
            "val_total_cost":  self.val_total_cost,
            "test_total_cost": self.test_total_cost,
            "val_roc_auc":     self.val_roc_auc,
            "test_roc_auc":    self.test_roc_auc,
            "test_precision":  self.test_precision,
            "test_recall":     self.test_recall,
            "test_f1":         self.test_f1,
            "test_pr_auc":     self.test_pr_auc,
            "threshold_t1":    self.thresholds[0] if self.thresholds else None,
            "threshold_t2":    self.thresholds[1] if self.thresholds else None,
            "threshold_t3":    self.thresholds[2] if self.thresholds else None,
            "threshold_t4":    self.thresholds[3] if self.thresholds else None,
        }


# ---------------------------------------------------------------------------
# LGBMTrainer
# ---------------------------------------------------------------------------

class LGBMTrainer:
    """
    Shared training loop for all three experiments.
    Only loss_fn changes between experiments.

    Parameters
    ----------
    loss_fn                 LogLoss | FocalLoss | CostAwareFocalLoss
    cost_matrix             CostMatrix — threshold optimisation + evaluation
    lgbm_params             LightGBM hyperparameters (tuned by Optuna in Exp)
    n_threshold_candidates  sweep resolution (20 → ~160k combos, fast)
    early_stopping_rounds   patience
    random_state            reproducibility seed
    """

    DEFAULT_LGBM_PARAMS = {
        "n_estimators":      500,
        "learning_rate":     0.05,
        "num_leaves":        31,
        "min_child_samples": 20,
        "subsample":         0.8,
        "colsample_bytree":  0.8,
        "reg_alpha":         0.1,
        "reg_lambda":        0.1,
        "verbose":           -1,
        "n_jobs":            -1,
    }

    def __init__(
        self,
        loss_fn:                BaseLoss,
        cost_matrix:            CostMatrix,
        lgbm_params:            dict = None,
        n_threshold_candidates: int  = 20,
        early_stopping_rounds:  int  = 50,
        random_state:           int  = 42,
    ):
        self.loss_fn                = loss_fn
        self.cost_matrix            = cost_matrix
        self.lgbm_params            = lgbm_params or self.DEFAULT_LGBM_PARAMS.copy()
        self.n_threshold_candidates = n_threshold_candidates
        self.early_stopping_rounds  = early_stopping_rounds
        self.random_state           = random_state
        self._booster:   Optional[lgb.Booster]        = None
        self._optimiser: Optional[ThresholdOptimiser] = None

    # ------------------------------------------------------------------
    def train(
        self,
        X_train:         pd.DataFrame,
        y_train:         pd.Series,
        X_val:           pd.DataFrame,
        y_val_binary:    pd.Series,
        y_val_5class:    np.ndarray,
        experiment_name: str = "experiment",
    ) -> TrainingResult:
        """
        Fit LightGBM with the configured loss function.

        y_val_binary  used for early stopping only (binary logloss monitor)
        y_val_5class  used for cost evaluation — NOT passed to LightGBM
        """
        print(f"\n{'='*60}")
        print(f"  Training : {experiment_name}  |  loss : {self.loss_fn.name}")
        print(f"  Params   : {self.loss_fn.params}")
        print(f"{'='*60}")

        # Build aligned categorical column list.
        # Both train and val must agree exactly on which columns are categorical
        # and their category levels. Filter operations can silently strip the
        # category dtype from val, causing the "do not match" ValueError.
        train_cat_cols = X_train.select_dtypes(include="category").columns.tolist()
        val_cat_cols   = X_val.select_dtypes(include="category").columns.tolist()
        cat_cols       = [c for c in train_cat_cols if c in val_cat_cols]

        # Re-align val category dtypes to exactly match train's levels
        X_val_aligned = X_val.copy()
        for col in cat_cols:
            X_val_aligned[col] = X_val_aligned[col].astype(X_train[col].dtype)

        train_data = lgb.Dataset(
            X_train, label=y_train,
            categorical_feature=cat_cols if cat_cols else "auto",
            free_raw_data=False,
        )
        val_data = lgb.Dataset(
            X_val_aligned, label=y_val_binary,
            categorical_feature=cat_cols if cat_cols else "auto",
            reference=train_data, free_raw_data=False,
        )

        # ------------------------------------------------------------------
        # LightGBM custom objective signature:
        #   fobj(y_pred: np.ndarray, train_data: lgb.Dataset) -> (grad, hess)
        #
        # y_pred     : raw log-odds scores from the current booster
        # train_data : lgb.Dataset — y_true is extracted via get_label()
        #
        # Our BaseLoss.gradients(y_true, y_pred) uses the natural math order.
        # The wrapper below bridges the two calling conventions correctly.
        # ------------------------------------------------------------------
        loss_fn = self.loss_fn   # capture for closure

        def custom_objective(y_pred: np.ndarray, train_data: lgb.Dataset):
            y_true = train_data.get_label()           # extract ground-truth labels
            return loss_fn.gradients(y_true, y_pred)  # natural (y_true, y_pred) order

        train_params = {
            k: v for k, v in self.lgbm_params.items()
            if k not in ("n_estimators", "random_state", "n_jobs", "verbose")
        }
        train_params.update({
            "objective":   custom_objective,
            "metric":      "binary_logloss",
            "seed":        self.random_state,
            "num_threads": self.lgbm_params.get("n_jobs", -1),
            "verbosity":   -1,
        })

        booster = lgb.train(
            params          = train_params,
            train_set       = train_data,
            num_boost_round = self.lgbm_params.get("n_estimators", 500),
            valid_sets      = [train_data, val_data],
            valid_names     = ["train", "val"],
            callbacks       = [
                lgb.early_stopping(self.early_stopping_rounds, verbose=False),
                lgb.log_evaluation(period=50),
            ],
        )
        self._booster = booster

        best_iter     = booster.best_iteration
        train_logloss = booster.best_score.get("train", {}).get("binary_logloss")
        val_logloss   = booster.best_score.get("val",   {}).get("binary_logloss")

        print(f"\n  Best iteration : {best_iter}")
        if train_logloss: print(f"  Train log-loss : {train_logloss:.4f}")
        if val_logloss:   print(f"  Val   log-loss : {val_logloss:.4f}")

        # Threshold optimisation on val using 5-class labels + cost matrix
        val_probs = self.predict_proba(X_val)
        print(f"\n  Optimising thresholds on validation set...")
        self._optimiser = ThresholdOptimiser(
            self.cost_matrix, n_thresholds=self.n_threshold_candidates
        )
        best_thresholds  = self._optimiser.fit(val_probs, y_val_5class, verbose=True)
        val_pred_classes = self._optimiser.predict(val_probs)
        val_total_cost   = self.cost_matrix.total_cost(y_val_5class, val_pred_classes)
        val_breakdown    = self.cost_matrix.cost_breakdown(
            y_val_5class, val_pred_classes, verbose=True
        )

        from sklearn.metrics import roc_auc_score
        val_roc_auc = roc_auc_score(y_val_binary, val_probs)
        print(f"  Val ROC-AUC    : {val_roc_auc:.4f}")

        return TrainingResult(
            experiment_name    = experiment_name,
            loss_name          = self.loss_fn.name,
            loss_params        = self.loss_fn.params,
            model              = booster,
            thresholds         = best_thresholds,
            best_iteration     = best_iter,
            train_logloss      = train_logloss,
            val_logloss        = val_logloss,
            val_total_cost     = val_total_cost,
            val_cost_breakdown = val_breakdown,
            val_probs          = val_probs,
            val_roc_auc        = val_roc_auc,
        )

    # ------------------------------------------------------------------
    def evaluate_test(
        self,
        result:        TrainingResult,
        X_test:        pd.DataFrame,
        y_test_5class: np.ndarray,
        y_test_binary: np.ndarray,
    ) -> TrainingResult:
        """
        Evaluate on test set. Called ONCE after all tuning is complete.
        Uses thresholds from val — never re-fitted on test.
        """
        if self._optimiser is None or self._optimiser.best_thresholds_ is None:
            raise RuntimeError("Call train() before evaluate_test().")

        print(f"\n  Test evaluation : {result.experiment_name}")
        test_probs        = self.predict_proba(X_test)
        test_pred_classes = self._optimiser.predict(test_probs)
        test_total_cost   = self.cost_matrix.total_cost(y_test_5class, test_pred_classes)
        test_breakdown    = self.cost_matrix.cost_breakdown(
            y_test_5class, test_pred_classes, verbose=True
        )
        from sklearn.metrics import (
            roc_auc_score, precision_score, recall_score,
            f1_score, average_precision_score
        )
        test_roc_auc = roc_auc_score(y_test_binary, test_probs)

        # Binary predictions using t1 as the positive-class decision threshold.
        # t1 is the lowest threshold — any vehicle predicted class 1 or above
        # is considered a positive (failure) prediction in binary terms.
        t1 = self._optimiser.best_thresholds_[0]
        test_binary_pred = (test_probs >= t1).astype(int)

        test_precision = precision_score(y_test_binary, test_binary_pred, zero_division=0)
        test_recall    = recall_score(   y_test_binary, test_binary_pred, zero_division=0)
        test_f1        = f1_score(       y_test_binary, test_binary_pred, zero_division=0)
        test_pr_auc    = average_precision_score(y_test_binary, test_probs)

        print(f"  Test total cost : {test_total_cost:.2f}")
        print(f"  Test ROC-AUC    : {test_roc_auc:.4f}")
        print(f"  Test PR-AUC     : {test_pr_auc:.4f}")
        print(f"  Test Precision  : {test_precision:.4f}  (threshold = t1 = {t1:.3f})")
        print(f"  Test Recall     : {test_recall:.4f}")
        print(f"  Test F1         : {test_f1:.4f}")

        result.test_total_cost     = test_total_cost
        result.test_cost_breakdown = test_breakdown
        result.test_probs          = test_probs
        result.test_roc_auc        = test_roc_auc
        result.test_precision      = test_precision
        result.test_recall         = test_recall
        result.test_f1             = test_f1
        result.test_pr_auc         = test_pr_auc
        return result

    # ------------------------------------------------------------------
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """P(failure) in [0,1]. Applies sigmoid to raw log-odds scores."""
        if self._booster is None:
            raise RuntimeError("Call train() first.")
        return 1.0 / (1.0 + np.exp(-self._booster.predict(X)))

    def predict_class(self, X: pd.DataFrame) -> np.ndarray:
        """Predicted temporal class {0..4}. Requires train()."""
        if self._optimiser is None:
            raise RuntimeError("Call train() first.")
        return self._optimiser.predict(self.predict_proba(X))

    # ------------------------------------------------------------------
    def save(self, result: TrainingResult, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({
                "booster_str":     result.model.model_to_string(),
                "thresholds":      result.thresholds,
                "optimiser_state": {
                    "best_thresholds": self._optimiser.best_thresholds_,
                    "best_cost":       self._optimiser.best_cost_,
                },
                "summary":     result.summary(),
                "loss_name":   self.loss_fn.name,
                "loss_params": self.loss_fn.params,
            }, f)
        print(f"  Model saved → {path}")

    def load(self, path: str) -> None:
        with open(path, "rb") as f:
            state = pickle.load(f)
        self._booster                     = lgb.Booster(model_str=state["booster_str"])
        self._optimiser                   = ThresholdOptimiser(self.cost_matrix)
        self._optimiser.best_thresholds_  = state["optimiser_state"]["best_thresholds"]
        self._optimiser.best_cost_        = state["optimiser_state"]["best_cost"]
        print(f"  Model loaded ← {path}  |  loss: {state['loss_name']}")

    # ------------------------------------------------------------------
    def plot_feature_importance(
        self, result: TrainingResult, top_n: int = 30, save_path: str = None
    ) -> None:
        importance = pd.Series(
            result.model.feature_importance(importance_type="gain"),
            index=result.model.feature_name(),
        ).sort_values(ascending=False).head(top_n)
        fig, ax = plt.subplots(figsize=(10, 6))
        importance.sort_values().plot(kind="barh", ax=ax, color="#378ADD", edgecolor="none")
        ax.set_title(f"Feature importance — {result.experiment_name}", fontsize=12)
        ax.set_xlabel("Gain importance")
        ax.tick_params(labelsize=9)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()

    def plot_probability_distribution(
        self, result: TrainingResult, y_true_binary: np.ndarray,
        split: str = "val", save_path: str = None
    ) -> None:
        probs = result.val_probs if split == "val" else result.test_probs
        if probs is None:
            raise ValueError(f"No {split} probabilities in result.")
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.hist(probs[y_true_binary == 0], bins=40, alpha=0.6,
                color="#378ADD", label="Healthy", density=True)
        ax.hist(probs[y_true_binary == 1], bins=40, alpha=0.6,
                color="#D85A30", label="Failure", density=True)
        if result.thresholds:
            for i, t in enumerate(result.thresholds):
                ax.axvline(t, color="#888780", linewidth=0.8, linestyle="--",
                           label=f"t{i+1}={t:.2f}")
        ax.set_xlabel("P(failure)")
        ax.set_ylabel("Density")
        ax.set_title(f"Probability distribution — {result.experiment_name} ({split})")
        ax.legend(fontsize=9, ncol=2)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def make_trainer(
    experiment:   str,
    cost_matrix:  CostMatrix,
    gamma:        float = 2.0,
    fn_weight:    float = 1.0,
    lgbm_params:  dict  = None,
    random_state: int   = 42,
) -> LGBMTrainer:
    """
    Create the correct LGBMTrainer for the named experiment.

    experiment  'log_loss' | 'focal_loss' | 'cost_aware_focal_loss'
    gamma       focal / cost-aware gamma (tuned per experiment via Optuna)
    fn_weight   scales FN costs for sensitivity analysis (Exp 3)
    """
    loss_map = {
        "log_loss":              LogLoss(),
        "focal_loss":            FocalLoss(gamma=gamma),
        "cost_aware_focal_loss": CostAwareFocalLoss(
            cost_matrix=cost_matrix, gamma=gamma, fn_weight=fn_weight
        ),
    }
    if experiment not in loss_map:
        raise ValueError(
            f"Unknown experiment '{experiment}'. "
            f"Valid: {list(loss_map.keys())}"
        )
    return LGBMTrainer(
        loss_fn=loss_map[experiment],
        cost_matrix=cost_matrix,
        lgbm_params=lgbm_params,
        random_state=random_state,
    )