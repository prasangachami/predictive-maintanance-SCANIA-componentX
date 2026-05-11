"""
src/cost_matrix.py
==================
Industrial Cost Matrix — SCANIA ComponentX Predictive Maintenance
Thesis: Cost-Aware Learning for Failure Prediction in Industrial Predictive Maintenance

Purpose
-------
This module is the single source of truth for all cost-related calculations.
Every experiment (log-loss, focal loss, cost-aware focal loss) uses this
module to evaluate performance. This guarantees fair comparison across all
three experiments.

The 5x5 industrial cost matrix
-------------------------------
Rows = actual class (temporal proximity to failure)
Cols = predicted class

        Pred 0   Pred 1   Pred 2   Pred 3   Pred 4
Act 0     0        7        8        9       10
Act 1    200       0        7        8        9
Act 2    300      200       0        7        8
Act 3    400      300      200       0        7
Act 4    500      400      300      200       0

Class definitions (temporal windows before failure):
    0 = > 48 time steps before failure  (healthy / far)
    1 = 12–24 time steps before failure
    2 = 24–48 time steps before failure
    3 =  6–12 time steps before failure
    4 =  0– 6 time steps before failure  (imminent failure)

Key design decisions
--------------------
1. The cost matrix operates on 5-class labels — NOT binary labels.
   Binary labels are only used for model training.
   The temporal_class column from val/test label files is used here.

2. The model outputs P(failure) in [0, 1].
   Four thresholds map this probability to a predicted class 0–4.
   These thresholds are optimised on the validation set to minimise total cost.

3. Sensitivity analysis scales the false-negative costs by a weight factor w
   to study how the optimal loss function changes as costs vary.
   This directly addresses RQ2 and RQ3 of the thesis.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from itertools import product
from typing import Optional


# ---------------------------------------------------------------------------
# The industrial cost matrix — defined once, used everywhere
# ---------------------------------------------------------------------------

# Raw cost matrix as documented by the industry
# COST_MATRIX[actual_class][predicted_class]
# Diagonal is 0 (correct prediction costs nothing)
COST_MATRIX_RAW = np.array([
    [  0,   7,   8,   9,  10],   # actual class 0
    [200,   0,   7,   8,   9],   # actual class 1
    [300, 200,   0,   7,   8],   # actual class 2
    [400, 300, 200,   0,   7],   # actual class 3
    [500, 400, 300, 200,   0],   # actual class 4
], dtype=float)

N_CLASSES = 5

# Temporal class descriptions for reporting
CLASS_DESCRIPTIONS = {
    0: "> 48 steps  (healthy)",
    1: "12–24 steps to failure",
    2: "24–48 steps to failure",
    3: " 6–12 steps to failure",
    4: "  0–6 steps to failure  (imminent)",
}


# ---------------------------------------------------------------------------
# CostMatrix class
# ---------------------------------------------------------------------------

class CostMatrix:
    """
    Encapsulates the industrial cost matrix and all cost-related operations.

    Parameters
    ----------
    fn_weight : float
        Multiplicative weight applied to all false-negative costs
        (below-diagonal entries where predicted < actual).
        Default 1.0 = original matrix.
        Used for sensitivity analysis — vary this to answer RQ2 and RQ3.

    fp_weight : float
        Multiplicative weight applied to all false-positive costs
        (above-diagonal entries where predicted > actual).
        Default 1.0 = original matrix.

    Usage
    -----
    cm = CostMatrix()                          # original matrix
    cm_sensitive = CostMatrix(fn_weight=2.0)   # double FN costs for sensitivity analysis
    total = cm.total_cost(y_true, y_pred)
    """

    def __init__(
        self,
        fn_weight: float = 1.0,
        fp_weight: float = 1.0,
    ):
        self.fn_weight = fn_weight
        self.fp_weight = fp_weight
        self.matrix    = self._build_weighted_matrix()

    # ------------------------------------------------------------------
    def _build_weighted_matrix(self) -> np.ndarray:
        """
        Apply fn_weight and fp_weight to the raw cost matrix.

        Below diagonal (predicted < actual) = false negatives → fn_weight
        Above diagonal (predicted > actual) = false positives → fp_weight
        Diagonal (correct predictions)      = always 0
        """
        matrix = COST_MATRIX_RAW.copy()

        for actual in range(N_CLASSES):
            for predicted in range(N_CLASSES):
                if predicted < actual:
                    # False negative: predicted healthier than actual
                    matrix[actual][predicted] *= self.fn_weight
                elif predicted > actual:
                    # False positive: predicted worse than actual
                    matrix[actual][predicted] *= self.fp_weight
                # diagonal stays 0

        return matrix

    # ------------------------------------------------------------------
    def cost_of(self, actual_class: int, predicted_class: int) -> float:
        """
        Return the cost of predicting predicted_class when actual is actual_class.

        Parameters
        ----------
        actual_class    : int  ground truth temporal class {0,1,2,3,4}
        predicted_class : int  model's predicted class     {0,1,2,3,4}
        """
        if not (0 <= actual_class < N_CLASSES):
            raise ValueError(f"actual_class must be in 0–{N_CLASSES-1}, got {actual_class}")
        if not (0 <= predicted_class < N_CLASSES):
            raise ValueError(f"predicted_class must be in 0–{N_CLASSES-1}, got {predicted_class}")

        return float(self.matrix[actual_class][predicted_class])

    # ------------------------------------------------------------------
    def total_cost(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
    ) -> float:
        """
        Compute total industrial cost given arrays of actual and predicted classes.

        Parameters
        ----------
        y_true : array-like of int  actual temporal classes    {0,1,2,3,4}
        y_pred : array-like of int  predicted temporal classes {0,1,2,3,4}

        Returns
        -------
        float  total cost summed across all vehicles
        """
        y_true = np.asarray(y_true, dtype=int)
        y_pred = np.asarray(y_pred, dtype=int)

        if len(y_true) != len(y_pred):
            raise ValueError(
                f"y_true length {len(y_true)} != y_pred length {len(y_pred)}"
            )

        return float(sum(
            self.matrix[a][p]
            for a, p in zip(y_true, y_pred)
        ))

    # ------------------------------------------------------------------
    def cost_breakdown(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        verbose: bool = True,
    ) -> pd.DataFrame:
        """
        Compute detailed cost breakdown showing contribution per (actual, predicted) pair.

        Returns a DataFrame with columns:
            actual_class, predicted_class, n_vehicles, unit_cost, total_cost, pct_of_total

        This is the primary table for the thesis Results chapter.
        """
        y_true = np.asarray(y_true, dtype=int)
        y_pred = np.asarray(y_pred, dtype=int)

        rows = []
        for actual in range(N_CLASSES):
            for predicted in range(N_CLASSES):
                mask   = (y_true == actual) & (y_pred == predicted)
                n      = int(mask.sum())
                if n == 0:
                    continue
                unit   = float(self.matrix[actual][predicted])
                total  = unit * n
                rows.append({
                    "actual_class":    actual,
                    "predicted_class": predicted,
                    "actual_desc":     CLASS_DESCRIPTIONS[actual],
                    "predicted_desc":  CLASS_DESCRIPTIONS[predicted],
                    "n_vehicles":      n,
                    "unit_cost":       unit,
                    "total_cost":      total,
                })

        df = pd.DataFrame(rows)
        if df.empty:
            return df

        df["pct_of_total"] = (df["total_cost"] / df["total_cost"].sum() * 100).round(2)
        df = df.sort_values("total_cost", ascending=False).reset_index(drop=True)

        if verbose:
            grand_total = df["total_cost"].sum()
            print(f"\n{'='*60}")
            print(f"  Cost breakdown  (fn_weight={self.fn_weight}  fp_weight={self.fp_weight})")
            print(f"{'='*60}")
            print(f"  {'Actual':<8} {'Predicted':<10} {'N':>6}  {'Unit':>6}  "
                  f"{'Total':>8}  {'%':>6}")
            print(f"  {'-'*56}")
            for _, row in df.iterrows():
                tag = " ← FN" if row["predicted_class"] < row["actual_class"] else \
                      " ← FP" if row["predicted_class"] > row["actual_class"] else ""
                print(f"  {row['actual_class']:<8} {row['predicted_class']:<10} "
                      f"{row['n_vehicles']:>6}  {row['unit_cost']:>6.0f}  "
                      f"{row['total_cost']:>8.0f}  {row['pct_of_total']:>5.1f}%{tag}")
            print(f"  {'-'*56}")
            print(f"  {'TOTAL':<30} {grand_total:>8.0f}")
            print(f"{'='*60}\n")

        return df

    # ------------------------------------------------------------------
    def confusion_matrix_with_costs(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
    ) -> pd.DataFrame:
        """
        Build a 5x5 confusion matrix annotated with costs.
        Each cell shows: count  (cost contribution).
        Useful for thesis figures.
        """
        y_true = np.asarray(y_true, dtype=int)
        y_pred = np.asarray(y_pred, dtype=int)

        cm = np.zeros((N_CLASSES, N_CLASSES), dtype=int)
        for a, p in zip(y_true, y_pred):
            cm[a][p] += 1

        cost_cm = np.zeros((N_CLASSES, N_CLASSES), dtype=float)
        for a in range(N_CLASSES):
            for p in range(N_CLASSES):
                cost_cm[a][p] = cm[a][p] * self.matrix[a][p]

        index   = [f"Actual {i}" for i in range(N_CLASSES)]
        columns = [f"Pred {i}"   for i in range(N_CLASSES)]
        return pd.DataFrame(cost_cm, index=index, columns=columns)

    # ------------------------------------------------------------------
    def summary(self) -> dict:
        """Return a summary dict for logging and comparison tables."""
        return {
            "fn_weight": self.fn_weight,
            "fp_weight": self.fp_weight,
            "max_fn_cost": float(self.matrix[4][0]),
            "max_fp_cost": float(self.matrix[0][4]),
            "total_matrix_sum": float(self.matrix.sum()),
        }


# ---------------------------------------------------------------------------
# Threshold optimiser
# Maps model P(failure) → predicted class 0–4 using 4 thresholds
# ---------------------------------------------------------------------------

class ThresholdOptimiser:
    """
    Find the four probability thresholds that map P(failure) → class 0–4
    to minimise total cost on the validation set.

    How it works
    ------------
    The model outputs P(failure) ∈ [0, 1].
    Four thresholds t1 < t2 < t3 < t4 define five regions:

        P < t1          → predicted class 0  (healthy)
        t1 ≤ P < t2     → predicted class 1
        t2 ≤ P < t3     → predicted class 2
        t3 ≤ P < t4     → predicted class 3
        P ≥ t4          → predicted class 4  (imminent failure)

    The optimiser sweeps candidate threshold combinations and picks
    the one that produces minimum total cost on the validation set.

    Parameters
    ----------
    cost_matrix : CostMatrix
    n_thresholds : int
        Number of candidate values per threshold to sweep.
        Default 20 gives 20^4 = 160,000 combinations — fast enough.
        Increase for finer resolution.
    """

    def __init__(
        self,
        cost_matrix: CostMatrix,
        n_thresholds: int = 50,   # increased from 20 — finer cost surface search
    ):
        self.cost_matrix  = cost_matrix
        self.n_thresholds = n_thresholds
        self.best_thresholds_: Optional[tuple] = None
        self.best_cost_:       Optional[float] = None

    # ------------------------------------------------------------------
    def prob_to_class(
        self,
        y_prob: np.ndarray,
        thresholds: tuple,
    ) -> np.ndarray:
        """
        Map probability array to predicted classes using four thresholds.

        Parameters
        ----------
        y_prob     : array of P(failure) values
        thresholds : (t1, t2, t3, t4)  must be strictly increasing

        Returns
        -------
        np.ndarray of int, predicted classes {0, 1, 2, 3, 4}
        """
        t1, t2, t3, t4 = thresholds
        y_pred = np.zeros(len(y_prob), dtype=int)
        y_pred[y_prob >= t1] = 1
        y_pred[y_prob >= t2] = 2
        y_pred[y_prob >= t3] = 3
        y_pred[y_prob >= t4] = 4
        return y_pred

    # ------------------------------------------------------------------
    def fit(
        self,
        y_prob: np.ndarray,
        y_true_5class: np.ndarray,
        verbose: bool = True,
    ) -> tuple:
        """
        Search for the four thresholds that minimise total cost on val set.

        Parameters
        ----------
        y_prob        : array of P(failure) from the model
        y_true_5class : array of actual temporal classes {0,1,2,3,4}
                        — use temporal_class column from val labels

        Returns
        -------
        best_thresholds : (t1, t2, t3, t4)
        """
        candidates = np.linspace(0.01, 0.99, self.n_thresholds)
        best_cost  = np.inf
        best_t     = (0.25, 0.50, 0.75, 0.90)

        # Sweep all strictly-increasing combinations of 4 thresholds
        for t1, t2, t3, t4 in product(candidates, repeat=4):
            if not (t1 < t2 < t3 < t4):
                continue
            y_pred = self.prob_to_class(y_prob, (t1, t2, t3, t4))
            cost   = self.cost_matrix.total_cost(y_true_5class, y_pred)
            if cost < best_cost:
                best_cost = cost
                best_t    = (t1, t2, t3, t4)

        self.best_thresholds_ = best_t
        self.best_cost_       = best_cost

        if verbose:
            print(f"\n[ThresholdOptimiser] Search complete")
            print(f"  Thresholds : t1={best_t[0]:.3f}  t2={best_t[1]:.3f}"
                  f"  t3={best_t[2]:.3f}  t4={best_t[3]:.3f}")
            print(f"  Min cost   : {best_cost:.2f}")

        return best_t

    # ------------------------------------------------------------------
    def predict(self, y_prob: np.ndarray) -> np.ndarray:
        """
        Map probabilities to classes using fitted thresholds.
        Must call fit() first.
        """
        if self.best_thresholds_ is None:
            raise RuntimeError("Call fit() before predict().")
        return self.prob_to_class(y_prob, self.best_thresholds_)


# ---------------------------------------------------------------------------
# Sensitivity analyser
# ---------------------------------------------------------------------------

class SensitivityAnalyser:
    """
    Study how total cost changes as FN/FP cost ratios vary.

    This directly addresses RQ2 and RQ3:
        RQ2: How does varying failure-class weight affect FN/FP trade-off?
        RQ3: How sensitive is performance to class weights?

    Usage
    -----
    analyser = SensitivityAnalyser(fn_weights=[0.5, 1.0, 1.5, 2.0, 3.0])
    results  = analyser.run(experiments_dict, y_prob_dict, y_true_5class)
    analyser.plot(results)
    """

    def __init__(
        self,
        fn_weights: list[float] = None,
        fp_weights: list[float] = None,
        n_thresholds: int = 20,
    ):
        self.fn_weights   = fn_weights or [0.5, 1.0, 1.5, 2.0, 3.0]
        self.fp_weights   = fp_weights or [1.0]  # usually held fixed
        self.n_thresholds = n_thresholds

    # ------------------------------------------------------------------
    def run(
        self,
        model_probs: dict[str, np.ndarray],
        y_true_5class: np.ndarray,
        verbose: bool = True,
    ) -> pd.DataFrame:
        """
        Run sensitivity analysis across all fn_weight values for each model.

        Parameters
        ----------
        model_probs   : dict  {experiment_name: y_prob array}
                        e.g. {"log_loss": probs1, "focal": probs2, "cost_aware": probs3}
        y_true_5class : np.ndarray  actual temporal classes from val/test

        Returns
        -------
        pd.DataFrame with columns:
            experiment, fn_weight, fp_weight, best_cost, t1, t2, t3, t4
        """
        rows = []

        for fn_w in self.fn_weights:
            for fp_w in self.fp_weights:

                cm        = CostMatrix(fn_weight=fn_w, fp_weight=fp_w)
                optimiser = ThresholdOptimiser(cm, n_thresholds=self.n_thresholds)

                for exp_name, y_prob in model_probs.items():
                    best_t = optimiser.fit(y_prob, y_true_5class, verbose=False)
                    y_pred = optimiser.predict(y_prob)
                    cost   = cm.total_cost(y_true_5class, y_pred)

                    rows.append({
                        "experiment": exp_name,
                        "fn_weight":  fn_w,
                        "fp_weight":  fp_w,
                        "best_cost":  cost,
                        "t1": best_t[0],
                        "t2": best_t[1],
                        "t3": best_t[2],
                        "t4": best_t[3],
                    })

                    if verbose:
                        print(f"  [{exp_name}]  fn_weight={fn_w:.1f}  "
                              f"fp_weight={fp_w:.1f}  cost={cost:.2f}")

        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    def plot(
        self,
        results: pd.DataFrame,
        save_path: str = None,
    ) -> None:
        """
        Plot total cost vs fn_weight for each experiment.
        This is one of the key figures for RQ3 in the thesis Results chapter.
        """
        fig, ax = plt.subplots(figsize=(9, 5))

        colors = {"log_loss": "#378ADD", "focal": "#1D9E75", "cost_aware": "#D85A30"}
        markers = {"log_loss": "o", "focal": "s", "cost_aware": "^"}

        for exp_name, grp in results.groupby("experiment"):
            grp_sorted = grp.sort_values("fn_weight")
            color  = colors.get(exp_name, "#888780")
            marker = markers.get(exp_name, "o")
            ax.plot(
                grp_sorted["fn_weight"],
                grp_sorted["best_cost"],
                marker=marker,
                linewidth=1.5,
                color=color,
                label=exp_name.replace("_", " ").title(),
            )

        ax.set_xlabel("FN cost weight  (1.0 = original matrix)", fontsize=11)
        ax.set_ylabel("Total industrial cost", fontsize=11)
        ax.set_title(
            "Sensitivity analysis — total cost vs false-negative weight",
            fontsize=12,
        )
        ax.legend(fontsize=10)
        ax.grid(axis="y", alpha=0.3)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"  Sensitivity plot saved → {save_path}")

        plt.close()


# ---------------------------------------------------------------------------
# Visualisation helpers
# ---------------------------------------------------------------------------

def plot_cost_matrix(
    cost_matrix: CostMatrix,
    save_path: str = None,
) -> None:
    """
    Plot the 5x5 cost matrix as an annotated heatmap.
    Thesis figure — Methods chapter.
    """
    fig, ax = plt.subplots(figsize=(7, 5))

    # Mask diagonal for colour scaling (diagonal is always 0)
    matrix = cost_matrix.matrix.copy()

    im = ax.imshow(matrix, cmap="YlOrRd", aspect="auto")
    plt.colorbar(im, ax=ax, label="Cost")

    labels = [f"Class {i}" for i in range(N_CLASSES)]
    ax.set_xticks(range(N_CLASSES))
    ax.set_yticks(range(N_CLASSES))
    ax.set_xticklabels([f"Pred {i}" for i in range(N_CLASSES)], fontsize=9)
    ax.set_yticklabels([f"Actual {i}" for i in range(N_CLASSES)], fontsize=9)

    for i in range(N_CLASSES):
        for j in range(N_CLASSES):
            val = matrix[i][j]
            color = "white" if val > 200 else "black"
            ax.text(j, i, f"{val:.0f}", ha="center", va="center",
                    fontsize=10, color=color, fontweight="bold")

    ax.set_title(
        f"Industrial cost matrix  "
        f"(fn_weight={cost_matrix.fn_weight}  fp_weight={cost_matrix.fp_weight})",
        fontsize=11,
    )
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("Actual class")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Cost matrix plot saved → {save_path}")

    plt.close()


def plot_threshold_cost_curve(
    cost_matrix: CostMatrix,
    y_prob: np.ndarray,
    y_true_5class: np.ndarray,
    experiment_name: str,
    save_path: str = None,
) -> None:
    """
    Plot total cost vs the primary threshold t1 (holding t2,t3,t4 fixed at
    evenly-spaced intervals above t1).  Useful for visualising how sensitive
    cost is to the decision boundary.

    Thesis figure — Results chapter.
    """
    t1_values = np.linspace(0.01, 0.70, 100)
    costs = []

    for t1 in t1_values:
        t2 = t1 + 0.10
        t3 = t2 + 0.10
        t4 = t3 + 0.10
        if t4 >= 1.0:
            costs.append(np.nan)
            continue
        optimiser = ThresholdOptimiser(cost_matrix)
        y_pred = optimiser.prob_to_class(y_prob, (t1, t2, t3, t4))
        costs.append(cost_matrix.total_cost(y_true_5class, y_pred))

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(t1_values, costs, color="#185FA5", linewidth=1.5)
    ax.set_xlabel("Primary threshold t1  (t2=t1+0.1, t3=t1+0.2, t4=t1+0.3)")
    ax.set_ylabel("Total industrial cost")
    ax.set_title(f"Cost vs threshold — {experiment_name}")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Threshold cost curve saved → {save_path}")

    plt.close()