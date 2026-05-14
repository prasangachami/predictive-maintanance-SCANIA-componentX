"""
src/results_tracker.py
======================
Results Tracker — Logging, Comparison Tables, and Thesis Figures
Thesis: Cost-Aware Learning for Failure Prediction in Industrial Predictive Maintenance

Purpose
-------
This module collects every result from all three experiments and the
sensitivity analysis, persists them to disk, and produces all the
comparison tables and figures needed for the thesis Results chapter.

Responsibilities
----------------
ResultsTracker
    log(result)            register one TrainingResult
    save_all_results()     write results.csv — one row per experiment
    load_results()         reload from results.csv
    comparison_table()     clean DataFrame for thesis table
    print_summary()        console summary

FigureGenerator
    plot_cost_comparison()          bar chart: Exp1 vs Exp2 vs Exp3 test cost
    plot_roc_comparison()           overlaid ROC curves for all three models
    plot_confusion_matrices()       5x5 cost-weighted confusion matrix per model
    plot_probability_distributions() P(failure) histograms per model
    plot_sensitivity_curve()        total cost vs fn_weight (RQ3 key figure)
    plot_threshold_comparison()     optimal thresholds across experiments
    plot_cost_breakdown_heatmap()   cost breakdown per (actual, predicted) pair
    save_all_figures()              render and save every figure at once

Design decisions
----------------
- ResultsTracker is a plain class with no hidden state — all data lives in
  self._records (list of dicts) and self._results (list of TrainingResult).
  This makes it easy to test and easy to serialise.
- FigureGenerator is separate from ResultsTracker — it only reads, never writes.
  This keeps concerns clean and makes figures independently reproducible.
- All figures use the same colour palette as the rest of the codebase.
- All figures are saved as 150dpi PNGs to outputs/figures/.
"""

import os
import sys
import json
import datetime
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from typing import Optional

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
outputs = os.path.join(ROOT, "backend/", "outputs/")


from src.cost_matrix import (
    CostMatrix, ThresholdOptimiser,
    N_CLASSES, CLASS_DESCRIPTIONS, COST_MATRIX_RAW,
)
from src.model import TrainingResult

# ---------------------------------------------------------------------------
# Colour palette — consistent across all figures
# ---------------------------------------------------------------------------

PALETTE = {
    "exp1_log_loss":             "#378ADD",   # blue
    "exp2_focal_loss":           "#1D9E75",   # teal
    "exp3_cost_aware_focal_loss": "#D85A30",  # coral
    "sensitivity":               "#BA7517",   # amber
    "healthy":                   "#378ADD",
    "failure":                   "#D85A30",
    "grid":                      "#E8E6DF",
    "text":                      "#3d3d3a",
}

EXP_LABELS = {
    "exp1_log_loss":              "Exp 1 — Log-loss",
    "exp2_focal_loss":            "Exp 2 — Focal loss",
    "exp3_cost_aware_focal_loss": "Exp 3 — Cost-aware focal (proposed)",
    "log_loss":                   "Exp 1 — Log-loss",
    "focal_loss":                 "Exp 2 — Focal loss",
    "cost_aware_focal_loss":      "Exp 3 — Cost-aware focal (proposed)",
}


# =============================================================================
# RESULTS TRACKER
# =============================================================================

class ResultsTracker:
    """
    Collects, persists, and reports results from all three experiments.

    Usage
    -----
    tracker = ResultsTracker(output_dir="outputs/results")
    tracker.log(exp1_result, split="test")
    tracker.log(exp2_result, split="test")
    tracker.log(exp3_result, split="test")
    tracker.save_all_results()
    print(tracker.comparison_table())
    tracker.print_summary()
    """

    REQUIRED_SUMMARY_KEYS = (
        "experiment_name", "loss_name",
        "val_total_cost", "test_total_cost",
        "val_roc_auc",    "test_roc_auc",
        "threshold_t1",   "threshold_t2",
        "threshold_t3",   "threshold_t4",
        "best_iteration",
    )

    def __init__(self, output_dir: str = outputs):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self._records:  list[dict]           = []
        self._results:  list[TrainingResult] = []
        self._sensitivity_df: Optional[pd.DataFrame] = None

    # ------------------------------------------------------------------
    def log(
        self,
        result:    TrainingResult,
        split:     str  = "test",
        timestamp: bool = True,
    ) -> None:
        """
        Register one TrainingResult.

        Parameters
        ----------
        result     TrainingResult returned by any run_exp*() function
        split      label for the evaluation split ('val' or 'test')
        timestamp  whether to add a run timestamp to the record
        """
        record = result.summary()
        record["split"]      = split
        record["logged_at"]  = (
            datetime.datetime.now().isoformat(timespec="seconds")
            if timestamp else None
        )

        self._records.append(record)
        self._results.append(result)

        print(f"  [Tracker] logged: {result.experiment_name}"
              f"  test_cost={result.test_total_cost}"
              f"  test_auc={result.test_roc_auc}")

    # ------------------------------------------------------------------
    def log_sensitivity(self, sensitivity_df: pd.DataFrame) -> None:
        """
        Register the sensitivity analysis DataFrame from
        run_sensitivity_analysis().
        """
        self._sensitivity_df = sensitivity_df.copy()
        path = os.path.join(self.output_dir, "sensitivity_results.csv")
        sensitivity_df.to_csv(path, index=False)
        print(f"  [Tracker] sensitivity results saved → {path}")

    # ------------------------------------------------------------------
    def save_all_results(self, filename: str = "results.csv") -> str:
        """
        Write all logged results to a CSV file.
        One row per logged experiment.

        Returns path to the saved file.
        """
        if not self._records:
            raise RuntimeError("No results logged yet. Call log() first.")

        df   = pd.DataFrame(self._records)
        path = os.path.join(self.output_dir, filename)
        df.to_csv(path, index=False)
        print(f"  [Tracker] results saved → {path}  ({len(df)} rows)")
        return path

    # ------------------------------------------------------------------
    def load_results(self, filename: str = "results.csv") -> pd.DataFrame:
        """Reload previously saved results from CSV."""
        path = os.path.join(self.output_dir, filename)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Results file not found: {path}")
        df = pd.read_csv(path)
        print(f"  [Tracker] loaded {len(df)} results ← {path}")
        return df

    # ------------------------------------------------------------------
    def comparison_table(
        self,
        metric_cols: list[str] = None,
    ) -> pd.DataFrame:
        """
        Build a clean comparison table across all logged experiments.

        Columns include val cost, test cost, ROC-AUC, thresholds,
        and percentage cost reduction vs Exp1 (baseline).

        Parameters
        ----------
        metric_cols  list of columns to include.  Defaults to the
                     standard thesis table columns.

        Returns
        -------
        pd.DataFrame  one row per experiment, sorted by test_total_cost
        """
        if not self._records:
            raise RuntimeError("No results logged. Call log() first.")

        default_cols = [
            "experiment_name", "loss_name",
            "val_total_cost",  "test_total_cost",
            "val_roc_auc",     "test_roc_auc",
            "threshold_t1",    "threshold_t4",
            "best_iteration",
        ]
        cols = metric_cols or default_cols
        df   = pd.DataFrame(self._records)

        # Keep only columns that exist
        cols = [c for c in cols if c in df.columns]
        df   = df[cols].copy()

        # Add cost reduction vs baseline (Exp1)
        if "test_total_cost" in df.columns:
            baseline = df.iloc[0]["test_total_cost"]
            df["cost_reduction_pct"] = (
                (baseline - df["test_total_cost"]) / (baseline + 1e-9) * 100
            ).round(2)

        return df.reset_index(drop=True)

    # ------------------------------------------------------------------
    def print_summary(self) -> None:
        """Print a clean summary table to console."""
        if not self._records:
            print("  [Tracker] No results logged yet.")
            return

        print("\n" + "="*70)
        print("  RESULTS SUMMARY")
        print("="*70)
        print(f"  {'Experiment':<35} {'Val cost':>10}  {'Test cost':>10}  {'AUC':>7}")
        print("  " + "─"*65)

        for r in self._records:
            name = r.get("experiment_name", "?")[:34]
            vc   = f"{r['val_total_cost']:.2f}"  if r.get("val_total_cost")  else "N/A"
            tc   = f"{r['test_total_cost']:.2f}" if r.get("test_total_cost") else "N/A"
            auc  = f"{r['test_roc_auc']:.4f}"    if r.get("test_roc_auc")    else "N/A"
            print(f"  {name:<35} {vc:>10}  {tc:>10}  {auc:>7}")

        # Cost reduction
        if len(self._records) >= 2:
            base  = self._records[0].get("test_total_cost", 0)
            last  = self._records[-1].get("test_total_cost", 0)
            delta = base - last
            pct   = delta / (base + 1e-9) * 100
            print(f"\n  Cost change (first → last) : {delta:+.2f}  ({pct:+.1f}%)")

        print("="*70)

    # ------------------------------------------------------------------
    @property
    def results(self) -> list[TrainingResult]:
        """Access raw TrainingResult objects."""
        return self._results

    @property
    def records(self) -> list[dict]:
        """Access flat summary dicts."""
        return self._records


# =============================================================================
# FIGURE GENERATOR
# =============================================================================

class FigureGenerator:
    """
    Generates all thesis figures from logged experiment results.

    All plots are saved to figures_dir at 150 dpi.
    All plots use a consistent colour palette keyed by experiment name.

    Usage
    -----
    gen = FigureGenerator(tracker, figures_dir="backend/outputs/figures")
    gen.save_all_figures(y_val_binary, y_test_binary)
    """

    def __init__(
        self,
        tracker:     ResultsTracker,
        figures_dir: str = "backend/outputs/figures",
    ):
        self.tracker     = tracker
        self.figures_dir = figures_dir
        os.makedirs(figures_dir, exist_ok=True)

    def _savefig(self, fig: plt.Figure, name: str) -> str:
        path = os.path.join(self.figures_dir, f"{name}.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  [Figures] saved → {path}")
        return path

    def _exp_color(self, result: TrainingResult) -> str:
        return PALETTE.get(
            result.experiment_name,
            PALETTE.get(result.loss_name, "#888780")
        )

    def _exp_label(self, result: TrainingResult) -> str:
        return EXP_LABELS.get(
            result.experiment_name,
            EXP_LABELS.get(result.loss_name, result.experiment_name)
        )

    # ------------------------------------------------------------------
    def plot_cost_comparison(self) -> plt.Figure:
        """
        Bar chart comparing val and test total cost across all three experiments.
        Primary figure for RQ1 — thesis Results chapter.
        """
        results = self.tracker.results
        if not results:
            raise RuntimeError("No results logged.")

        labels    = [self._exp_label(r) for r in results]
        val_costs  = [r.val_total_cost  or 0 for r in results]
        test_costs = [r.test_total_cost or 0 for r in results]
        colors     = [self._exp_color(r) for r in results]

        x      = np.arange(len(results))
        width  = 0.35

        fig, ax = plt.subplots(figsize=(10, 5))
        bars_val  = ax.bar(x - width/2, val_costs,  width, label="Validation",
                           color=colors, alpha=0.55, edgecolor="none")
        bars_test = ax.bar(x + width/2, test_costs, width, label="Test",
                           color=colors, alpha=0.90, edgecolor="none")

        # Annotate bars
        for bar in list(bars_val) + list(bars_test):
            h = bar.get_height()
            if h > 0:
                ax.text(bar.get_x() + bar.get_width()/2, h + max(test_costs)*0.01,
                        f"{h:.0f}", ha="center", va="bottom", fontsize=9,
                        color=PALETTE["text"])

        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=10)
        ax.set_ylabel("Total industrial cost", fontsize=11)
        ax.set_title("Total cost comparison — all three experiments", fontsize=12)
        ax.legend(fontsize=10)
        ax.yaxis.grid(True, alpha=0.3, color=PALETTE["grid"])
        ax.set_axisbelow(True)
        plt.tight_layout()
        return fig

    # ------------------------------------------------------------------
    def plot_roc_comparison(
        self,
        y_val_binary:  np.ndarray,
        y_test_binary: np.ndarray,
        split:         str = "test",
    ) -> plt.Figure:
        """
        Overlaid ROC curves for all three models on val or test set.
        Shows overall discriminative ability independently of thresholds.
        """
        from sklearn.metrics import roc_curve, roc_auc_score

        results = self.tracker.results
        fig, ax = plt.subplots(figsize=(7, 6))

        y_true = y_test_binary if split == "test" else y_val_binary

        for r in results:
            probs = r.test_probs if split == "test" else r.val_probs
            if probs is None:
                continue
            fpr, tpr, _ = roc_curve(y_true, probs)
            auc         = roc_auc_score(y_true, probs)
            ax.plot(fpr, tpr, linewidth=1.8,
                    color=self._exp_color(r),
                    label=f"{self._exp_label(r)}  (AUC={auc:.4f})")

        ax.plot([0,1], [0,1], "--", linewidth=0.8, color="#B4B2A9")
        ax.set_xlabel("False positive rate", fontsize=11)
        ax.set_ylabel("True positive rate", fontsize=11)
        ax.set_title(f"ROC curves — {split} set", fontsize=12)
        ax.legend(fontsize=9, loc="lower right")
        ax.yaxis.grid(True, alpha=0.3, color=PALETTE["grid"])
        ax.set_axisbelow(True)
        plt.tight_layout()
        return fig

    # ------------------------------------------------------------------
    def plot_confusion_matrices(
        self,
        y_true_5class: np.ndarray,
        cost_matrix:   CostMatrix,
        split:         str = "test",
    ) -> plt.Figure:
        """
        One 5×5 cost-weighted confusion matrix per experiment.
        Each cell shows: count  (cost contribution).
        """
        results = [r for r in self.tracker.results
                   if (r.test_probs if split=="test" else r.val_probs) is not None]

        n   = len(results)
        fig = plt.figure(figsize=(6*n, 5))
        gs  = gridspec.GridSpec(1, n, figure=fig)

        for idx, r in enumerate(results):
            probs = r.test_probs if split == "test" else r.val_probs
            opt   = ThresholdOptimiser(cost_matrix)
            opt.best_thresholds_ = r.thresholds
            y_pred = opt.predict(probs)

            cm_counts = np.zeros((N_CLASSES, N_CLASSES), dtype=int)
            for a, p in zip(y_true_5class, y_pred):
                cm_counts[a][p] += 1

            cm_costs = cm_counts * cost_matrix.matrix

            ax = fig.add_subplot(gs[idx])
            im = ax.imshow(cm_costs, cmap="YlOrRd", aspect="auto")
            plt.colorbar(im, ax=ax, shrink=0.8, label="Cost")

            for i in range(N_CLASSES):
                for j in range(N_CLASSES):
                    c     = cm_counts[i][j]
                    cost  = cm_costs[i][j]
                    color = "white" if cost > 200 else PALETTE["text"]
                    ax.text(j, i, f"{c}\n({cost:.0f})",
                            ha="center", va="center", fontsize=8, color=color)

            ax.set_xticks(range(N_CLASSES))
            ax.set_yticks(range(N_CLASSES))
            ax.set_xticklabels([f"P{i}" for i in range(N_CLASSES)], fontsize=9)
            ax.set_yticklabels([f"A{i}" for i in range(N_CLASSES)], fontsize=9)
            ax.set_xlabel("Predicted class", fontsize=10)
            ax.set_ylabel("Actual class",    fontsize=10)
            ax.set_title(self._exp_label(r), fontsize=10)

        fig.suptitle(
            f"Cost-weighted confusion matrices — {split} set\n"
            f"cell: count (cost contribution)",
            fontsize=11, y=1.02,
        )
        plt.tight_layout()
        return fig

    # ------------------------------------------------------------------
    def plot_probability_distributions(
        self,
        y_true_binary: np.ndarray,
        split:         str = "test",
    ) -> plt.Figure:
        """
        P(failure) distribution for healthy vs failure vehicles, per model.
        Shows how well each loss function separates the two classes.
        """
        results = [r for r in self.tracker.results
                   if (r.test_probs if split=="test" else r.val_probs) is not None]

        n   = len(results)
        fig, axes = plt.subplots(1, n, figsize=(6*n, 4), sharey=False)
        if n == 1:
            axes = [axes]

        for ax, r in zip(axes, results):
            probs = r.test_probs if split == "test" else r.val_probs

            ax.hist(probs[y_true_binary == 0], bins=30, alpha=0.6,
                    color=PALETTE["healthy"], label="Healthy (0)", density=True)
            ax.hist(probs[y_true_binary == 1], bins=30, alpha=0.7,
                    color=PALETTE["failure"], label="Failure (1)", density=True)

            if r.thresholds:
                for i, t in enumerate(r.thresholds):
                    ax.axvline(t, color="#888780", linewidth=0.9,
                               linestyle="--", label=f"t{i+1}={t:.2f}")

            ax.set_xlabel("P(failure)", fontsize=10)
            ax.set_ylabel("Density",    fontsize=10)
            ax.set_title(self._exp_label(r), fontsize=10)
            ax.legend(fontsize=8, ncol=2)
            ax.yaxis.grid(True, alpha=0.3, color=PALETTE["grid"])

        fig.suptitle(
            f"P(failure) distributions — {split} set",
            fontsize=12, y=1.02,
        )
        plt.tight_layout()
        return fig

    # ------------------------------------------------------------------
    def plot_sensitivity_curve(self) -> plt.Figure:
        """
        Total cost vs fn_weight for each experiment.
        Key figure for RQ3 — shows how sensitive each approach is
        to the FN/FP cost ratio.

        Requires log_sensitivity() to have been called on the tracker.
        """
        df = self.tracker._sensitivity_df
        if df is None:
            raise RuntimeError(
                "No sensitivity results. Call tracker.log_sensitivity() first."
            )

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Left: val cost vs fn_weight
        axes[0].plot(
            df["fn_weight"], df["val_total_cost"],
            marker="o", linewidth=1.8,
            color=PALETTE["exp3_cost_aware_focal_loss"],
            label="Exp 3 — cost-aware focal",
        )
        axes[0].set_xlabel("FN cost weight", fontsize=11)
        axes[0].set_ylabel("Total cost (validation)", fontsize=11)
        axes[0].set_title("Validation cost vs FN weight", fontsize=12)
        axes[0].yaxis.grid(True, alpha=0.3, color=PALETTE["grid"])
        axes[0].axvline(1.0, color="#888780", linewidth=0.8,
                        linestyle=":", label="Original matrix (w=1.0)")
        axes[0].legend(fontsize=9)

        # Right: alpha vs fn_weight
        if "alpha" in df.columns:
            axes[1].plot(
                df["fn_weight"], df["alpha"],
                marker="s", linewidth=1.8,
                color=PALETTE["sensitivity"],
            )
            axes[1].set_xlabel("FN cost weight", fontsize=11)
            axes[1].set_ylabel("Alpha (positive class weight)", fontsize=11)
            axes[1].set_title("Derived alpha vs FN weight", fontsize=12)
            axes[1].yaxis.grid(True, alpha=0.3, color=PALETTE["grid"])
            axes[1].set_ylim(0, 1.05)
            axes[1].axhline(0.5, color="#888780", linewidth=0.8,
                            linestyle=":", label="Equal weighting (α=0.5)")
            axes[1].legend(fontsize=9)

        fig.suptitle(
            "Sensitivity analysis — how FN/FP cost ratio affects model behaviour",
            fontsize=12,
        )
        plt.tight_layout()
        return fig

    # ------------------------------------------------------------------
    def plot_threshold_comparison(self) -> plt.Figure:
        """
        Side-by-side comparison of optimal thresholds across experiments.
        Shows how each loss function shifts the decision boundaries.
        Directly supports RQ2 discussion.
        """
        results = [r for r in self.tracker.results if r.thresholds]

        fig, ax = plt.subplots(figsize=(9, 4))

        y_positions = np.arange(len(results))
        threshold_labels = ["t1 (class 0→1)", "t2 (class 1→2)",
                            "t3 (class 2→3)", "t4 (class 3→4)"]
        threshold_colors = ["#378ADD", "#1D9E75", "#BA7517", "#D85A30"]
        markers = ["o", "s", "^", "D"]

        for ti in range(4):
            vals = [r.thresholds[ti] for r in results]
            ax.scatter(vals, y_positions,
                       color=threshold_colors[ti],
                       marker=markers[ti], s=80, zorder=3,
                       label=threshold_labels[ti])

        ax.set_yticks(y_positions)
        ax.set_yticklabels([self._exp_label(r) for r in results], fontsize=10)
        ax.set_xlabel("Threshold value", fontsize=11)
        ax.set_title(
            "Optimal decision thresholds per experiment\n"
            "lower thresholds = more aggressive failure prediction",
            fontsize=11,
        )
        ax.set_xlim(-0.02, 1.02)
        ax.axvline(0.5, color="#888780", linewidth=0.8,
                   linestyle=":", label="Default (0.5)")
        ax.xaxis.grid(True, alpha=0.3, color=PALETTE["grid"])
        ax.set_axisbelow(True)
        ax.legend(fontsize=9, ncol=2, loc="lower right")
        plt.tight_layout()
        return fig

    # ------------------------------------------------------------------
    def plot_cost_breakdown_heatmap(
        self,
        result:        TrainingResult,
        y_true_5class: np.ndarray,
        cost_matrix:   CostMatrix,
        split:         str = "test",
    ) -> plt.Figure:
        """
        Heatmap showing cost contribution by (actual, predicted) class pair
        for one experiment.  Reveals exactly which error types dominate the cost.
        """
        probs = result.test_probs if split == "test" else result.val_probs
        if probs is None:
            raise ValueError(f"No {split} probs in result.")

        opt = ThresholdOptimiser(cost_matrix)
        opt.best_thresholds_ = result.thresholds
        y_pred = opt.predict(probs)

        breakdown = cost_matrix.cost_breakdown(y_true_5class, y_pred, verbose=False)

        # Build cost matrix for heatmap
        cm_heatmap = np.zeros((N_CLASSES, N_CLASSES), dtype=float)
        for _, row in breakdown.iterrows():
            a = int(row["actual_class"])
            p = int(row["predicted_class"])
            cm_heatmap[a][p] = row["total_cost"]

        fig, ax = plt.subplots(figsize=(7, 5))
        im = ax.imshow(cm_heatmap, cmap="YlOrRd", aspect="auto")
        plt.colorbar(im, ax=ax, label="Total cost contribution")

        ax.set_xticks(range(N_CLASSES))
        ax.set_yticks(range(N_CLASSES))
        ax.set_xticklabels(
            [f"Pred {i}\n{CLASS_DESCRIPTIONS[i][:12]}" for i in range(N_CLASSES)],
            fontsize=8,
        )
        ax.set_yticklabels(
            [f"Actual {i}\n{CLASS_DESCRIPTIONS[i][:12]}" for i in range(N_CLASSES)],
            fontsize=8,
        )

        total = cm_heatmap.sum()
        for i in range(N_CLASSES):
            for j in range(N_CLASSES):
                v = cm_heatmap[i][j]
                if v > 0:
                    pct   = v / (total + 1e-9) * 100
                    color = "white" if v > total * 0.15 else PALETTE["text"]
                    ax.text(j, i, f"{v:.0f}\n({pct:.1f}%)",
                            ha="center", va="center", fontsize=8, color=color)

        ax.set_xlabel("Predicted class", fontsize=10)
        ax.set_ylabel("Actual class",    fontsize=10)
        ax.set_title(
            f"Cost breakdown heatmap — {self._exp_label(result)} ({split})\n"
            f"Total cost: {total:.0f}",
            fontsize=11,
        )
        plt.tight_layout()
        return fig

    # ------------------------------------------------------------------
    def plot_optuna_history(self, result: TrainingResult) -> plt.Figure:
        """
        Plot Optuna optimisation history for one experiment.
        Shows how val cost improved across trials.
        Useful for Methods chapter — demonstrates the tuning process.
        """
        study = getattr(result, "optuna_study", None)
        if study is None:
            raise ValueError(
                "No Optuna study attached to result. "
                "Run via run_exp*() functions which set result.optuna_study."
            )

        trials     = study.trials
        trial_nums = [t.number for t in trials if t.value is not None]
        values     = [t.value  for t in trials if t.value is not None]
        best_so_far = np.minimum.accumulate(values)

        fig, ax = plt.subplots(figsize=(9, 4))
        ax.scatter(trial_nums, values, s=18, alpha=0.5,
                   color=self._exp_color(result), label="Trial cost")
        ax.plot(trial_nums, best_so_far, linewidth=1.5,
                color=self._exp_color(result), label="Best so far")
        ax.set_xlabel("Trial number", fontsize=11)
        ax.set_ylabel("Val total cost", fontsize=11)
        ax.set_title(
            f"Optuna tuning history — {self._exp_label(result)}",
            fontsize=12,
        )
        ax.legend(fontsize=10)
        ax.yaxis.grid(True, alpha=0.3, color=PALETTE["grid"])
        plt.tight_layout()
        return fig

    # ------------------------------------------------------------------
    def save_all_figures(
        self,
        y_val_binary:   np.ndarray,
        y_test_binary:  np.ndarray,
        y_test_5class:  np.ndarray,
        cost_matrix:    CostMatrix,
    ) -> dict[str, str]:
        """
        Render and save every standard figure.
        Returns dict of {figure_name: saved_path}.

        Parameters
        ----------
        y_val_binary    binary val labels for ROC curves and prob distributions
        y_test_binary   binary test labels
        y_test_5class   temporal test labels for confusion matrices
        cost_matrix     CostMatrix used in the experiments
        """
        saved = {}

        # 1. Cost comparison bar chart
        fig = self.plot_cost_comparison()
        saved["cost_comparison"] = self._savefig(fig, "cost_comparison")

        # 2. ROC comparison
        fig = self.plot_roc_comparison(y_val_binary, y_test_binary, split="test")
        saved["roc_comparison_test"] = self._savefig(fig, "roc_comparison_test")

        fig = self.plot_roc_comparison(y_val_binary, y_test_binary, split="val")
        saved["roc_comparison_val"] = self._savefig(fig, "roc_comparison_val")

        # 3. Confusion matrices
        try:
            fig = self.plot_confusion_matrices(y_test_5class, cost_matrix, split="test")
            saved["confusion_matrices"] = self._savefig(fig, "confusion_matrices")
        except Exception as e:
            print(f"  [Figures] confusion matrices skipped: {e}")

        # 4. Probability distributions
        fig = self.plot_probability_distributions(y_test_binary, split="test")
        saved["prob_distributions"] = self._savefig(fig, "prob_distributions")

        # 5. Threshold comparison
        fig = self.plot_threshold_comparison()
        saved["threshold_comparison"] = self._savefig(fig, "threshold_comparison")

        # 6. Cost breakdown heatmap for Exp3 (proposed method)
        if self.tracker.results:
            exp3 = next(
                (r for r in self.tracker.results
                 if r.loss_name == "cost_aware_focal_loss"), None
            )
            if exp3:
                try:
                    fig = self.plot_cost_breakdown_heatmap(
                        exp3, y_test_5class, cost_matrix, split="test"
                    )
                    saved["cost_breakdown_exp3"] = self._savefig(
                        fig, "cost_breakdown_exp3"
                    )
                except Exception as e:
                    print(f"  [Figures] cost breakdown skipped: {e}")

        # 7. Sensitivity curve if available
        if self.tracker._sensitivity_df is not None:
            try:
                fig = self.plot_sensitivity_curve()
                saved["sensitivity_curve"] = self._savefig(fig, "sensitivity_curve")
            except Exception as e:
                print(f"  [Figures] sensitivity curve skipped: {e}")

        # 8. Optuna history for each experiment
        for r in self.tracker.results:
            if hasattr(r, "optuna_study") and r.optuna_study is not None:
                try:
                    fig  = self.plot_optuna_history(r)
                    name = f"optuna_history_{r.loss_name}"
                    saved[name] = self._savefig(fig, name)
                except Exception as e:
                    print(f"  [Figures] optuna history skipped for "
                          f"{r.experiment_name}: {e}")

        print(f"\n  [Figures] {len(saved)} figures saved to {self.figures_dir}")
        return saved


# =============================================================================
# REPORT GENERATOR
# =============================================================================

class ReportGenerator:
    """
    Assembles a plain-text experiment report from tracker results.
    Written to outputs/results/experiment_report.txt.

    Useful as a quick-check summary and for attaching to thesis appendix.
    """

    def __init__(
        self,
        tracker:     ResultsTracker,
        output_dir:  str = "outputs/results",
    ):
        self.tracker    = tracker
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    # ------------------------------------------------------------------
    def generate(self, save: bool = True) -> str:
        """
        Build the full report string and optionally save it.

        Returns
        -------
        str  full report text
        """
        lines = []
        sep   = "=" * 70

        lines += [
            sep,
            "EXPERIMENT REPORT",
            "Thesis: Cost-Aware Learning for Failure Prediction",
            f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            sep, "",
        ]

        # Research questions
        lines += [
            "RESEARCH QUESTIONS",
            "─" * 40,
            "RQ1: Does cost-aware training reduce total maintenance cost",
            "     compared to standard loss functions?",
            "RQ2: How does varying failure-class weight affect FN/FP trade-off?",
            "RQ3: How sensitive is performance to class weights?",
            "",
        ]

        # Experiment results table
        lines += ["EXPERIMENT RESULTS", "─" * 40]
        header = f"{'Experiment':<35} {'Val cost':>10}  {'Test cost':>10}  {'AUC':>7}"
        lines.append(header)
        lines.append("─" * 65)

        for r in self.tracker.records:
            name = r.get("experiment_name", "?")[:34]
            vc   = f"{r['val_total_cost']:.2f}"  if r.get("val_total_cost")  else "N/A"
            tc   = f"{r['test_total_cost']:.2f}" if r.get("test_total_cost") else "N/A"
            auc  = f"{r['test_roc_auc']:.4f}"    if r.get("test_roc_auc")    else "N/A"
            lines.append(f"{name:<35} {vc:>10}  {tc:>10}  {auc:>7}")

        lines.append("")

        # Cost reduction
        records = self.tracker.records
        if len(records) >= 2:
            base  = records[0].get("test_total_cost", 0) or 0
            exp3r = next(
                (r for r in records if "cost_aware" in r.get("loss_name", "")),
                records[-1]
            )
            last  = exp3r.get("test_total_cost", 0) or 0
            delta = base - last
            pct   = delta / (base + 1e-9) * 100
            lines += [
                "COST REDUCTION ANALYSIS",
                "─" * 40,
                f"Baseline (Exp1 log-loss) test cost : {base:.2f}",
                f"Proposed (Exp3 cost-aware) test cost: {last:.2f}",
                f"Absolute reduction                  : {delta:+.2f}",
                f"Percentage reduction                : {pct:+.1f}%",
                "",
                "INTERPRETATION",
                "─" * 40,
            ]
            if delta > 0:
                lines += [
                    "Cost-aware focal loss (Exp3) achieved lower total industrial",
                    "cost than the log-loss baseline (Exp1), supporting RQ1.",
                    f"The reduction of {pct:.1f}% demonstrates that embedding",
                    "cost-awareness into the training objective — rather than",
                    "only at evaluation — produces a more industrially optimal model.",
                ]
            else:
                lines += [
                    "On this run, Exp3 did not outperform the baseline.",
                    "Consider increasing n_trials for hyperparameter tuning,",
                    "or examine the sensitivity analysis for cost-weight insights.",
                ]

        # Threshold analysis
        lines += ["", "THRESHOLD ANALYSIS", "─" * 40]
        for r in self.tracker.results:
            if r.thresholds:
                t = r.thresholds
                lines.append(
                    f"{self._short_name(r):<30}  "
                    f"t1={t[0]:.3f}  t2={t[1]:.3f}  "
                    f"t3={t[2]:.3f}  t4={t[3]:.3f}"
                )

        # Sensitivity summary
        df = self.tracker._sensitivity_df
        if df is not None:
            lines += ["", "SENSITIVITY ANALYSIS (RQ2 / RQ3)", "─" * 40]
            lines.append(
                df[["fn_weight","alpha","val_total_cost","test_total_cost"]]
                .to_string(index=False)
            )

        lines += ["", sep]
        report = "\n".join(lines)

        if save:
            path = os.path.join(self.output_dir, "experiment_report.txt")
            with open(path, "w") as f:
                f.write(report)
            print(f"  [Report] saved → {path}")

        return report

    def _short_name(self, r: TrainingResult) -> str:
        return EXP_LABELS.get(r.loss_name, r.experiment_name)[:30]