"""
run_experiments.py
==================
Single Entry Point — SCANIA ComponentX Predictive Maintenance
Thesis: Cost-Aware Learning for Failure Prediction in Industrial Predictive Maintenance

Usage
-----
    # Run everything end-to-end
    python run_experiments.py --all

    # Run only preprocessing pipeline (skip experiments)
    python run_experiments.py --pipeline-only

    # Run specific experiments
    python run_experiments.py --exp1
    python run_experiments.py --exp2
    python run_experiments.py --exp3
    python run_experiments.py --exp1 --exp2 --exp3

    # Run sensitivity analysis only (needs pre-trained exp3 results)
    python run_experiments.py --sensitivity

    # Reload saved pipeline, skip re-preprocessing
    python run_experiments.py --all --skip-pipeline

    # Override data directory
    python run_experiments.py --all --data-dir /path/to/data

    # Quick run with fewer Optuna trials (for development/testing)
    python run_experiments.py --all --fast

    # Generate figures only from saved results
    python run_experiments.py --figures-only

Full end-to-end flow
--------------------
    Step 1  Pipeline        — preprocess all three splits, select features
    Step 2  Labels          — load 5-class temporal labels for val and test
    Step 3  Experiments     — train and evaluate three LightGBM models
    Step 4  Sensitivity     — sweep fn_weight across Exp3
    Step 5  Track results   — log all results, save CSV
    Step 6  Figures         — generate all thesis figures
    Step 7  Report          — write plain-text experiment report
"""

import os
import sys
import time
import pickle
import argparse
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Path setup — make src/ importable from project root
# ---------------------------------------------------------------------------
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from src.scania_pipeline  import SCANIAPipeline
from src.cost_matrix  import CostMatrix
from src.experiments  import (
    run_exp1_log_loss,
    run_exp2_focal_loss,
    run_exp3_cost_aware_focal,
    run_exp4_cost_aware_fn05,
    run_sensitivity_analysis,
    build_comparison_table,
)
from src.results_tracker import (
    ResultsTracker,
    FigureGenerator,
    ReportGenerator,
)


# =============================================================================
# CONFIGURATION — all paths and defaults in one place
# =============================================================================

CONFIG = {
    # Data directories
    "data_dir":        os.path.join(ROOT, "data", "raw"),
    "processed_dir":   os.path.join(ROOT, "data", "processed"),

    # Output directories
    "models_dir":      os.path.join(ROOT, "outputs", "models"),
    "figures_dir":     os.path.join(ROOT, "outputs", "figures"),
    "results_dir":     os.path.join(ROOT, "outputs", "results"),

    # Pipeline state file
    "pipeline_path":   os.path.join(ROOT, "outputs", "models", "pipeline_state.pkl"),

    # Optuna trial counts (full run) — 100 trials for meaningful search
    "n_trials_exp1":   100,
    "n_trials_exp2":   100,
    "n_trials_exp3":   100,
    "n_trials_exp4":   100,
    "n_trials_sens":    30,

    # Optuna trial counts (fast/dev run)
    "n_trials_fast":   5,

    # Sensitivity fn_weights — expanded range including 0.25
    "fn_weights":      [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0],

    # Reproducibility
    "random_state":    42,
}


# =============================================================================
# STEP 1 — PIPELINE
# =============================================================================

def run_pipeline(
    data_dir:    str,
    pipeline_path: str,
    cost_matrix: CostMatrix,
    plot:        bool = False,
) -> tuple[pd.DataFrame, pd.Series,
           pd.DataFrame, pd.Series,
           pd.DataFrame, pd.Series]:
    """
    Run the full preprocessing pipeline on all three splits.

    Returns
    -------
    X_train, y_train  — training features and binary labels
    X_val,   y_val    — validation features and binary labels
    X_test,  y_test   — test features and binary labels
    """
    print_section("STEP 1 — Preprocessing pipeline")

    pipeline = SCANIAPipeline(
        data_dir           = data_dir,
        variance_threshold = 0.01,
        corr_threshold     = 0.95,
        mi_top_n           = 80,
        lgbm_top_n         = 80,
        perm_top_n         = 80,
        min_methods        = 2,
        cost_matrix        = {
            "fp_cost": int(cost_matrix.matrix[0, 1]),
            "fn_cost": int(cost_matrix.matrix[1, 0]),
        },
        plot = plot,
    )

    X_train, y_train = pipeline.fit_transform(split="train")
    X_val,   y_val   = pipeline.transform(split="val")
    X_test,  y_test  = pipeline.transform(split="test")

    # Save pipeline state
    os.makedirs(os.path.dirname(pipeline_path), exist_ok=True)
    pipeline.save(pipeline_path)

    # Save flat tables as parquet for quick reload
    _save_processed(X_train, y_train, X_val, y_val, X_test, y_test)

    print(f"\n  Pipeline complete")
    print(f"  X_train : {X_train.shape}   y_train : {y_train.value_counts().to_dict()}")
    print(f"  X_val   : {X_val.shape}   y_val   : {y_val.value_counts().to_dict()}")
    print(f"  X_test  : {X_test.shape}   y_test  : {y_test.value_counts().to_dict()}")

    return X_train, y_train, X_val, y_val, X_test, y_test


def load_pipeline_outputs() -> tuple:
    """
    Reload previously saved pipeline outputs from parquet.
    Use when --skip-pipeline is set.
    """
    print_section("STEP 1 — Loading saved pipeline outputs")
    processed = CONFIG["processed_dir"]

    X_train = pd.read_parquet(os.path.join(processed, "X_train.parquet"))
    y_train = pd.read_parquet(os.path.join(processed, "y_train.parquet")).squeeze()
    X_val   = pd.read_parquet(os.path.join(processed, "X_val.parquet"))
    y_val   = pd.read_parquet(os.path.join(processed, "y_val.parquet")).squeeze()
    X_test  = pd.read_parquet(os.path.join(processed, "X_test.parquet"))
    y_test  = pd.read_parquet(os.path.join(processed, "y_test.parquet")).squeeze()

    print(f"  Loaded X_train: {X_train.shape}  X_val: {X_val.shape}  X_test: {X_test.shape}")
    return X_train, y_train, X_val, y_val, X_test, y_test


def _save_processed(
    X_train, y_train, X_val, y_val, X_test, y_test
) -> None:
    """Save flat tables as parquet for fast reload on subsequent runs."""
    d = CONFIG["processed_dir"]
    os.makedirs(d, exist_ok=True)
    X_train.to_parquet(os.path.join(d, "X_train.parquet"))
    y_train.to_frame().to_parquet(os.path.join(d, "y_train.parquet"))
    X_val.to_parquet(os.path.join(d, "X_val.parquet"))
    y_val.to_frame().to_parquet(os.path.join(d, "y_val.parquet"))
    X_test.to_parquet(os.path.join(d, "X_test.parquet"))
    y_test.to_frame().to_parquet(os.path.join(d, "y_test.parquet"))
    print(f"  Flat tables saved → {d}")


# =============================================================================
# STEP 2 — LABEL PREPARATION
# =============================================================================

def prepare_labels(
    data_dir:    str,
    y_val:       pd.Series,
    y_test:      pd.Series,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Prepare the two label forms needed downstream.

    Val and test both have 5-class temporal labels (from their label files).
    The pipeline's transform() already converts them to binary for y_val / y_test.
    Here we additionally load the raw 5-class labels for cost evaluation.

    Returns
    -------
    y_val_binary    np.ndarray  binary {0,1} — from pipeline transform
    y_val_5class    np.ndarray  temporal classes {0..4} — from label file
    y_test_binary   np.ndarray  binary {0,1}
    y_test_5class   np.ndarray  temporal classes {0..4}
    """
    print_section("STEP 2 — Label preparation")

    # Binary labels come directly from the pipeline transform output
    y_val_binary  = y_val.values.astype(int)
    y_test_binary = y_test.values.astype(int)

    # 5-class temporal labels — load from raw label files
    val_labels  = _load_temporal_labels(data_dir, split="val")
    test_labels = _load_temporal_labels(data_dir, split="test")

    y_val_5class  = val_labels.values.astype(int)
    y_test_5class = test_labels.values.astype(int)

    print(f"  Val  binary dist  : {dict(zip(*np.unique(y_val_binary,  return_counts=True)))}")
    print(f"  Val  5-class dist : {dict(zip(*np.unique(y_val_5class,  return_counts=True)))}")
    print(f"  Test binary dist  : {dict(zip(*np.unique(y_test_binary, return_counts=True)))}")
    print(f"  Test 5-class dist : {dict(zip(*np.unique(y_test_5class, return_counts=True)))}")

    return y_val_binary, y_val_5class, y_test_binary, y_test_5class


def _load_temporal_labels(data_dir: str, split: str) -> pd.Series:
    """
    Load raw 5-class temporal labels from val/test label CSV files.
    Returns a Series of class_label values aligned by row index.
    """
    filename = "validation_labels.csv" if split == "val" else "test_labels.csv"
    path     = os.path.join(data_dir, filename)
    df       = pd.read_csv(path)
    return df["class_label"].reset_index(drop=True)


# =============================================================================
# STEP 3 — EXPERIMENTS
# =============================================================================

def run_experiments(
    X_train:       pd.DataFrame,
    y_train:       pd.Series,
    X_val:         pd.DataFrame,
    y_val_binary:  np.ndarray,
    y_val_5class:  np.ndarray,
    X_test:        pd.DataFrame,
    y_test_5class: np.ndarray,
    y_test_binary: np.ndarray,
    cost_matrix:   CostMatrix,
    run_exp1:      bool = True,
    run_exp2:      bool = True,
    run_exp3:      bool = True,
    run_exp4:      bool = True,
    n_trials_exp1: int  = 100,
    n_trials_exp2: int  = 100,
    n_trials_exp3: int  = 100,
    n_trials_exp4: int  = 100,
    random_state:  int  = 42,
    models_dir:    str  = "outputs/models",
) -> dict:
    """
    Run the selected experiments and return their results.

    Returns
    -------
    dict with keys 'exp1', 'exp2', 'exp3', 'exp4' (whichever were run)
    """
    results = {}

    # Convert binary labels to pandas Series for LightGBM compatibility
    y_train_s      = pd.Series(y_train.astype(int))
    y_val_binary_s = pd.Series(y_val_binary.astype(int))

    if run_exp1:
        print_section("STEP 3a — Experiment 1: Log-loss + scale_pos_weight (baseline)")
        t0 = time.time()
        results["exp1"] = run_exp1_log_loss(
            X_train, y_train_s,
            X_val, y_val_binary_s, y_val_5class,
            X_test, y_test_5class, y_test_binary,
            cost_matrix    = cost_matrix,
            n_trials       = n_trials_exp1,
            random_state   = random_state,
            model_save_dir = models_dir,
        )
        print(f"  Exp1 completed in {time.time()-t0:.1f}s")

    if run_exp2:
        print_section("STEP 3b — Experiment 2: Focal loss")
        t0 = time.time()
        results["exp2"] = run_exp2_focal_loss(
            X_train, y_train_s,
            X_val, y_val_binary_s, y_val_5class,
            X_test, y_test_5class, y_test_binary,
            cost_matrix    = cost_matrix,
            n_trials       = n_trials_exp2,
            random_state   = random_state,
            model_save_dir = models_dir,
        )
        print(f"  Exp2 completed in {time.time()-t0:.1f}s")

    if run_exp3:
        print_section("STEP 3c — Experiment 3: Cost-aware focal (fn_weight=1.0)")
        t0 = time.time()
        results["exp3"] = run_exp3_cost_aware_focal(
            X_train, y_train_s,
            X_val, y_val_binary_s, y_val_5class,
            X_test, y_test_5class, y_test_binary,
            cost_matrix    = cost_matrix,
            fn_weight      = 1.0,
            n_trials       = n_trials_exp3,
            random_state   = random_state,
            model_save_dir = models_dir,
        )
        print(f"  Exp3 completed in {time.time()-t0:.1f}s")

    if run_exp4:
        print_section("STEP 3d — Experiment 4: Cost-aware focal (fn_weight=0.5) [KEY FINDING]")
        t0 = time.time()
        results["exp4"] = run_exp4_cost_aware_fn05(
            X_train, y_train_s,
            X_val, y_val_binary_s, y_val_5class,
            X_test, y_test_5class, y_test_binary,
            n_trials       = n_trials_exp4,
            random_state   = random_state,
            model_save_dir = models_dir,
        )
        print(f"  Exp4 completed in {time.time()-t0:.1f}s")

    return results


# =============================================================================
# STEP 4 — SENSITIVITY ANALYSIS
# =============================================================================

def run_sensitivity(
    X_train:       pd.DataFrame,
    y_train:       pd.Series,
    X_val:         pd.DataFrame,
    y_val_binary:  np.ndarray,
    y_val_5class:  np.ndarray,
    X_test:        pd.DataFrame,
    y_test_5class: np.ndarray,
    y_test_binary: np.ndarray,
    fn_weights:    list,
    n_trials:      int,
    random_state:  int,
    models_dir:    str,
) -> pd.DataFrame:
    """Run sensitivity analysis — Exp3 across fn_weight values."""
    print_section("STEP 4 — Sensitivity analysis (RQ2 / RQ3)")

    y_train_s      = pd.Series(y_train.astype(int))
    y_val_binary_s = pd.Series(y_val_binary.astype(int))

    t0 = time.time()
    sensitivity_df = run_sensitivity_analysis(
        X_train, y_train_s,
        X_val, y_val_binary_s, y_val_5class,
        X_test, y_test_5class, y_test_binary,
        fn_weights     = fn_weights,
        n_trials       = n_trials,
        random_state   = random_state,
        model_save_dir = models_dir,
    )
    print(f"  Sensitivity analysis completed in {time.time()-t0:.1f}s")
    return sensitivity_df


# =============================================================================
# STEP 5 — TRACK RESULTS
# =============================================================================

def track_and_save(
    experiment_results: dict,
    sensitivity_df:     pd.DataFrame,
    results_dir:        str,
) -> ResultsTracker:
    """
    Log all experiment results and save to CSV.

    Returns
    -------
    ResultsTracker  populated tracker for figure generation
    """
    print_section("STEP 5 — Tracking and saving results")

    tracker = ResultsTracker(output_dir=results_dir)

    for key in ("exp1", "exp2", "exp3", "exp4"):
        if key in experiment_results:
            tracker.log(experiment_results[key], split="test")

    if sensitivity_df is not None:
        tracker.log_sensitivity(sensitivity_df)

    # Guard: only save if at least one result was logged
    if not tracker.records:
        print("  [Tracker] No results to save — all experiments may have been skipped.")
        return tracker

    tracker.save_all_results()
    tracker.print_summary()

    # Save comparison table
    if len(tracker.records) > 0:
        comparison = tracker.comparison_table()
        comparison_path = os.path.join(results_dir, "comparison_table.csv")
        comparison.to_csv(comparison_path, index=False)
        print(f"\n  Comparison table saved → {comparison_path}")
        print(f"\n{comparison.to_string(index=False)}")

    return tracker


# =============================================================================
# STEP 6 — FIGURES
# =============================================================================

def generate_figures(
    tracker:       ResultsTracker,
    y_val_binary:  np.ndarray,
    y_test_binary: np.ndarray,
    y_test_5class: np.ndarray,
    cost_matrix:   CostMatrix,
    figures_dir:   str,
) -> None:
    """Generate and save all thesis figures."""
    print_section("STEP 6 — Generating thesis figures")

    gen   = FigureGenerator(tracker, figures_dir=figures_dir)
    saved = gen.save_all_figures(
        y_val_binary  = y_val_binary,
        y_test_binary = y_test_binary,
        y_test_5class = y_test_5class,
        cost_matrix   = cost_matrix,
    )

    print(f"\n  {len(saved)} figures saved to {figures_dir}:")
    for name, path in saved.items():
        print(f"    {name:<35} → {os.path.basename(path)}")


# =============================================================================
# STEP 7 — REPORT
# =============================================================================

def generate_report(
    tracker:     ResultsTracker,
    results_dir: str,
) -> None:
    """Generate and save the plain-text experiment report."""
    print_section("STEP 7 — Generating experiment report")

    reporter = ReportGenerator(tracker, output_dir=results_dir)
    report   = reporter.generate(save=True)

    # Print a short preview
    lines = report.split("\n")
    print("\n  Report preview (first 20 lines):")
    for line in lines[:20]:
        print(f"    {line}")
    print(f"  ... ({len(lines)} lines total)")


# =============================================================================
# FIGURES-ONLY MODE
# =============================================================================

def figures_only_mode(
    results_dir:  str,
    figures_dir:  str,
    data_dir:     str,
) -> None:
    """
    Reload saved results and regenerate all figures without re-running experiments.
    Requires results.csv and sensitivity_results.csv to exist.
    """
    print_section("FIGURES-ONLY MODE — reloading saved results")

    tracker = ResultsTracker(output_dir=results_dir)

    # Load saved CSV results
    df = tracker.load_results()
    print(f"  Loaded {len(df)} experiment records")

    # Reload 5-class test labels
    test_labels = _load_temporal_labels(data_dir, split="test")
    val_labels  = _load_temporal_labels(data_dir, split="val")

    # Load binary labels from processed parquet
    y_val_binary  = pd.read_parquet(
        os.path.join(CONFIG["processed_dir"], "y_val.parquet")
    ).squeeze().values.astype(int)
    y_test_binary = pd.read_parquet(
        os.path.join(CONFIG["processed_dir"], "y_test.parquet")
    ).squeeze().values.astype(int)

    # Load sensitivity results if available
    sens_path = os.path.join(results_dir, "sensitivity_results.csv")
    if os.path.exists(sens_path):
        tracker._sensitivity_df = pd.read_csv(sens_path)
        print(f"  Loaded sensitivity results ← {sens_path}")

    cm = CostMatrix(fn_weight=1.0)
    gen = FigureGenerator(tracker, figures_dir=figures_dir)

    # Only generate figures that don't require TrainingResult.probs
    fig = gen.plot_cost_comparison()
    gen._savefig(fig, "cost_comparison")

    if tracker._sensitivity_df is not None:
        fig = gen.plot_sensitivity_curve()
        gen._savefig(fig, "sensitivity_curve")

    print(f"\n  Figures saved to {figures_dir}")


# =============================================================================
# ARGUMENT PARSER
# =============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="SCANIA ComponentX — Predictive Maintenance Experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # What to run
    p.add_argument("--all",          action="store_true",
                   help="Run full pipeline: preprocessing → all experiments → figures")
    p.add_argument("--pipeline-only", action="store_true",
                   help="Run preprocessing pipeline only, no experiments")
    p.add_argument("--exp1",          action="store_true",
                   help="Run Experiment 1 (log-loss baseline)")
    p.add_argument("--exp2",          action="store_true",
                   help="Run Experiment 2 (focal loss)")
    p.add_argument("--exp3",          action="store_true",
                   help="Run Experiment 3 (cost-aware focal loss fn_weight=1.0)")
    p.add_argument("--exp4",          action="store_true",
                   help="Run Experiment 4 (cost-aware focal loss fn_weight=0.5 — key finding)")
    p.add_argument("--sensitivity",   action="store_true",
                   help="Run sensitivity analysis (fn_weight sweep)")
    p.add_argument("--figures-only",  action="store_true",
                   help="Regenerate figures from saved results, skip experiments")

    # Options
    p.add_argument("--skip-pipeline", action="store_true",
                   help="Skip preprocessing, load saved pipeline outputs")
    p.add_argument("--data-dir",      type=str, default=None,
                   help="Override data directory (default: data/raw)")
    p.add_argument("--fast",          action="store_true",
                   help=f"Quick run with {CONFIG['n_trials_fast']} Optuna trials")
    p.add_argument("--no-figures",    action="store_true",
                   help="Skip figure generation")
    p.add_argument("--no-report",     action="store_true",
                   help="Skip report generation")
    p.add_argument("--random-state",  type=int, default=42,
                   help="Random seed (default: 42)")
    p.add_argument("--fn-weights",    type=float, nargs="+",
                   default=None,
                   help="fn_weight values for sensitivity analysis "
                        "(default: 0.5 1.0 1.5 2.0 3.0)")

    return p.parse_args()


# =============================================================================
# HELPERS
# =============================================================================

def print_section(title: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def print_banner() -> None:
    print("\n" + "="*60)
    print("  SCANIA ComponentX — Predictive Maintenance")
    print("  Thesis: Cost-Aware Learning for Failure Prediction")
    print("="*60)
    print(f"  Project root : {ROOT}")
    print(f"  Python path  : {sys.executable}")


def ensure_output_dirs() -> None:
    for key in ("processed_dir", "models_dir", "figures_dir", "results_dir"):
        os.makedirs(CONFIG[key], exist_ok=True)


def get_trial_counts(fast: bool) -> dict:
    if fast:
        n = CONFIG["n_trials_fast"]
        print(f"\n  [Fast mode] Using {n} Optuna trials per experiment")
        return {"exp1": n, "exp2": n, "exp3": n, "sens": n}
    return {
        "exp1": CONFIG["n_trials_exp1"],
        "exp2": CONFIG["n_trials_exp2"],
        "exp3": CONFIG["n_trials_exp3"],
        "exp4": CONFIG["n_trials_exp4"],
        "sens": CONFIG["n_trials_sens"],
    }


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    args = parse_args()
    print_banner()
    ensure_output_dirs()

    # Override data directory if provided
    if args.data_dir:
        CONFIG["data_dir"] = args.data_dir

    # Determine what to run
    run_any_exp = args.all or args.exp1 or args.exp2 or args.exp3 or getattr(args, "exp4", False)
    run_exp1    = args.all or args.exp1
    run_exp2    = args.all or args.exp2
    run_exp3    = args.all or args.exp3
    run_exp4    = args.all or getattr(args, "exp4", False)
    run_sens    = args.all or args.sensitivity

    trials      = get_trial_counts(args.fast)
    random_state = args.random_state
    fn_weights  = args.fn_weights or CONFIG["fn_weights"]
    cm          = CostMatrix(fn_weight=1.0)

    # ------------------------------------------------------------------
    # Figures-only mode — reload and exit early
    # ------------------------------------------------------------------
    if args.figures_only:
        figures_only_mode(
            results_dir = CONFIG["results_dir"],
            figures_dir = CONFIG["figures_dir"],
            data_dir    = CONFIG["data_dir"],
        )
        return

    # ------------------------------------------------------------------
    # Step 1 — Pipeline
    # ------------------------------------------------------------------
    if args.pipeline_only:
        run_pipeline(CONFIG["data_dir"], CONFIG["pipeline_path"], cm)
        print("\n  Pipeline-only run complete.")
        return

    if args.skip_pipeline:
        X_train, y_train, X_val, y_val, X_test, y_test = load_pipeline_outputs()
    else:
        X_train, y_train, X_val, y_val, X_test, y_test = run_pipeline(
            CONFIG["data_dir"], CONFIG["pipeline_path"], cm
        )

    # ------------------------------------------------------------------
    # Step 2 — Labels
    # ------------------------------------------------------------------
    y_val_binary, y_val_5class, y_test_binary, y_test_5class = prepare_labels(
        CONFIG["data_dir"], y_val, y_test
    )

    if not run_any_exp and not run_sens:
        print("\n  No experiments selected. Use --exp1 --exp2 --exp3 or --all.")
        print("  Run python run_experiments.py --help for options.")
        return

    # ------------------------------------------------------------------
    # Step 3 — Experiments
    # ------------------------------------------------------------------
    experiment_results = {}
    if run_any_exp:
        experiment_results = run_experiments(
            X_train       = X_train,
            y_train       = y_train,
            X_val         = X_val,
            y_val_binary  = y_val_binary,
            y_val_5class  = y_val_5class,
            X_test        = X_test,
            y_test_5class = y_test_5class,
            y_test_binary = y_test_binary,
            cost_matrix   = cm,
            run_exp1      = run_exp1,
            run_exp2      = run_exp2,
            run_exp3      = run_exp3,
            run_exp4      = run_exp4,
            n_trials_exp1 = trials["exp1"],
            n_trials_exp2 = trials["exp2"],
            n_trials_exp3 = trials["exp3"],
            n_trials_exp4 = trials["exp4"],
            random_state  = random_state,
            models_dir    = CONFIG["models_dir"],
        )

    # ------------------------------------------------------------------
    # Step 4 — Sensitivity analysis
    # ------------------------------------------------------------------
    sensitivity_df = None
    if run_sens and (run_exp3 or args.sensitivity):
        sensitivity_df = run_sensitivity(
            X_train       = X_train,
            y_train       = y_train,
            X_val         = X_val,
            y_val_binary  = y_val_binary,
            y_val_5class  = y_val_5class,
            X_test        = X_test,
            y_test_5class = y_test_5class,
            y_test_binary = y_test_binary,
            fn_weights    = fn_weights,
            n_trials      = trials["sens"],
            random_state  = random_state,
            models_dir    = CONFIG["models_dir"],
        )

    # ------------------------------------------------------------------
    # Step 5 — Track results
    # ------------------------------------------------------------------
    if experiment_results:
        tracker = track_and_save(
            experiment_results = experiment_results,
            sensitivity_df     = sensitivity_df,
            results_dir        = CONFIG["results_dir"],
        )

        # ------------------------------------------------------------------
        # Step 6 — Figures
        # ------------------------------------------------------------------
        if not args.no_figures:
            generate_figures(
                tracker       = tracker,
                y_val_binary  = y_val_binary,
                y_test_binary = y_test_binary,
                y_test_5class = y_test_5class,
                cost_matrix   = cm,
                figures_dir   = CONFIG["figures_dir"],
            )

        # ------------------------------------------------------------------
        # Step 7 — Report
        # ------------------------------------------------------------------
        if not args.no_report:
            generate_report(tracker, CONFIG["results_dir"])

    # ------------------------------------------------------------------
    # Final summary
    # ------------------------------------------------------------------
    print_section("RUN COMPLETE")
    print(f"  Models   → {CONFIG['models_dir']}")
    print(f"  Results  → {CONFIG['results_dir']}")
    print(f"  Figures  → {CONFIG['figures_dir']}")

    if experiment_results:
        print(f"\n  Experiments completed: {list(experiment_results.keys())}")
        for key, r in experiment_results.items():
            tc  = f"{r.test_total_cost:.2f}" if r.test_total_cost else "N/A"
            auc = f"{r.test_roc_auc:.4f}"    if r.test_roc_auc    else "N/A"
            print(f"    {key} — test cost: {tc}   ROC-AUC: {auc}")


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    main()