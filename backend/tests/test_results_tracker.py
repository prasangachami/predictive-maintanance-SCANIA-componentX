"""
tests/test_results_tracker.py
==============================
Test suite for src/results_tracker.py

Four test layers:
    Layer 1 — Unit tests         : ResultsTracker log/save/load, comparison_table
    Layer 2 — Data contract tests: output shapes, required columns, file creation
    Layer 3 — Integrity tests    : no mutation after log, reproducibility, CSV round-trip
    Layer 4 — Statistical sanity : cost ordering, report content, palette coverage

Run with:
    pytest tests/test_results_tracker.py -v --tb=short
"""

import pytest
import numpy as np
import pandas as pd
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.cost_matrix import CostMatrix, N_CLASSES
from src.model       import TrainingResult, LogLoss, FocalLoss, CostAwareFocalLoss
from src.results_tracker import (
    ResultsTracker,
    FigureGenerator,
    ReportGenerator,
    PALETTE,
    EXP_LABELS,
)

import lightgbm as lgb


# =============================================================================
# FIXTURES
# =============================================================================

N_VAL  = 50
N_TEST = 50
RNG    = np.random.default_rng(42)


def _tiny_booster():
    """Minimal LightGBM booster for constructing TrainingResult stubs."""
    X = pd.DataFrame({"a": np.arange(1.0, 11.0), "b": np.arange(10.0, 0.0, -1.0)})
    y = np.array([0,0,0,0,0,0,0,1,1,1])
    return lgb.train(
        {"objective": "binary", "verbosity": -1, "num_leaves": 4,
         "min_data_in_leaf": 1},
        lgb.Dataset(X, label=y),
        num_boost_round=5,
    )


def _make_result(
    name:         str,
    loss_name:    str,
    loss_params:  dict,
    val_cost:     float,
    test_cost:    float,
    val_auc:      float  = 0.72,
    test_auc:     float  = 0.70,
    thresholds:   tuple  = (0.15, 0.35, 0.60, 0.80),
    n_val:        int    = N_VAL,
    n_test:       int    = N_TEST,
) -> TrainingResult:
    """Build a fully-populated TrainingResult stub."""
    rng = np.random.default_rng(0)
    return TrainingResult(
        experiment_name    = name,
        loss_name          = loss_name,
        loss_params        = loss_params,
        model              = _tiny_booster(),
        thresholds         = thresholds,
        best_iteration     = 50,
        train_logloss      = 0.45,
        val_logloss        = 0.50,
        val_total_cost     = val_cost,
        test_total_cost    = test_cost,
        val_probs          = rng.uniform(0, 1, n_val),
        test_probs         = rng.uniform(0, 1, n_test),
        val_roc_auc        = val_auc,
        test_roc_auc       = test_auc,
    )


@pytest.fixture(scope="module")
def result_exp1():
    return _make_result(
        "exp1_log_loss", "log_loss", {"loss": "log_loss"},
        val_cost=1200.0, test_cost=1400.0,
    )


@pytest.fixture(scope="module")
def result_exp2():
    return _make_result(
        "exp2_focal_loss", "focal_loss",
        {"loss": "focal_loss", "gamma": 2.0},
        val_cost=1050.0, test_cost=1180.0,
    )


@pytest.fixture(scope="module")
def result_exp3():
    return _make_result(
        "exp3_cost_aware_focal_loss", "cost_aware_focal_loss",
        {"loss": "cost_aware_focal_loss", "gamma": 2.0,
         "alpha": 0.976, "fn_weight": 1.0},
        val_cost=900.0, test_cost=1000.0,
    )


@pytest.fixture(scope="module")
def populated_tracker(tmp_path_factory, result_exp1, result_exp2, result_exp3):
    """A tracker with all three experiments logged."""
    tmpdir  = tmp_path_factory.mktemp("tracker_results")
    tracker = ResultsTracker(output_dir=str(tmpdir))
    tracker.log(result_exp1)
    tracker.log(result_exp2)
    tracker.log(result_exp3)
    return tracker


@pytest.fixture(scope="module")
def sensitivity_df():
    return pd.DataFrame({
        "fn_weight":       [0.5, 1.0, 1.5, 2.0, 3.0],
        "alpha":           [0.93, 0.976, 0.985, 0.990, 0.995],
        "best_gamma":      [1.5,  2.0,   2.2,   2.5,   3.0],
        "val_total_cost":  [1100, 900,  860,  840,  870],
        "test_total_cost": [1200, 1000, 960,  940,  980],
        "val_roc_auc":     [0.71, 0.73, 0.73, 0.74, 0.73],
        "test_roc_auc":    [0.70, 0.71, 0.72, 0.72, 0.71],
        "threshold_t1":    [0.10, 0.15, 0.12, 0.11, 0.10],
        "threshold_t2":    [0.30, 0.35, 0.32, 0.30, 0.28],
        "threshold_t3":    [0.55, 0.60, 0.58, 0.56, 0.54],
        "threshold_t4":    [0.78, 0.80, 0.78, 0.76, 0.74],
    })


@pytest.fixture(scope="module")
def y_val_binary():
    return (RNG.random(N_VAL) < 0.12).astype(int)


@pytest.fixture(scope="module")
def y_test_binary():
    return (RNG.random(N_TEST) < 0.12).astype(int)


@pytest.fixture(scope="module")
def y_test_5class():
    return RNG.choice([0,1,2,3,4], N_TEST, p=[0.88,0.05,0.03,0.02,0.02])


@pytest.fixture(scope="module")
def default_cm():
    return CostMatrix()


# =============================================================================
# LAYER 1 — UNIT TESTS: ResultsTracker
# =============================================================================

class TestResultsTrackerLog:
    """Unit tests for ResultsTracker.log()."""

    def test_log_adds_one_record(self, result_exp1, tmp_path):
        t = ResultsTracker(str(tmp_path))
        t.log(result_exp1)
        assert len(t.records) == 1

    def test_log_stores_training_result(self, result_exp1, tmp_path):
        t = ResultsTracker(str(tmp_path))
        t.log(result_exp1)
        assert len(t.results) == 1
        assert isinstance(t.results[0], TrainingResult)

    def test_log_three_results(self, populated_tracker):
        assert len(populated_tracker.records) == 3
        assert len(populated_tracker.results) == 3

    def test_log_record_has_experiment_name(self, result_exp1, tmp_path):
        t = ResultsTracker(str(tmp_path))
        t.log(result_exp1)
        assert t.records[0]["experiment_name"] == "exp1_log_loss"

    def test_log_record_has_val_cost(self, result_exp1, tmp_path):
        t = ResultsTracker(str(tmp_path))
        t.log(result_exp1)
        assert t.records[0]["val_total_cost"] == pytest.approx(1200.0)

    def test_log_record_has_test_cost(self, result_exp1, tmp_path):
        t = ResultsTracker(str(tmp_path))
        t.log(result_exp1)
        assert t.records[0]["test_total_cost"] == pytest.approx(1400.0)

    def test_log_record_has_split_field(self, result_exp1, tmp_path):
        t = ResultsTracker(str(tmp_path))
        t.log(result_exp1, split="test")
        assert t.records[0]["split"] == "test"

    def test_log_record_has_timestamp(self, result_exp1, tmp_path):
        t = ResultsTracker(str(tmp_path))
        t.log(result_exp1, timestamp=True)
        assert t.records[0]["logged_at"] is not None

    def test_log_no_timestamp(self, result_exp1, tmp_path):
        t = ResultsTracker(str(tmp_path))
        t.log(result_exp1, timestamp=False)
        assert t.records[0]["logged_at"] is None


class TestResultsTrackerSaveLoad:
    """Unit tests for save_all_results and load_results."""

    def test_save_creates_csv_file(self, populated_tracker):
        path = populated_tracker.save_all_results()
        assert os.path.exists(path)

    def test_saved_csv_has_three_rows(self, populated_tracker):
        path = populated_tracker.save_all_results()
        df   = pd.read_csv(path)
        assert len(df) == 3

    def test_save_before_log_raises(self, tmp_path):
        t = ResultsTracker(str(tmp_path))
        with pytest.raises(RuntimeError):
            t.save_all_results()

    def test_load_returns_dataframe(self, populated_tracker):
        populated_tracker.save_all_results()
        df = populated_tracker.load_results()
        assert isinstance(df, pd.DataFrame)

    def test_load_nonexistent_raises(self, tmp_path):
        t = ResultsTracker(str(tmp_path))
        with pytest.raises(FileNotFoundError):
            t.load_results()

    def test_csv_round_trip_preserves_costs(self, populated_tracker):
        """Save then reload — test_total_cost values must be preserved."""
        populated_tracker.save_all_results()
        df = populated_tracker.load_results()
        original = [r["test_total_cost"] for r in populated_tracker.records]
        loaded   = df["test_total_cost"].tolist()
        for o, l in zip(original, loaded):
            assert o == pytest.approx(l, rel=1e-4)


class TestComparisonTable:
    """Unit tests for ResultsTracker.comparison_table()."""

    def test_returns_dataframe(self, populated_tracker):
        df = populated_tracker.comparison_table()
        assert isinstance(df, pd.DataFrame)

    def test_has_three_rows(self, populated_tracker):
        df = populated_tracker.comparison_table()
        assert len(df) == 3

    def test_required_columns_present(self, populated_tracker):
        df = populated_tracker.comparison_table()
        for col in ("experiment_name", "loss_name",
                    "val_total_cost", "test_total_cost",
                    "val_roc_auc",    "test_roc_auc"):
            assert col in df.columns, f"Missing column: {col}"

    def test_cost_reduction_column_present(self, populated_tracker):
        df = populated_tracker.comparison_table()
        assert "cost_reduction_pct" in df.columns

    def test_baseline_cost_reduction_is_zero(self, populated_tracker):
        """First experiment (baseline) must have 0% cost reduction vs itself."""
        df = populated_tracker.comparison_table()
        assert df.iloc[0]["cost_reduction_pct"] == pytest.approx(0.0, abs=0.1)

    def test_exp3_has_positive_cost_reduction(self, populated_tracker):
        """Exp3 test cost (1000) < Exp1 test cost (1400) → positive reduction."""
        df  = populated_tracker.comparison_table()
        exp3 = df[df["experiment_name"] == "exp3_cost_aware_focal_loss"]
        assert exp3["cost_reduction_pct"].values[0] > 0

    def test_custom_metric_cols(self, populated_tracker):
        cols = ["experiment_name", "test_total_cost"]
        df   = populated_tracker.comparison_table(metric_cols=cols)
        assert list(df.columns) == cols + ["cost_reduction_pct"]

    def test_before_log_raises(self, tmp_path):
        t = ResultsTracker(str(tmp_path))
        with pytest.raises(RuntimeError):
            t.comparison_table()


class TestLogSensitivity:
    """Unit tests for log_sensitivity()."""

    def test_saves_csv(self, tmp_path, sensitivity_df):
        t = ResultsTracker(str(tmp_path))
        t.log_sensitivity(sensitivity_df)
        path = os.path.join(str(tmp_path), "sensitivity_results.csv")
        assert os.path.exists(path)

    def test_stores_on_tracker(self, tmp_path, sensitivity_df):
        t = ResultsTracker(str(tmp_path))
        t.log_sensitivity(sensitivity_df)
        assert t._sensitivity_df is not None
        assert len(t._sensitivity_df) == len(sensitivity_df)

    def test_does_not_mutate_input(self, tmp_path, sensitivity_df):
        original_len = len(sensitivity_df)
        t = ResultsTracker(str(tmp_path))
        t.log_sensitivity(sensitivity_df)
        assert len(sensitivity_df) == original_len


# =============================================================================
# LAYER 2 — DATA CONTRACT TESTS: FigureGenerator
# =============================================================================

class TestFigureGeneratorContracts:
    """Verify figures are created, have correct types, and save to disk."""

    @pytest.fixture(scope="class")
    def gen(self, populated_tracker, tmp_path_factory):
        tmpdir = str(tmp_path_factory.mktemp("figures"))
        return FigureGenerator(populated_tracker, figures_dir=tmpdir)

    def test_plot_cost_comparison_returns_figure(self, gen):
        import matplotlib.pyplot as plt
        fig = gen.plot_cost_comparison()
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_cost_comparison_saved_to_disk(self, gen):
        fig  = gen.plot_cost_comparison()
        path = gen._savefig(fig, "test_cost_comp")
        assert os.path.exists(path)

    def test_plot_roc_comparison_returns_figure(
        self, gen, y_val_binary, y_test_binary
    ):
        import matplotlib.pyplot as plt
        fig = gen.plot_roc_comparison(y_val_binary, y_test_binary, split="test")
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_plot_prob_distributions_returns_figure(
        self, gen, y_test_binary
    ):
        import matplotlib.pyplot as plt
        fig = gen.plot_probability_distributions(y_test_binary, split="test")
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_plot_threshold_comparison_returns_figure(self, gen):
        import matplotlib.pyplot as plt
        fig = gen.plot_threshold_comparison()
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_plot_sensitivity_without_data_raises(
        self, populated_tracker, tmp_path
    ):
        gen_empty = FigureGenerator(
            ResultsTracker(str(tmp_path)), figures_dir=str(tmp_path)
        )
        with pytest.raises(RuntimeError):
            gen_empty.plot_sensitivity_curve()

    def test_plot_sensitivity_returns_figure(
        self, populated_tracker, sensitivity_df, tmp_path_factory
    ):
        import matplotlib.pyplot as plt
        tmpdir  = str(tmp_path_factory.mktemp("sens_fig"))
        tracker = ResultsTracker(tmpdir)
        tracker.log(
            _make_result("exp3_cost_aware_focal_loss", "cost_aware_focal_loss",
                         {"loss":"cost_aware_focal_loss","gamma":2.0,
                          "alpha":0.976,"fn_weight":1.0},
                         900.0, 1000.0)
        )
        tracker.log_sensitivity(sensitivity_df)
        gen = FigureGenerator(tracker, figures_dir=tmpdir)
        fig = gen.plot_sensitivity_curve()
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_plot_cost_breakdown_returns_figure(
        self, gen, result_exp3, y_test_5class, default_cm
    ):
        import matplotlib.pyplot as plt
        fig = gen.plot_cost_breakdown_heatmap(
            result_exp3, y_test_5class, default_cm, split="test"
        )
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_save_all_figures_returns_dict(
        self, gen, y_val_binary, y_test_binary, y_test_5class, default_cm
    ):
        saved = gen.save_all_figures(
            y_val_binary, y_test_binary, y_test_5class, default_cm
        )
        assert isinstance(saved, dict)
        assert len(saved) > 0

    def test_save_all_figures_files_exist(
        self, gen, y_val_binary, y_test_binary, y_test_5class, default_cm
    ):
        saved = gen.save_all_figures(
            y_val_binary, y_test_binary, y_test_5class, default_cm
        )
        for name, path in saved.items():
            assert os.path.exists(path), f"Figure not saved: {name} → {path}"


# =============================================================================
# LAYER 2 — DATA CONTRACT TESTS: ReportGenerator
# =============================================================================

class TestReportGeneratorContracts:
    """Data contract tests for ReportGenerator."""

    @pytest.fixture(scope="class")
    def report_gen(self, populated_tracker, tmp_path_factory):
        tmpdir = str(tmp_path_factory.mktemp("report"))
        return ReportGenerator(populated_tracker, output_dir=tmpdir)

    def test_generate_returns_string(self, report_gen):
        report = report_gen.generate(save=False)
        assert isinstance(report, str)

    def test_report_not_empty(self, report_gen):
        report = report_gen.generate(save=False)
        assert len(report) > 100

    def test_report_contains_experiment_names(self, report_gen):
        report = report_gen.generate(save=False)
        assert "exp1" in report.lower() or "log" in report.lower()

    def test_report_contains_rq_references(self, report_gen):
        report = report_gen.generate(save=False)
        assert "RQ1" in report and "RQ2" in report and "RQ3" in report

    def test_report_saved_to_disk(self, report_gen):
        report_gen.generate(save=True)
        path = os.path.join(report_gen.output_dir, "experiment_report.txt")
        assert os.path.exists(path)

    def test_saved_report_readable(self, report_gen):
        report_gen.generate(save=True)
        path = os.path.join(report_gen.output_dir, "experiment_report.txt")
        with open(path) as f:
            content = f.read()
        assert len(content) > 100

    def test_report_contains_cost_values(self, report_gen):
        report = report_gen.generate(save=False)
        # Should contain the test cost values we set
        assert "1400" in report or "1000" in report


# =============================================================================
# LAYER 3 — INTEGRITY TESTS
# =============================================================================

class TestResultsTrackerIntegrity:
    """No mutation, reproducibility, CSV round-trip integrity."""

    def test_log_does_not_mutate_training_result(
        self, result_exp1, tmp_path
    ):
        """Logging must not modify the TrainingResult object."""
        original_cost = result_exp1.test_total_cost
        t = ResultsTracker(str(tmp_path))
        t.log(result_exp1)
        assert result_exp1.test_total_cost == original_cost

    def test_records_are_independent_copies(self, tmp_path):
        """Modifying a record after logging must not affect the tracker."""
        r = _make_result(
            "tmp_exp", "log_loss", {"loss": "log_loss"},
            val_cost=500.0, test_cost=600.0
        )
        t = ResultsTracker(str(tmp_path))
        t.log(r)
        # Modify the result after logging
        r.test_total_cost = 9999.0
        # Tracker's stored record must be unchanged
        assert t.records[0]["test_total_cost"] == pytest.approx(600.0)

    def test_csv_round_trip_experiment_names(self, populated_tracker):
        populated_tracker.save_all_results()
        df = populated_tracker.load_results()
        names_original = [r["experiment_name"] for r in populated_tracker.records]
        names_loaded   = df["experiment_name"].tolist()
        assert names_original == names_loaded

    def test_csv_round_trip_thresholds(self, populated_tracker):
        populated_tracker.save_all_results()
        df = populated_tracker.load_results()
        for i, r in enumerate(populated_tracker.records):
            if r.get("threshold_t1") is not None:
                assert df.iloc[i]["threshold_t1"] == pytest.approx(
                    r["threshold_t1"], rel=1e-4
                )

    def test_sensitivity_df_is_copy(self, tmp_path, sensitivity_df):
        """log_sensitivity must store a copy, not a reference."""
        t = ResultsTracker(str(tmp_path))
        t.log_sensitivity(sensitivity_df)
        sensitivity_df.iloc[0, sensitivity_df.columns.get_loc("fn_weight")] = 999.0
        # Tracker's copy must be unchanged
        assert t._sensitivity_df.iloc[0]["fn_weight"] != 999.0

    def test_multiple_log_calls_accumulate(self, tmp_path):
        t = ResultsTracker(str(tmp_path))
        for i in range(4):
            r = _make_result(
                f"exp_{i}", "log_loss", {"loss": "log_loss"},
                val_cost=float(i*100), test_cost=float(i*110)
            )
            t.log(r)
        assert len(t.records) == 4
        assert len(t.results) == 4

    def test_output_dir_created_if_missing(self, tmp_path):
        new_dir = os.path.join(str(tmp_path), "new", "nested", "dir")
        t = ResultsTracker(new_dir)
        assert os.path.isdir(new_dir)


class TestFigureIntegrity:
    """Figure output integrity tests."""

    def test_cost_comparison_has_three_bars(
        self, populated_tracker, tmp_path_factory
    ):
        """Bar chart should have 2 bars per experiment = 6 bars total."""
        import matplotlib.pyplot as plt
        tmpdir = str(tmp_path_factory.mktemp("fig_integrity"))
        gen = FigureGenerator(populated_tracker, figures_dir=tmpdir)
        fig = gen.plot_cost_comparison()
        ax  = fig.axes[0]
        n_bars = len([p for p in ax.patches if p.get_height() >= 0])
        plt.close(fig)
        # 3 experiments × 2 splits (val + test) = 6
        assert n_bars == 6

    def test_threshold_plot_has_four_series(
        self, populated_tracker, tmp_path_factory
    ):
        """Threshold plot should have 4 scatter series (t1..t4) + 1 vline."""
        import matplotlib.pyplot as plt
        tmpdir = str(tmp_path_factory.mktemp("thresh_integrity"))
        gen = FigureGenerator(populated_tracker, figures_dir=tmpdir)
        fig = gen.plot_threshold_comparison()
        ax  = fig.axes[0]
        # Count PathCollections (scatter series)
        scatter_count = sum(
            1 for c in ax.collections
            if hasattr(c, "get_offsets")
        )
        plt.close(fig)
        assert scatter_count == 4   # t1, t2, t3, t4

    def test_exp_color_returns_string(
        self, populated_tracker, tmp_path_factory, result_exp1
    ):
        tmpdir = str(tmp_path_factory.mktemp("color_test"))
        gen = FigureGenerator(populated_tracker, figures_dir=tmpdir)
        color = gen._exp_color(result_exp1)
        assert isinstance(color, str)
        assert color.startswith("#")


# =============================================================================
# LAYER 4 — STATISTICAL SANITY TESTS
# =============================================================================

class TestStatisticalSanity:
    """Plausibility checks on tracker outputs."""

    def test_exp3_has_lowest_test_cost(self, populated_tracker):
        """
        In our fixture: exp1=1400, exp2=1180, exp3=1000.
        Cost reduction column should show exp3 as best.
        """
        df = populated_tracker.comparison_table()
        exp3 = df[df["experiment_name"] == "exp3_cost_aware_focal_loss"]
        exp1 = df[df["experiment_name"] == "exp1_log_loss"]
        assert exp3["test_total_cost"].values[0] < exp1["test_total_cost"].values[0]

    def test_cost_reduction_increases_exp1_to_exp3(self, populated_tracker):
        """Cost reduction should be positive and increasing from Exp1→Exp3."""
        df       = populated_tracker.comparison_table()
        reductions = df["cost_reduction_pct"].values
        # Exp1 = 0%, Exp2 > 0%, Exp3 > Exp2
        assert reductions[0] == pytest.approx(0.0, abs=0.1)
        assert reductions[1] > 0
        assert reductions[2] > reductions[1]

    def test_all_roc_aucs_in_unit_interval(self, populated_tracker):
        df = populated_tracker.comparison_table()
        assert (df["val_roc_auc"].between(0, 1)).all()
        assert (df["test_roc_auc"].between(0, 1)).all()

    def test_all_costs_non_negative(self, populated_tracker):
        df = populated_tracker.comparison_table()
        assert (df["val_total_cost"]  >= 0).all()
        assert (df["test_total_cost"] >= 0).all()

    def test_palette_covers_all_three_experiments(self):
        """PALETTE must have entries for all three loss names."""
        for loss in ("log_loss", "focal_loss", "cost_aware_focal_loss"):
            assert loss in PALETTE or any(loss in k for k in PALETTE), (
                f"PALETTE missing entry for {loss}"
            )

    def test_exp_labels_covers_all_three_losses(self):
        """EXP_LABELS must have readable labels for all three loss names."""
        for loss in ("log_loss", "focal_loss", "cost_aware_focal_loss"):
            assert loss in EXP_LABELS, f"EXP_LABELS missing {loss}"

    def test_sensitivity_csv_has_correct_columns(self, tmp_path, sensitivity_df):
        t = ResultsTracker(str(tmp_path))
        t.log_sensitivity(sensitivity_df)
        path = os.path.join(str(tmp_path), "sensitivity_results.csv")
        df   = pd.read_csv(path)
        for col in ("fn_weight","alpha","val_total_cost","test_total_cost"):
            assert col in df.columns

    def test_report_cost_reduction_sign_correct(
        self, populated_tracker, tmp_path_factory
    ):
        """
        With exp3 cost (1000) < exp1 cost (1400), report should
        state a positive reduction.
        """
        tmpdir = str(tmp_path_factory.mktemp("report_sign"))
        rg     = ReportGenerator(populated_tracker, output_dir=tmpdir)
        report = rg.generate(save=False)
        # Report must not say cost-aware did not reduce cost
        assert "did not outperform" not in report

    @pytest.mark.parametrize("split", ["val", "test"])
    def test_all_probs_in_unit_interval(
        self, populated_tracker, split
    ):
        for r in populated_tracker.results:
            probs = r.val_probs if split == "val" else r.test_probs
            assert ((probs >= 0) & (probs <= 1)).all(), (
                f"{r.experiment_name} {split} probs out of [0,1]"
            )

    def test_comparison_table_loss_names_correct(self, populated_tracker):
        df = populated_tracker.comparison_table()
        assert "log_loss"              in df["loss_name"].values
        assert "focal_loss"            in df["loss_name"].values
        assert "cost_aware_focal_loss" in df["loss_name"].values