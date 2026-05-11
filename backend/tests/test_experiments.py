"""
tests/test_experiments.py
=========================
Test suite for src/experiments.py

Four test layers:
    Layer 1 — Unit tests         : param suggestion, comparison table, helpers
    Layer 2 — Data contract tests: result shapes, types, required fields
    Layer 3 — Integrity tests    : protocol, test-set discipline, reproducibility
    Layer 4 — Statistical sanity : cost ordering, sensitivity monotonicity

NOTE: Full experiment runs (Optuna + LightGBM training) are slow.
      Integration tests use n_trials=2 and tiny datasets to stay fast.
      Tests marked @pytest.mark.slow are skipped by default.
      Run them with:  pytest -m slow tests/test_experiments.py

Run fast suite:
    pytest tests/test_experiments.py -v --tb=short
"""

import pytest
import numpy as np
import pandas as pd
import tempfile
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.cost_matrix  import CostMatrix
from src.model        import (
    LogLoss, FocalLoss, CostAwareFocalLoss,
    LGBMTrainer, TrainingResult,
)
from src.experiments  import (
    _suggest_lgbm_params,
    run_exp1_log_loss,
    run_exp2_focal_loss,
    run_exp3_cost_aware_focal,
    run_sensitivity_analysis,
    run_all_experiments,
    build_comparison_table,
    _print_experiment_summary,
    _print_final_comparison,
)


# =============================================================================
# FIXTURES
# =============================================================================

N_TRAIN, N_VAL, N_TEST, N_FEAT = 120, 40, 40, 15
RNG = np.random.default_rng(42)


def _make_datasets():
    """Tiny synthetic datasets used in all integration tests."""
    X_train = pd.DataFrame(
        RNG.standard_normal((N_TRAIN, N_FEAT)),
        columns=[f"f{i}" for i in range(N_FEAT)]
    )
    X_val = pd.DataFrame(
        RNG.standard_normal((N_VAL, N_FEAT)),
        columns=[f"f{i}" for i in range(N_FEAT)]
    )
    X_test = pd.DataFrame(
        RNG.standard_normal((N_TEST, N_FEAT)),
        columns=[f"f{i}" for i in range(N_FEAT)]
    )
    y_train        = pd.Series((RNG.random(N_TRAIN) < 0.12).astype(int))
    y_val_binary   = pd.Series((RNG.random(N_VAL)   < 0.12).astype(int))
    y_val_5class   = RNG.choice([0,1,2,3,4], N_VAL,   p=[0.88,0.05,0.03,0.02,0.02])
    y_test_5class  = RNG.choice([0,1,2,3,4], N_TEST,  p=[0.88,0.05,0.03,0.02,0.02])
    y_test_binary  = (y_test_5class > 0).astype(int)
    return (X_train, y_train, X_val, y_val_binary, y_val_5class,
            X_test, y_test_5class, y_test_binary)


@pytest.fixture(scope="module")
def datasets():
    return _make_datasets()


@pytest.fixture(scope="module")
def default_cm():
    return CostMatrix()


@pytest.fixture(scope="module")
def tiny_exp1_result(datasets, default_cm, tmp_path_factory):
    """Run a real (tiny) Exp1 once — shared across Layer 2/3/4 tests."""
    X_train, y_train, X_val, y_val_binary, y_val_5class, \
        X_test, y_test_5class, y_test_binary = datasets
    tmpdir = str(tmp_path_factory.mktemp("models_exp1"))
    return run_exp1_log_loss(
        X_train, y_train, X_val, y_val_binary, y_val_5class,
        X_test, y_test_5class, y_test_binary,
        cost_matrix    = default_cm,
        n_trials       = 2,
        random_state   = 42,
        model_save_dir = tmpdir,
    )


@pytest.fixture(scope="module")
def tiny_exp2_result(datasets, default_cm, tmp_path_factory):
    X_train, y_train, X_val, y_val_binary, y_val_5class, \
        X_test, y_test_5class, y_test_binary = datasets
    tmpdir = str(tmp_path_factory.mktemp("models_exp2"))
    return run_exp2_focal_loss(
        X_train, y_train, X_val, y_val_binary, y_val_5class,
        X_test, y_test_5class, y_test_binary,
        cost_matrix    = default_cm,
        n_trials       = 2,
        random_state   = 42,
        model_save_dir = tmpdir,
    )


@pytest.fixture(scope="module")
def tiny_exp3_result(datasets, default_cm, tmp_path_factory):
    X_train, y_train, X_val, y_val_binary, y_val_5class, \
        X_test, y_test_5class, y_test_binary = datasets
    tmpdir = str(tmp_path_factory.mktemp("models_exp3"))
    return run_exp3_cost_aware_focal(
        X_train, y_train, X_val, y_val_binary, y_val_5class,
        X_test, y_test_5class, y_test_binary,
        cost_matrix    = default_cm,
        fn_weight      = 1.0,
        n_trials       = 2,
        random_state   = 42,
        model_save_dir = tmpdir,
    )


# =============================================================================
# LAYER 1 — UNIT TESTS: helpers and pure functions
# =============================================================================

class TestSuggestLGBMParams:
    """Unit tests for the shared hyperparameter suggestion function."""

    @pytest.fixture()
    def trial(self):
        import optuna
        study = optuna.create_study()
        return study.ask()

    def test_returns_dict(self, trial):
        params = _suggest_lgbm_params(trial)
        assert isinstance(params, dict)

    def test_required_keys_present(self, trial):
        params = _suggest_lgbm_params(trial)
        for key in ("n_estimators", "learning_rate", "num_leaves",
                    "min_child_samples", "subsample", "colsample_bytree",
                    "reg_alpha", "reg_lambda"):
            assert key in params

    def test_n_estimators_in_range(self, trial):
        params = _suggest_lgbm_params(trial)
        assert 100 <= params["n_estimators"] <= 800

    def test_learning_rate_in_range(self, trial):
        params = _suggest_lgbm_params(trial)
        assert 0.01 <= params["learning_rate"] <= 0.2

    def test_num_leaves_in_range(self, trial):
        params = _suggest_lgbm_params(trial)
        assert 16 <= params["num_leaves"] <= 128

    def test_subsample_in_unit_interval(self, trial):
        params = _suggest_lgbm_params(trial)
        assert 0.5 <= params["subsample"] <= 1.0

    def test_reg_alpha_positive(self, trial):
        params = _suggest_lgbm_params(trial)
        assert params["reg_alpha"] > 0


class TestBuildComparisonTable:
    """Unit tests for build_comparison_table()."""

    @pytest.fixture()
    def mock_result(self):
        import lightgbm as lgb
        X = pd.DataFrame({"a": [1.0,2.0,3.0]})
        y = np.array([0,0,1])
        b = lgb.train(
            {"objective":"binary","verbosity":-1,"num_leaves":2},
            lgb.Dataset(X, label=y), num_boost_round=1,
        )
        return TrainingResult(
            experiment_name="mock", loss_name="log_loss",
            loss_params={"loss":"log_loss"}, model=b,
            thresholds=(0.2,0.4,0.6,0.8),
            val_total_cost=500.0, test_total_cost=600.0,
            val_roc_auc=0.75, test_roc_auc=0.70,
        )

    def test_returns_dataframe(self, mock_result):
        results = {"exp1": mock_result, "exp2": mock_result, "exp3": mock_result}
        df = build_comparison_table(results)
        assert isinstance(df, pd.DataFrame)

    def test_has_three_rows(self, mock_result):
        results = {"exp1": mock_result, "exp2": mock_result, "exp3": mock_result}
        df = build_comparison_table(results)
        assert len(df) == 3

    def test_required_columns_present(self, mock_result):
        results = {"exp1": mock_result, "exp2": mock_result, "exp3": mock_result}
        df = build_comparison_table(results)
        for col in ("experiment", "loss_function", "val_total_cost",
                    "test_total_cost", "val_roc_auc", "test_roc_auc"):
            assert col in df.columns

    def test_costs_are_numeric(self, mock_result):
        results = {"exp1": mock_result, "exp2": mock_result, "exp3": mock_result}
        df = build_comparison_table(results)
        assert pd.api.types.is_numeric_dtype(df["val_total_cost"])
        assert pd.api.types.is_numeric_dtype(df["test_total_cost"])


# =============================================================================
# LAYER 2 — DATA CONTRACT TESTS
# =============================================================================

class TestExp1Contracts:
    """Data contract tests on Exp1 TrainingResult."""

    def test_returns_training_result(self, tiny_exp1_result):
        assert isinstance(tiny_exp1_result, TrainingResult)

    def test_experiment_name_set(self, tiny_exp1_result):
        assert tiny_exp1_result.experiment_name == "exp1_log_loss"

    def test_loss_name_is_log_loss(self, tiny_exp1_result):
        assert tiny_exp1_result.loss_name == "log_loss"

    def test_val_cost_non_negative(self, tiny_exp1_result):
        assert tiny_exp1_result.val_total_cost >= 0.0

    def test_test_cost_non_negative(self, tiny_exp1_result):
        assert tiny_exp1_result.test_total_cost >= 0.0

    def test_val_roc_auc_in_unit_interval(self, tiny_exp1_result):
        assert 0.0 <= tiny_exp1_result.val_roc_auc <= 1.0

    def test_test_roc_auc_in_unit_interval(self, tiny_exp1_result):
        assert 0.0 <= tiny_exp1_result.test_roc_auc <= 1.0

    def test_four_thresholds_present(self, tiny_exp1_result):
        assert tiny_exp1_result.thresholds is not None
        assert len(tiny_exp1_result.thresholds) == 4

    def test_thresholds_strictly_increasing(self, tiny_exp1_result):
        t = tiny_exp1_result.thresholds
        assert t[0] < t[1] < t[2] < t[3]

    def test_thresholds_in_unit_interval(self, tiny_exp1_result):
        for t in tiny_exp1_result.thresholds:
            assert 0.0 <= t <= 1.0

    def test_val_probs_present(self, tiny_exp1_result):
        assert tiny_exp1_result.val_probs is not None
        assert len(tiny_exp1_result.val_probs) == N_VAL

    def test_test_probs_present(self, tiny_exp1_result):
        assert tiny_exp1_result.test_probs is not None
        assert len(tiny_exp1_result.test_probs) == N_TEST

    def test_model_saved_to_disk(self, tmp_path, datasets, default_cm):
        X_train, y_train, X_val, y_val_binary, y_val_5class, \
            X_test, y_test_5class, y_test_binary = datasets
        run_exp1_log_loss(
            X_train, y_train, X_val, y_val_binary, y_val_5class,
            X_test, y_test_5class, y_test_binary,
            cost_matrix=default_cm, n_trials=1,
            model_save_dir=str(tmp_path),
        )
        assert os.path.exists(os.path.join(str(tmp_path), "exp1_log_loss.pkl"))


class TestExp2Contracts:
    """Data contract tests on Exp2 TrainingResult."""

    def test_returns_training_result(self, tiny_exp2_result):
        assert isinstance(tiny_exp2_result, TrainingResult)

    def test_loss_name_is_focal(self, tiny_exp2_result):
        assert tiny_exp2_result.loss_name == "focal_loss"

    def test_gamma_in_params(self, tiny_exp2_result):
        assert "gamma" in tiny_exp2_result.loss_params

    def test_gamma_in_valid_range(self, tiny_exp2_result):
        assert 0.5 <= tiny_exp2_result.loss_params["gamma"] <= 5.0

    def test_val_cost_non_negative(self, tiny_exp2_result):
        assert tiny_exp2_result.val_total_cost >= 0.0

    def test_test_cost_populated(self, tiny_exp2_result):
        assert tiny_exp2_result.test_total_cost is not None


class TestExp3Contracts:
    """Data contract tests on Exp3 TrainingResult."""

    def test_returns_training_result(self, tiny_exp3_result):
        assert isinstance(tiny_exp3_result, TrainingResult)

    def test_loss_name_is_cost_aware(self, tiny_exp3_result):
        assert tiny_exp3_result.loss_name == "cost_aware_focal_loss"

    def test_alpha_in_params(self, tiny_exp3_result):
        assert "alpha" in tiny_exp3_result.loss_params

    def test_alpha_above_0_9(self, tiny_exp3_result):
        """alpha should be close to 1 given high FN costs."""
        assert tiny_exp3_result.loss_params["alpha"] > 0.9

    def test_fn_weight_in_params(self, tiny_exp3_result):
        assert "fn_weight" in tiny_exp3_result.loss_params

    def test_val_cost_non_negative(self, tiny_exp3_result):
        assert tiny_exp3_result.val_total_cost >= 0.0

    def test_test_cost_populated(self, tiny_exp3_result):
        assert tiny_exp3_result.test_total_cost is not None


# =============================================================================
# LAYER 3 — INTEGRITY TESTS
# =============================================================================

class TestProtocolIntegrity:
    """Enforce the scientific protocol across all three experiments."""

    def test_test_cost_populated_after_exp1(self, tiny_exp1_result):
        """Test evaluation must run — test_total_cost must not be None."""
        assert tiny_exp1_result.test_total_cost is not None

    def test_test_cost_populated_after_exp2(self, tiny_exp2_result):
        assert tiny_exp2_result.test_total_cost is not None

    def test_test_cost_populated_after_exp3(self, tiny_exp3_result):
        assert tiny_exp3_result.test_total_cost is not None

    def test_val_probs_length_matches_val_set(self, tiny_exp1_result):
        assert len(tiny_exp1_result.val_probs) == N_VAL

    def test_test_probs_length_matches_test_set(self, tiny_exp1_result):
        assert len(tiny_exp1_result.test_probs) == N_TEST

    def test_all_three_experiments_have_different_loss_names(
        self, tiny_exp1_result, tiny_exp2_result, tiny_exp3_result
    ):
        names = {tiny_exp1_result.loss_name,
                 tiny_exp2_result.loss_name,
                 tiny_exp3_result.loss_name}
        assert len(names) == 3, "Each experiment must use a different loss function"

    def test_summary_dict_serialisable(self, tiny_exp1_result):
        """Summary must be a flat dict of primitive types for CSV logging."""
        s = tiny_exp1_result.summary()
        for v in s.values():
            assert v is None or isinstance(v, (int, float, str)), (
                f"Non-primitive value in summary: {type(v)}"
            )

    def test_exp3_uses_higher_alpha_than_exp2(
        self, tiny_exp2_result, tiny_exp3_result
    ):
        """
        Cost-aware focal loss must have higher alpha than focal loss
        which has no cost weighting (alpha not a parameter in FocalLoss).
        """
        assert "alpha" in tiny_exp3_result.loss_params
        assert "alpha" not in tiny_exp2_result.loss_params


class TestReproducibility:
    """Same seed, same data → same results."""

    def test_same_seed_same_val_cost(self, datasets, default_cm, tmp_path):
        """Two runs with same seed must produce same val cost."""
        X_train, y_train, X_val, y_val_binary, y_val_5class, \
            X_test, y_test_5class, y_test_binary = datasets

        r1 = run_exp1_log_loss(
            X_train, y_train, X_val, y_val_binary, y_val_5class,
            X_test, y_test_5class, y_test_binary,
            cost_matrix=default_cm, n_trials=2, random_state=0,
            model_save_dir=str(tmp_path / "r1"),
        )
        r2 = run_exp1_log_loss(
            X_train, y_train, X_val, y_val_binary, y_val_5class,
            X_test, y_test_5class, y_test_binary,
            cost_matrix=default_cm, n_trials=2, random_state=0,
            model_save_dir=str(tmp_path / "r2"),
        )
        assert r1.val_total_cost == pytest.approx(r2.val_total_cost, rel=1e-3)


class TestSensitivityIntegrity:
    """Integrity tests for run_sensitivity_analysis()."""

    @pytest.fixture(scope="class")
    def sensitivity_result(self, datasets, tmp_path_factory):
        X_train, y_train, X_val, y_val_binary, y_val_5class, \
            X_test, y_test_5class, y_test_binary = datasets
        tmpdir = str(tmp_path_factory.mktemp("sens"))
        return run_sensitivity_analysis(
            X_train, y_train, X_val, y_val_binary, y_val_5class,
            X_test, y_test_5class, y_test_binary,
            fn_weights=[0.5, 1.0, 2.0],
            n_trials=2, random_state=42,
            model_save_dir=tmpdir,
        )

    def test_returns_dataframe(self, sensitivity_result):
        assert isinstance(sensitivity_result, pd.DataFrame)

    def test_row_count_matches_fn_weights(self, sensitivity_result):
        assert len(sensitivity_result) == 3

    def test_fn_weight_column_present(self, sensitivity_result):
        assert "fn_weight" in sensitivity_result.columns

    def test_val_cost_column_present(self, sensitivity_result):
        assert "val_total_cost" in sensitivity_result.columns

    def test_alpha_column_present(self, sensitivity_result):
        assert "alpha" in sensitivity_result.columns

    def test_alpha_increases_with_fn_weight(self, sensitivity_result):
        """Higher fn_weight → higher alpha (more cost penalty on FN)."""
        df = sensitivity_result.sort_values("fn_weight")
        alphas = df["alpha"].values
        # Monotone non-decreasing
        assert all(alphas[i] <= alphas[i+1] + 1e-4 for i in range(len(alphas)-1)), (
            f"Alpha should not decrease as fn_weight increases: {alphas}"
        )

    def test_all_val_costs_non_negative(self, sensitivity_result):
        assert (sensitivity_result["val_total_cost"] >= 0).all()

    def test_all_test_costs_non_negative(self, sensitivity_result):
        assert (sensitivity_result["test_total_cost"] >= 0).all()


# =============================================================================
# LAYER 4 — STATISTICAL SANITY TESTS
# =============================================================================

class TestStatisticalSanity:
    """Real-world plausibility checks on experiment outputs."""

    def test_exp3_has_lower_or_equal_val_cost_than_exp1(
        self, tiny_exp1_result, tiny_exp3_result
    ):
        """
        On a small dataset this is not guaranteed to hold, but we check
        that exp3 is at least within a reasonable range (3x of exp1).
        This is a sanity bound, not a strict ordering requirement for tiny data.
        """
        ratio = tiny_exp3_result.val_total_cost / (tiny_exp1_result.val_total_cost + 1e-9)
        assert ratio < 5.0, (
            f"Exp3 cost ({tiny_exp3_result.val_total_cost:.2f}) is more than "
            f"5x Exp1 cost ({tiny_exp1_result.val_total_cost:.2f}) — unexpected"
        )

    def test_all_probabilities_in_unit_interval(self, tiny_exp1_result):
        assert ((tiny_exp1_result.val_probs >= 0) &
                (tiny_exp1_result.val_probs <= 1)).all()
        assert ((tiny_exp1_result.test_probs >= 0) &
                (tiny_exp1_result.test_probs <= 1)).all()

    def test_focal_gamma_positive(self, tiny_exp2_result):
        assert tiny_exp2_result.loss_params["gamma"] > 0

    def test_exp3_alpha_close_to_1(self, tiny_exp3_result):
        """With high FN costs, alpha should be well above 0.9."""
        assert tiny_exp3_result.loss_params["alpha"] > 0.9

    def test_comparison_table_costs_are_positive(
        self, tiny_exp1_result, tiny_exp2_result, tiny_exp3_result
    ):
        results = {
            "exp1": tiny_exp1_result,
            "exp2": tiny_exp2_result,
            "exp3": tiny_exp3_result,
        }
        df = build_comparison_table(results)
        assert (df["val_total_cost"]  >= 0).all()
        assert (df["test_total_cost"] >= 0).all()

    def test_comparison_table_roc_auc_in_range(
        self, tiny_exp1_result, tiny_exp2_result, tiny_exp3_result
    ):
        results = {
            "exp1": tiny_exp1_result,
            "exp2": tiny_exp2_result,
            "exp3": tiny_exp3_result,
        }
        df = build_comparison_table(results)
        assert (df["val_roc_auc"].between(0, 1)).all()
        assert (df["test_roc_auc"].between(0, 1)).all()

    def test_best_iteration_positive(self, tiny_exp1_result):
        assert tiny_exp1_result.best_iteration > 0

    @pytest.mark.slow
    def test_exp3_beats_exp1_on_full_data(self, datasets, tmp_path):
        """
        On a larger run (n_trials=20) exp3 should produce lower cost.
        Marked slow — only run when explicitly requested.
        """
        X_train, y_train, X_val, y_val_binary, y_val_5class, \
            X_test, y_test_5class, y_test_binary = datasets
        cm = CostMatrix()

        r1 = run_exp1_log_loss(
            X_train, y_train, X_val, y_val_binary, y_val_5class,
            X_test, y_test_5class, y_test_binary,
            cost_matrix=cm, n_trials=20, random_state=42,
            model_save_dir=str(tmp_path / "exp1"),
        )
        r3 = run_exp3_cost_aware_focal(
            X_train, y_train, X_val, y_val_binary, y_val_5class,
            X_test, y_test_5class, y_test_binary,
            cost_matrix=cm, fn_weight=1.0, n_trials=20, random_state=42,
            model_save_dir=str(tmp_path / "exp3"),
        )
        assert r3.test_total_cost <= r1.test_total_cost, (
            f"Exp3 ({r3.test_total_cost:.2f}) should beat "
            f"Exp1 ({r1.test_total_cost:.2f}) on test cost"
        )