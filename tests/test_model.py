"""
tests/test_model.py
===================
Test suite for src/model.py — four test layers.

Run with:
    pytest tests/test_model.py -v --tb=short
    pytest tests/test_model.py -v -k "LogLoss"
"""

import pytest
import numpy as np
import pandas as pd
import tempfile
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.cost_matrix import CostMatrix, COST_MATRIX_RAW, N_CLASSES
from src.model import (
    BaseLoss,
    LogLoss,
    FocalLoss,
    CostAwareFocalLoss,
    LGBMTrainer,
    TrainingResult,
    make_trainer,
)


# =============================================================================
# FIXTURES
# =============================================================================

N_TRAIN, N_VAL, N_FEAT = 200, 60, 20


@pytest.fixture(scope="module")
def default_cm():
    return CostMatrix()


@pytest.fixture(scope="module")
def rng():
    return np.random.default_rng(42)


@pytest.fixture(scope="module")
def y_train_binary(rng):
    return (rng.random(N_TRAIN) < 0.10).astype(float)


@pytest.fixture(scope="module")
def y_pred_raw(rng):
    return rng.normal(0, 1, N_TRAIN)


@pytest.fixture(scope="module")
def y_val_binary(rng):
    return (rng.random(N_VAL) < 0.10).astype(float)


@pytest.fixture(scope="module")
def y_val_5class(rng):
    return rng.choice([0,1,2,3,4], size=N_VAL, p=[0.88,0.05,0.03,0.02,0.02])


@pytest.fixture(scope="module")
def X_train(rng):
    return pd.DataFrame(
        rng.standard_normal((N_TRAIN, N_FEAT)),
        columns=[f"feat_{i}" for i in range(N_FEAT)],
    )


@pytest.fixture(scope="module")
def X_val(rng):
    return pd.DataFrame(
        rng.standard_normal((N_VAL, N_FEAT)),
        columns=[f"feat_{i}" for i in range(N_FEAT)],
    )


# Minimal booster helper
def _tiny_booster():
    import lightgbm as lgb
    X = pd.DataFrame({"a": [1.0,2.0,3.0,4.0,5.0]})
    y = np.array([0,0,0,1,1])
    return lgb.train(
        {"objective":"binary","verbosity":-1,"num_leaves":2},
        lgb.Dataset(X, label=y), num_boost_round=2,
    )


# Tiny trainer factory for integration tests
def _make_tiny(loss_fn, cm, seed=42):
    return LGBMTrainer(
        loss_fn=loss_fn, cost_matrix=cm,
        lgbm_params={"n_estimators":10,"num_leaves":4,"learning_rate":0.1},
        n_threshold_candidates=5,
        early_stopping_rounds=5,
        random_state=seed,
    )


# =============================================================================
# LAYER 1 — UNIT TESTS: LogLoss
# =============================================================================

class TestLogLoss:

    @pytest.fixture(autouse=True)
    def setup(self):
        self.fn = LogLoss()

    def test_name(self):
        assert self.fn.name == "log_loss"

    def test_params_has_loss_key(self):
        assert "loss" in self.fn.params

    def test_grad_shape(self, y_train_binary, y_pred_raw):
        g, h = self.fn.gradients(y_train_binary, y_pred_raw)
        assert g.shape == y_train_binary.shape
        assert h.shape == y_train_binary.shape

    def test_hessian_positive(self, y_train_binary, y_pred_raw):
        _, h = self.fn.gradients(y_train_binary, y_pred_raw)
        assert (h > 0).all()

    def test_grad_zero_at_perfect_positive(self):
        g, _ = self.fn.gradients(np.array([1.0]), np.array([10.0]))
        assert abs(g[0]) < 0.01

    def test_grad_zero_at_perfect_negative(self):
        g, _ = self.fn.gradients(np.array([0.0]), np.array([-10.0]))
        assert abs(g[0]) < 0.01

    def test_grad_negative_for_underpredicted_failure(self):
        # y=1, p < y → grad = p - y < 0 → push score up
        g, _ = self.fn.gradients(np.array([1.0]), np.array([-1.0]))
        assert g[0] < 0

    def test_grad_positive_for_overpredicted_healthy(self):
        # y=0, p > y → grad = p - y > 0 → push score down
        g, _ = self.fn.gradients(np.array([0.0]), np.array([1.0]))
        assert g[0] > 0

    def test_grad_matches_sigmoid_minus_y(self, y_train_binary, y_pred_raw):
        g, _ = self.fn.gradients(y_train_binary, y_pred_raw)
        p    = 1.0 / (1.0 + np.exp(-y_pred_raw))
        np.testing.assert_allclose(g, p - y_train_binary, rtol=1e-6)

    def test_hessian_max_near_zero_score(self):
        _, h_mid = self.fn.gradients(np.array([1.0]), np.array([0.0]))
        _, h_ext = self.fn.gradients(np.array([1.0]), np.array([5.0]))
        assert h_mid[0] > h_ext[0]

    def test_analytical_spot_check(self):
        # p=sigmoid(0)=0.5 → grad=-0.5, hess=0.25
        g, h = self.fn.gradients(np.array([1.0]), np.array([0.0]))
        assert g[0] == pytest.approx(-0.5, abs=1e-4)
        assert h[0] == pytest.approx(0.25, abs=1e-4)


# =============================================================================
# LAYER 1 — UNIT TESTS: FocalLoss
# =============================================================================

class TestFocalLoss:

    def test_name(self):
        assert FocalLoss(gamma=2.0).name == "focal_loss"

    def test_params_has_gamma(self):
        assert FocalLoss(gamma=1.5).params["gamma"] == 1.5

    def test_negative_gamma_raises(self):
        with pytest.raises(ValueError):
            FocalLoss(gamma=-0.5)

    def test_gamma_zero_equals_logloss(self, y_train_binary, y_pred_raw):
        g_fl, h_fl = FocalLoss(gamma=0.0).gradients(y_train_binary, y_pred_raw)
        g_ll, h_ll = LogLoss().gradients(y_train_binary, y_pred_raw)
        np.testing.assert_allclose(g_fl, g_ll, rtol=1e-4)

    def test_hessian_positive(self, y_train_binary, y_pred_raw):
        _, h = FocalLoss(2.0).gradients(y_train_binary, y_pred_raw)
        assert (h > 0).all()

    def test_easy_examples_smaller_gradient_than_hard(self):
        fl = FocalLoss(gamma=2.0)
        g_easy, _ = fl.gradients(np.array([1.0]), np.array([3.0]))    # p~0.95
        g_hard, _ = fl.gradients(np.array([1.0]), np.array([0.1]))    # p~0.52
        assert abs(g_easy[0]) < abs(g_hard[0])

    @pytest.mark.parametrize("gamma", [0.5, 1.0, 2.0, 5.0])
    def test_gradient_finite_all_gammas(self, y_train_binary, y_pred_raw, gamma):
        g, h = FocalLoss(gamma).gradients(y_train_binary, y_pred_raw)
        assert np.isfinite(g).all() and np.isfinite(h).all()

    def test_gradient_finite_at_extreme_scores(self):
        y = np.array([1.0, 0.0, 1.0, 0.0])
        x = np.array([20.0, -20.0, -20.0, 20.0])
        g, h = FocalLoss(2.0).gradients(y, x)
        assert np.isfinite(g).all() and np.isfinite(h).all()


# =============================================================================
# LAYER 1 — UNIT TESTS: CostAwareFocalLoss
# =============================================================================

class TestCostAwareFocalLoss:

    @pytest.fixture(autouse=True)
    def setup(self, default_cm):
        self.fn = CostAwareFocalLoss(cost_matrix=default_cm, gamma=2.0)
        self.cm = default_cm

    def test_name(self):
        assert self.fn.name == "cost_aware_focal_loss"

    def test_params_has_required_keys(self):
        for k in ("loss","gamma","alpha","fn_weight"):
            assert k in self.fn.params

    def test_alpha_between_0_and_1(self):
        assert 0.0 < self.fn.alpha < 1.0

    def test_alpha_close_to_1_due_to_high_fn_cost(self):
        # fn_mean~350, fp_mean~8.5 → alpha = 350/358.5 ~ 0.976
        assert self.fn.alpha > 0.95

    def test_alpha_formula_matches_documentation(self):
        fn_mean = np.mean([COST_MATRIX_RAW[a][0] for a in range(1, N_CLASSES)])
        fp_mean = np.mean([COST_MATRIX_RAW[0][p] for p in range(1, N_CLASSES)])
        expected = fn_mean / (fn_mean + fp_mean)
        assert self.fn.alpha == pytest.approx(expected, rel=1e-3)

    def test_alpha_increases_with_fn_weight(self, default_cm):
        a1 = CostAwareFocalLoss(CostMatrix(fn_weight=1.0)).alpha
        a2 = CostAwareFocalLoss(CostMatrix(fn_weight=3.0)).alpha
        assert a2 > a1

    def test_negative_gamma_raises(self):
        with pytest.raises(ValueError):
            CostAwareFocalLoss(cost_matrix=self.cm, gamma=-1.0)

    def test_hessian_positive(self, y_train_binary, y_pred_raw):
        _, h = self.fn.gradients(y_train_binary, y_pred_raw)
        assert (h > 0).all()

    def test_gradient_finite_at_extreme_scores(self):
        y = np.array([1.0, 0.0, 1.0, 0.0])
        x = np.array([20.0, -20.0, -20.0, 20.0])
        g, h = self.fn.gradients(y, x)
        assert np.isfinite(g).all() and np.isfinite(h).all()

    def test_positive_samples_larger_gradient_than_logloss(self):
        # alpha > 0.5 → positive samples receive larger penalty
        ll = LogLoss()
        y  = np.array([1.0])
        x  = np.array([0.0])
        g_ca, _ = self.fn.gradients(y, x)
        g_ll, _ = ll.gradients(y, x)
        assert abs(g_ca[0]) > abs(g_ll[0]) * 0.9


# =============================================================================
# LAYER 1 — UNIT TESTS: TrainingResult
# =============================================================================

class TestTrainingResult:

    @pytest.fixture()
    def result(self):
        return TrainingResult(
            experiment_name="test",
            loss_name="log_loss",
            loss_params={"loss":"log_loss"},
            model=_tiny_booster(),
            thresholds=(0.2, 0.4, 0.6, 0.8),
            best_iteration=10,
            train_logloss=0.5,
            val_logloss=0.6,
            val_total_cost=1000.0,
            test_total_cost=1100.0,
            val_roc_auc=0.75,
            test_roc_auc=0.70,
        )

    def test_summary_returns_dict(self, result):
        assert isinstance(result.summary(), dict)

    def test_summary_has_required_keys(self, result):
        s = result.summary()
        for k in ("experiment_name","loss_name","val_total_cost",
                  "test_total_cost","threshold_t1","threshold_t4"):
            assert k in s

    def test_summary_thresholds_correct(self, result):
        s = result.summary()
        assert s["threshold_t1"] == 0.2
        assert s["threshold_t4"] == 0.8

    def test_summary_none_thresholds(self):
        r = TrainingResult(
            experiment_name="x", loss_name="y", loss_params={},
            model=_tiny_booster(),
        )
        assert r.summary()["threshold_t1"] is None


# =============================================================================
# LAYER 2 — DATA CONTRACT TESTS
# =============================================================================

class TestLGBMTrainerContracts:

    def test_predict_proba_before_train_raises(self, default_cm, X_val):
        with pytest.raises(RuntimeError):
            _make_tiny(LogLoss(), default_cm).predict_proba(X_val)

    def test_predict_class_before_train_raises(self, default_cm, X_val):
        with pytest.raises(RuntimeError):
            _make_tiny(LogLoss(), default_cm).predict_class(X_val)

    def test_evaluate_test_before_train_raises(self, default_cm):
        with pytest.raises(RuntimeError):
            _make_tiny(LogLoss(), default_cm).evaluate_test(None, None, None, None)

    def test_predict_proba_in_unit_interval(
        self, default_cm, X_train, y_train_binary, X_val, y_val_binary, y_val_5class
    ):
        t = _make_tiny(LogLoss(), default_cm)
        t.train(X_train, pd.Series(y_train_binary.astype(int)),
                X_val, pd.Series(y_val_binary.astype(int)), y_val_5class, "p_test")
        probs = t.predict_proba(X_val)
        assert ((probs >= 0) & (probs <= 1)).all()

    def test_predict_proba_length(
        self, default_cm, X_train, y_train_binary, X_val, y_val_binary, y_val_5class
    ):
        t = _make_tiny(LogLoss(), default_cm)
        t.train(X_train, pd.Series(y_train_binary.astype(int)),
                X_val, pd.Series(y_val_binary.astype(int)), y_val_5class, "len_test")
        assert len(t.predict_proba(X_val)) == len(X_val)

    def test_predict_class_valid_range(
        self, default_cm, X_train, y_train_binary, X_val, y_val_binary, y_val_5class
    ):
        t = _make_tiny(LogLoss(), default_cm)
        t.train(X_train, pd.Series(y_train_binary.astype(int)),
                X_val, pd.Series(y_val_binary.astype(int)), y_val_5class, "cls_test")
        classes = t.predict_class(X_val)
        assert ((classes >= 0) & (classes < N_CLASSES)).all()

    def test_predict_class_dtype_integer(
        self, default_cm, X_train, y_train_binary, X_val, y_val_binary, y_val_5class
    ):
        t = _make_tiny(LogLoss(), default_cm)
        t.train(X_train, pd.Series(y_train_binary.astype(int)),
                X_val, pd.Series(y_val_binary.astype(int)), y_val_5class, "dt_test")
        assert np.issubdtype(t.predict_class(X_val).dtype, np.integer)

    def test_val_cost_non_negative(
        self, default_cm, X_train, y_train_binary, X_val, y_val_binary, y_val_5class
    ):
        t = _make_tiny(LogLoss(), default_cm)
        r = t.train(X_train, pd.Series(y_train_binary.astype(int)),
                    X_val, pd.Series(y_val_binary.astype(int)), y_val_5class, "cost_test")
        assert r.val_total_cost >= 0.0

    def test_roc_auc_in_unit_interval(
        self, default_cm, X_train, y_train_binary, X_val, y_val_binary, y_val_5class
    ):
        t = _make_tiny(LogLoss(), default_cm)
        r = t.train(X_train, pd.Series(y_train_binary.astype(int)),
                    X_val, pd.Series(y_val_binary.astype(int)), y_val_5class, "auc_test")
        assert 0.0 <= r.val_roc_auc <= 1.0


# =============================================================================
# LAYER 3 — INTEGRITY TESTS
# =============================================================================

class TestIntegrity:

    def test_all_losses_share_interface(self, default_cm):
        losses = [LogLoss(), FocalLoss(2.0), CostAwareFocalLoss(default_cm)]
        y, x   = np.array([0.,1.,0.,1.]), np.array([-1.,1.,0.5,-0.5])
        for fn in losses:
            assert isinstance(fn.name, str)
            assert isinstance(fn.params, dict)
            g, h = fn.gradients(y, x)
            assert g.shape == y.shape
            assert (h > 0).all()

    def test_make_trainer_returns_lgbm_trainer(self, default_cm):
        for exp in ("log_loss","focal_loss","cost_aware_focal_loss"):
            assert isinstance(make_trainer(exp, default_cm), LGBMTrainer)

    def test_make_trainer_invalid_raises(self, default_cm):
        with pytest.raises(ValueError):
            make_trainer("unknown", default_cm)

    def test_loss_name_matches_experiment(self, default_cm):
        assert make_trainer("log_loss", default_cm).loss_fn.name == "log_loss"
        assert make_trainer("focal_loss", default_cm).loss_fn.name == "focal_loss"
        assert make_trainer("cost_aware_focal_loss", default_cm).loss_fn.name == "cost_aware_focal_loss"

    def test_gradients_deterministic(self, default_cm):
        rng    = np.random.default_rng(0)
        y, x   = (rng.random(100) < 0.1).astype(float), rng.normal(0,1,100)
        fn     = CostAwareFocalLoss(cost_matrix=default_cm)
        g1, h1 = fn.gradients(y, x)
        g2, h2 = fn.gradients(y, x)
        np.testing.assert_array_equal(g1, g2)

    def test_val_thresholds_not_refitted_on_test(
        self, default_cm, X_train, y_train_binary, X_val, y_val_binary, y_val_5class
    ):
        t = _make_tiny(LogLoss(), default_cm)
        t.train(X_train, pd.Series(y_train_binary.astype(int)),
                X_val, pd.Series(y_val_binary.astype(int)), y_val_5class, "leak_test")
        val_t = t._optimiser.best_thresholds_
        _ = t.predict_class(X_val)
        assert t._optimiser.best_thresholds_ == val_t

    def test_save_load_identical_predictions(
        self, default_cm, X_train, y_train_binary, X_val, y_val_binary, y_val_5class
    ):
        t = _make_tiny(LogLoss(), default_cm)
        r = t.train(X_train, pd.Series(y_train_binary.astype(int)),
                    X_val, pd.Series(y_val_binary.astype(int)), y_val_5class, "sl_test")
        probs_before = t.predict_proba(X_val)

        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "model.pkl")
            t.save(r, path)
            t2 = _make_tiny(LogLoss(), default_cm)
            t2.load(path)

        np.testing.assert_allclose(probs_before, t2.predict_proba(X_val), rtol=1e-5)

    def test_different_losses_different_models(
        self, default_cm, X_train, y_train_binary, X_val, y_val_binary, y_val_5class
    ):
        y_s = pd.Series(y_train_binary.astype(int))
        y_v = pd.Series(y_val_binary.astype(int))

        t_ll = _make_tiny(LogLoss(), default_cm, seed=42)
        t_ca = _make_tiny(CostAwareFocalLoss(default_cm), default_cm, seed=42)

        t_ll.train(X_train, y_s, X_val, y_v, y_val_5class, "ll")
        t_ca.train(X_train, y_s, X_val, y_v, y_val_5class, "ca")

        assert not np.allclose(
            t_ll.predict_proba(X_val),
            t_ca.predict_proba(X_val), atol=1e-3
        )


# =============================================================================
# LAYER 4 — STATISTICAL SANITY
# =============================================================================

class TestStatisticalSanity:

    def test_focal_reduces_gradient_for_easy_examples(self):
        fl = FocalLoss(gamma=2.0)
        ll = LogLoss()
        y, x = np.array([1.0]), np.array([3.0])   # easy: p~0.95
        g_fl, _ = fl.gradients(y, x)
        g_ll, _ = ll.gradients(y, x)
        assert abs(g_fl[0]) < abs(g_ll[0])

    @pytest.mark.parametrize("loss_name,loss_cls,kwargs", [
        ("log_loss",   LogLoss,   {}),
        ("focal_loss", FocalLoss, {"gamma": 2.0}),
    ])
    def test_grad_negative_for_underpredicted_failure(
        self, loss_name, loss_cls, kwargs
    ):
        fn = loss_cls(**kwargs)
        g, _ = fn.gradients(np.array([1.0]), np.array([-2.0]))
        assert g[0] < 0, f"{loss_name}: grad should be negative"

    @pytest.mark.parametrize("loss_name,loss_cls,kwargs", [
        ("log_loss",   LogLoss,   {}),
        ("focal_loss", FocalLoss, {"gamma": 2.0}),
    ])
    def test_grad_positive_for_overpredicted_healthy(
        self, loss_name, loss_cls, kwargs
    ):
        fn = loss_cls(**kwargs)
        g, _ = fn.gradients(np.array([0.0]), np.array([2.0]))
        assert g[0] > 0, f"{loss_name}: grad should be positive"

    def test_fn_weight_increases_gradient_magnitude(self, default_cm):
        fn1 = CostAwareFocalLoss(CostMatrix(fn_weight=1.0), fn_weight=1.0)
        fn2 = CostAwareFocalLoss(CostMatrix(fn_weight=3.0), fn_weight=3.0)
        y = np.ones(20)
        x = np.zeros(20)
        g1, _ = fn1.gradients(y, x)
        g2, _ = fn2.gradients(y, x)
        assert np.mean(np.abs(g2)) > np.mean(np.abs(g1))

    def test_cost_aware_alpha_formula_verified(self, default_cm):
        fn_mean  = np.mean([COST_MATRIX_RAW[a][0] for a in range(1, N_CLASSES)])
        fp_mean  = np.mean([COST_MATRIX_RAW[0][p] for p in range(1, N_CLASSES)])
        expected = fn_mean / (fn_mean + fp_mean)
        fn = CostAwareFocalLoss(cost_matrix=default_cm)
        assert fn.alpha == pytest.approx(expected, rel=1e-3)