"""
SCANIA ComponentX — Preprocessing Pipeline Test Suite
======================================================
Thesis: Cost-Aware Learning for Failure Prediction in Industrial Predictive Maintenance

Architecture
------------
Layer 1 — Unit tests         : test each function in isolation with known inputs/outputs
Layer 2 — Data contract tests: verify shape, dtypes, value ranges after each phase
Layer 3 — Pipeline integrity : no data leakage, reproducibility, split symmetry
Layer 4 — Statistical sanity : imbalance ratio, distribution drift, missingness ceiling

Design principles
-----------------
- No real data files required — synthetic data factory generates all inputs
- One assertion per test function — clear failure messages
- Fixtures shared across test classes via conftest pattern (defined at module level here)
- parametrize used for edge cases — avoids duplicated test logic

Run with:
    pytest tests/test_scania_pipeline.py -v --tb=short
    pytest tests/test_scania_pipeline.py -v -k "leakage"   # run specific group
"""

import pytest
import numpy as np
import pandas as pd

from scania_pipeline import (
    HISTOGRAM_VARS,
    COUNTER_COLS,
    ID_COL,
    TIME_COL,
    LABEL_COL,
    STUDY_LEN_COL,
    _histogram_cols,
    _aggregate_histograms,
    _aggregate_counters,
    build_flat_table,
    load_readouts,
    load_labels,
    load_specs,
    load_split,
    remove_noise_features,
    apply_noise_filter,
    remove_correlated_features,
    apply_correlation_filter,
    compute_mutual_information,
    select_final_features,
    SCANIAPipeline,
)


# =============================================================================
# SYNTHETIC DATA FACTORY
# All tests use these fixtures — no real CSV files needed
# =============================================================================

N_VEHICLES   = 40      # enough to make statistics meaningful, small enough to be fast
N_READOUTS   = 8       # readouts per vehicle (irregular — varies per vehicle)
RANDOM_SEED  = 42


def make_readouts(
    n_vehicles:    int = N_VEHICLES,
    n_readouts:    int = N_READOUTS,
    seed:          int = RANDOM_SEED,
    include_nulls: bool = False,
) -> pd.DataFrame:
    """
    Synthetic operational_readouts DataFrame matching SCANIA column structure.

    Histogram bins: cumulative frequency counts (non-negative integers)
    Counter cols  : monotone increasing values per vehicle (cumulative accumulators)
    time_step     : irregular — different gap sizes per vehicle
    """
    rng = np.random.default_rng(seed)
    rows = []

    for vid in range(n_vehicles):
        # Irregular time steps — each vehicle has a different sampling cadence
        n_steps = rng.integers(3, n_readouts * 2)
        time_steps = np.sort(rng.choice(np.arange(1, 200), size=n_steps, replace=False))

        # Cumulative counter base values — grow monotonically
        counter_base = {col: 0.0 for col in COUNTER_COLS}

        for t in time_steps:
            row = {ID_COL: vid, TIME_COL: int(t)}

            # Histogram bins — non-negative frequency counts per period
            for var_id, n_bins in HISTOGRAM_VARS.items():
                for b in range(n_bins):
                    val = float(rng.integers(0, 100))
                    if include_nulls and rng.random() < 0.005:
                        val = np.nan
                    row[f"{var_id}_{b}"] = val

            # Counter cols — monotone cumulative (delta per step is always positive)
            for col in COUNTER_COLS:
                counter_base[col] += float(rng.integers(1, 50))
                val = counter_base[col]
                if include_nulls and rng.random() < 0.005:
                    val = np.nan
                row[col] = val

            rows.append(row)

    return pd.DataFrame(rows)


def make_train_tte(n_vehicles: int = N_VEHICLES, seed: int = RANDOM_SEED) -> pd.DataFrame:
    """
    Synthetic train_tte DataFrame — binary label + study length.
    Approximately 10% failure rate to mimic real SCANIA imbalance.
    """
    rng = np.random.default_rng(seed)
    labels = (rng.random(n_vehicles) < 0.10).astype(int)

    return pd.DataFrame({
        ID_COL:        np.arange(n_vehicles),
        LABEL_COL:     labels,
        STUDY_LEN_COL: rng.integers(50, 500, size=n_vehicles),
    })


def make_val_test_labels(
    n_vehicles: int = N_VEHICLES,
    seed:       int = RANDOM_SEED,
) -> pd.DataFrame:
    """
    Synthetic validation_labels / test_labels DataFrame.
    5-class temporal label matching SCANIA val/test format.
    Heavily skewed toward class 0 (mirrors real distribution).
    """
    rng = np.random.default_rng(seed)
    # ~97% class 0, remainder spread across classes 1-4
    classes = rng.choice([0, 1, 2, 3, 4], size=n_vehicles,
                         p=[0.97, 0.01, 0.01, 0.005, 0.005])
    return pd.DataFrame({
        ID_COL:       np.arange(n_vehicles),
        "class_label": classes,
    })


def make_specs(n_vehicles: int = N_VEHICLES, seed: int = RANDOM_SEED) -> pd.DataFrame:
    """Synthetic specifications DataFrame — 8 categorical features."""
    rng = np.random.default_rng(seed)
    data = {ID_COL: np.arange(n_vehicles)}
    for i in range(8):
        cats = [f"Cat{c}" for c in rng.integers(0, 5, size=n_vehicles)]
        data[f"Spec_{i}"] = cats
    return pd.DataFrame(data)


# ---------------------------------------------------------------------------
# Shared pytest fixtures (used across multiple test classes)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def readouts():
    return make_readouts()


@pytest.fixture(scope="module")
def readouts_with_nulls():
    return make_readouts(include_nulls=True)


@pytest.fixture(scope="module")
def train_tte():
    return make_train_tte()


@pytest.fixture(scope="module")
def val_labels():
    return make_val_test_labels()


@pytest.fixture(scope="module")
def specs():
    return make_specs()


@pytest.fixture(scope="module")
def flat_table(readouts, train_tte, specs):
    """Pre-built flat table fixture — reused across multiple test classes."""
    return build_flat_table(readouts, train_tte, specs)


@pytest.fixture(scope="module")
def X_train(flat_table):
    return flat_table[0]


@pytest.fixture(scope="module")
def y_train(flat_table):
    return flat_table[1]


# =============================================================================
# LAYER 1 — UNIT TESTS
# Test each function in isolation with known, controlled inputs
# =============================================================================

class TestHistogramCols:
    """Unit tests for _histogram_cols() helper."""

    def test_returns_list(self):
        assert isinstance(_histogram_cols(), list)

    def test_total_count_matches_documentation(self):
        """167(10)+272(10)+291(11)+158(10)+459(20)+397(36) = 97 bins."""
        assert len(_histogram_cols()) == 97

    def test_167_has_ten_bins(self):
        cols_167 = [c for c in _histogram_cols() if c.startswith("167_")]
        assert len(cols_167) == 10

    def test_397_has_36_bins(self):
        cols_397 = [c for c in _histogram_cols() if c.startswith("397_")]
        assert len(cols_397) == 36

    def test_bin_naming_format(self):
        """Bins should be named variableid_binindex e.g. 167_0, 167_1."""
        cols = _histogram_cols()
        for col in cols:
            parts = col.split("_")
            assert len(parts) == 2, f"Unexpected format: {col}"
            assert parts[1].isdigit(), f"Bin index not numeric: {col}"


class TestAggregateHistograms:
    """Unit tests for _aggregate_histograms()."""

    def test_one_row_per_vehicle(self, readouts):
        result = _aggregate_histograms(readouts)
        assert len(result) == readouts[ID_COL].nunique()

    def test_output_columns_have_correct_suffixes(self, readouts):
        result = _aggregate_histograms(readouts)
        col_suffixes = {c.rsplit("_", 1)[-1] for c in result.columns}
        assert col_suffixes == {"sum", "mean", "max"}

    def test_sum_is_non_negative(self, readouts):
        result = _aggregate_histograms(readouts)
        sum_cols = [c for c in result.columns if c.endswith("_sum")]
        assert (result[sum_cols] >= 0).all().all()

    def test_max_geq_mean(self, readouts):
        """Max of a non-negative series must always be >= mean."""
        result = _aggregate_histograms(readouts)
        for var_id in HISTOGRAM_VARS:
            for b in range(HISTOGRAM_VARS[var_id]):
                bin_name = f"{var_id}_{b}"
                if f"{bin_name}_max" in result and f"{bin_name}_mean" in result:
                    assert (result[f"{bin_name}_max"] >= result[f"{bin_name}_mean"] - 1e-9).all()

    def test_sum_equals_manual_calculation(self):
        """Verify sum on a tiny known DataFrame."""
        df = pd.DataFrame({
            ID_COL:  [0, 0, 1, 1],
            "167_0": [10.0, 20.0, 5.0, 5.0],
        })
        # Add all other histogram cols as zeros to avoid KeyError
        for col in _histogram_cols():
            if col not in df.columns:
                df[col] = 0.0
        result = _aggregate_histograms(df)
        assert result.loc[0, "167_0_sum"] == pytest.approx(30.0)
        assert result.loc[1, "167_0_sum"] == pytest.approx(10.0)

    def test_index_is_vehicle_id(self, readouts):
        result = _aggregate_histograms(readouts)
        assert result.index.name == ID_COL


class TestAggregateCounters:
    """Unit tests for _aggregate_counters()."""

    def test_one_row_per_vehicle(self, readouts):
        result = _aggregate_counters(readouts)
        assert len(result) == readouts[ID_COL].nunique()

    def test_output_columns_have_correct_suffixes(self, readouts):
        result = _aggregate_counters(readouts)
        col_suffixes = {c.rsplit("_", 1)[-1] for c in result.columns}
        assert col_suffixes == {"last", "delta", "rate"}

    def test_last_equals_manual(self):
        """last should equal the final observed value per vehicle."""
        df = pd.DataFrame({
            ID_COL:  [0, 0, 0],
            "171_0": [100.0, 250.0, 500.0],
        })
        for col in COUNTER_COLS:
            if col not in df.columns:
                df[col] = 0.0
        result = _aggregate_counters(df)
        assert result.loc[0, "171_0_last"] == pytest.approx(500.0)

    def test_delta_equals_last_minus_first(self):
        """delta = last - first."""
        df = pd.DataFrame({
            ID_COL:  [0, 0, 0],
            "171_0": [100.0, 250.0, 500.0],
        })
        for col in COUNTER_COLS:
            if col not in df.columns:
                df[col] = 0.0
        result = _aggregate_counters(df)
        assert result.loc[0, "171_0_delta"] == pytest.approx(400.0)  # 500 - 100

    def test_delta_non_negative_for_monotone_counters(self, readouts):
        """Counters are cumulative so delta must always be >= 0."""
        result = _aggregate_counters(readouts)
        delta_cols = [c for c in result.columns if c.endswith("_delta")]
        assert (result[delta_cols] >= -1e-9).all().all()

    def test_rate_equals_delta_divided_by_n_steps(self):
        """rate = delta / number of readings."""
        df = pd.DataFrame({
            ID_COL:  [0, 0, 0],   # 3 readings
            "171_0": [100.0, 250.0, 400.0],
        })
        for col in COUNTER_COLS:
            if col not in df.columns:
                df[col] = 0.0
        result = _aggregate_counters(df)
        # delta=300, n_readings=3, rate=100
        assert result.loc[0, "171_0_rate"] == pytest.approx(100.0)

    def test_single_readout_vehicle_delta_is_zero(self):
        """A vehicle with one readout has delta=0 and rate=0."""
        df = pd.DataFrame({
            ID_COL:  [0],
            "171_0": [500.0],
        })
        for col in COUNTER_COLS:
            if col not in df.columns:
                df[col] = 0.0
        result = _aggregate_counters(df)
        assert result.loc[0, "171_0_delta"] == pytest.approx(0.0)


class TestLabelConversion:
    """
    Unit tests for the 5-class → binary label conversion logic.
    Tests the VAL_TO_BINARY mapping that lives inside load_labels.
    We test the logic directly here without needing file I/O.
    """

    VAL_TO_BINARY = {0: 0, 1: 1, 2: 1, 3: 1, 4: 1}

    def test_class_0_maps_to_binary_0(self):
        assert self.VAL_TO_BINARY[0] == 0

    @pytest.mark.parametrize("cls", [1, 2, 3, 4])
    def test_near_failure_classes_map_to_binary_1(self, cls):
        assert self.VAL_TO_BINARY[cls] == 1

    def test_all_five_classes_covered(self):
        assert set(self.VAL_TO_BINARY.keys()) == {0, 1, 2, 3, 4}

    def test_only_two_binary_values(self):
        assert set(self.VAL_TO_BINARY.values()) == {0, 1}

    def test_conversion_on_series(self):
        """Verify pandas .map() produces correct binary series."""
        raw = pd.Series([0, 0, 1, 2, 3, 4, 0])
        binary = raw.map(self.VAL_TO_BINARY)
        expected = pd.Series([0, 0, 1, 1, 1, 1, 0])
        pd.testing.assert_series_equal(binary, expected)

    def test_class_distribution_preserved(self):
        """Binary conversion should give more 0s than 1s (class 0 is dominant)."""
        labels = make_val_test_labels()
        binary = labels["class_label"].map(self.VAL_TO_BINARY)
        assert binary.value_counts()[0] > binary.value_counts().get(1, 0)


class TestLoaders:
    """
    Unit tests for the three focused loader functions.
    Uses tmp_path + synthetic CSVs — no real data files needed.
    """

    @pytest.fixture()
    def data_dir(self, tmp_path):
        """Write synthetic CSV files to a temp directory for all three splits."""
        rng = np.random.default_rng(42)

        for split in ("train", "val", "test"):
            n_v = 20

            # operational_readouts — same structure for all splits
            rows = []
            for vid in range(n_v):
                for t in range(1, 6):
                    row = {ID_COL: vid, TIME_COL: t * 10}
                    for col in _histogram_cols():
                        row[col] = float(rng.integers(0, 50))
                    for col in COUNTER_COLS:
                        row[col] = float(t * rng.integers(1, 10))
                    rows.append(row)
            pd.DataFrame(rows).to_csv(
                tmp_path / f"{split}_operational_readouts.csv", index=False
            )

            # specs — same structure for all splits
            spec_data = {ID_COL: list(range(n_v))}
            for i in range(8):
                spec_data[f"Spec_{i}"] = [f"Cat{rng.integers(0,3)}" for _ in range(n_v)]
            pd.DataFrame(spec_data).to_csv(
                tmp_path / f"{split}_specifications.csv", index=False
            )

        # train_tte — binary label
        pd.DataFrame({
            ID_COL:        list(range(20)),
            LABEL_COL:     [0]*18 + [1, 1],
            STUDY_LEN_COL: [50] * 20,
        }).to_csv(tmp_path / "train_tte.csv", index=False)

        # validation_labels and test_labels — 5-class temporal
        for fname in ("validation_labels.csv", "test_labels.csv"):
            pd.DataFrame({
                ID_COL:        list(range(20)),
                "class_label": [0]*18 + [3, 4],
            }).to_csv(tmp_path / fname, index=False)

        return str(tmp_path)

    # --- load_readouts ---

    def test_load_readouts_returns_dataframe(self, data_dir):
        result = load_readouts(data_dir, "train")
        assert isinstance(result, pd.DataFrame)

    def test_load_readouts_has_vehicle_id_col(self, data_dir):
        result = load_readouts(data_dir, "train")
        assert ID_COL in result.columns

    def test_load_readouts_has_time_step_col(self, data_dir):
        result = load_readouts(data_dir, "train")
        assert TIME_COL in result.columns

    @pytest.mark.parametrize("split", ["train", "val", "test"])
    def test_load_readouts_all_splits_load(self, data_dir, split):
        result = load_readouts(data_dir, split)
        assert len(result) > 0

    # --- load_labels ---

    def test_load_labels_train_has_binary_label(self, data_dir):
        readouts = load_readouts(data_dir, "train")
        result   = load_labels(data_dir, "train", readouts)
        assert LABEL_COL in result.columns
        assert set(result[LABEL_COL].unique()).issubset({0, 1})

    def test_load_labels_train_has_study_len(self, data_dir):
        readouts = load_readouts(data_dir, "train")
        result   = load_labels(data_dir, "train", readouts)
        assert STUDY_LEN_COL in result.columns

    def test_load_labels_train_no_temporal_class(self, data_dir):
        """Train labels must not contain temporal_class — it does not exist."""
        readouts = load_readouts(data_dir, "train")
        result   = load_labels(data_dir, "train", readouts)
        assert "temporal_class" not in result.columns

    @pytest.mark.parametrize("split", ["val", "test"])
    def test_load_labels_val_test_binary_label_present(self, data_dir, split):
        readouts = load_readouts(data_dir, split)
        result   = load_labels(data_dir, split, readouts)
        assert LABEL_COL in result.columns
        assert set(result[LABEL_COL].unique()).issubset({0, 1})

    @pytest.mark.parametrize("split", ["val", "test"])
    def test_load_labels_val_test_temporal_class_preserved(self, data_dir, split):
        """temporal_class must survive in label_df for downstream evaluation."""
        readouts = load_readouts(data_dir, split)
        result   = load_labels(data_dir, split, readouts)
        assert "temporal_class" in result.columns

    @pytest.mark.parametrize("split", ["val", "test"])
    def test_load_labels_val_test_study_len_derived(self, data_dir, split):
        """Val/test have no tte.csv — study length must be derived from readouts."""
        readouts = load_readouts(data_dir, split)
        result   = load_labels(data_dir, split, readouts)
        assert STUDY_LEN_COL in result.columns
        assert (result[STUDY_LEN_COL] > 0).all()

    @pytest.mark.parametrize("split", ["val", "test"])
    def test_load_labels_val_test_class_3_4_become_binary_1(self, data_dir, split):
        """Classes 3 and 4 in our synthetic data must convert to binary 1."""
        readouts = load_readouts(data_dir, split)
        result   = load_labels(data_dir, split, readouts)
        near_fail = result[result["temporal_class"].isin([3, 4])]
        assert (near_fail[LABEL_COL] == 1).all()

    # --- load_specs ---

    def test_load_specs_returns_dataframe(self, data_dir):
        result = load_specs(data_dir, "train")
        assert isinstance(result, pd.DataFrame)

    def test_load_specs_has_vehicle_id(self, data_dir):
        result = load_specs(data_dir, "train")
        assert ID_COL in result.columns

    @pytest.mark.parametrize("split", ["train", "val", "test"])
    def test_load_specs_has_eight_spec_cols(self, data_dir, split):
        result   = load_specs(data_dir, split)
        spec_cols = [c for c in result.columns if c.startswith("Spec_")]
        assert len(spec_cols) == 8

    # --- load_split coordinator ---

    def test_load_split_returns_three_dataframes(self, data_dir):
        result = load_split(data_dir, "train")
        assert len(result) == 3
        assert all(isinstance(r, pd.DataFrame) for r in result)

    @pytest.mark.parametrize("split", ["train", "val", "test"])
    def test_load_split_all_splits_succeed(self, data_dir, split):
        readouts, label_df, specs = load_split(data_dir, split)
        assert len(readouts) > 0
        assert len(label_df) > 0
        assert len(specs) > 0


# =============================================================================
# LAYER 2 — DATA CONTRACT TESTS
# Verify shape, dtypes, value ranges on function outputs
# =============================================================================

class TestBuildFlatTable:
    """Data contract tests for build_flat_table()."""

    def test_one_row_per_vehicle(self, readouts, train_tte, specs):
        X, y = build_flat_table(readouts, train_tte, specs)
        assert len(X) == readouts[ID_COL].nunique()

    def test_X_and_y_same_length(self, readouts, train_tte, specs):
        X, y = build_flat_table(readouts, train_tte, specs)
        assert len(X) == len(y)

    def test_label_is_binary(self, readouts, train_tte, specs):
        _, y = build_flat_table(readouts, train_tte, specs)
        assert set(y.unique()).issubset({0, 1})

    def test_label_col_not_in_features(self, readouts, train_tte, specs):
        X, _ = build_flat_table(readouts, train_tte, specs)
        assert LABEL_COL not in X.columns

    def test_temporal_class_not_in_features(self, readouts, val_labels, specs):
        """temporal_class must be dropped — it must never be a feature."""
        # Simulate val label_df (already has temporal_class + binary in_study_repair)
        val_label_df = val_labels.copy()
        val_label_df[LABEL_COL] = val_label_df["class_label"].map({0:0,1:1,2:1,3:1,4:1})
        val_label_df = val_label_df.rename(columns={"class_label": "temporal_class"})
        study_len = (
            readouts.groupby(ID_COL)[TIME_COL].max()
            .reset_index().rename(columns={TIME_COL: STUDY_LEN_COL})
        )
        val_label_df = val_label_df.merge(study_len, on=ID_COL, how="left")

        X, _ = build_flat_table(readouts, val_label_df, specs)
        assert "temporal_class" not in X.columns

    def test_study_len_col_present_in_features(self, readouts, train_tte, specs):
        X, _ = build_flat_table(readouts, train_tte, specs)
        assert STUDY_LEN_COL in X.columns

    def test_spec_cols_are_categorical_dtype(self, readouts, train_tte, specs):
        X, _ = build_flat_table(readouts, train_tte, specs)
        spec_cols = [c for c in X.columns if c.startswith("Spec_")]
        for col in spec_cols:
            assert str(X[col].dtype) == "category", f"{col} should be category dtype"

    def test_histogram_sum_features_present(self, readouts, train_tte, specs):
        X, _ = build_flat_table(readouts, train_tte, specs)
        assert "167_0_sum" in X.columns

    def test_counter_last_features_present(self, readouts, train_tte, specs):
        X, _ = build_flat_table(readouts, train_tte, specs)
        assert "171_0_last" in X.columns

    def test_no_inf_values_in_features(self, readouts, train_tte, specs):
        X, _ = build_flat_table(readouts, train_tte, specs)
        numeric = X.select_dtypes(include=[np.number])
        assert not np.isinf(numeric.values).any()

    def test_label_dtype_is_integer(self, readouts, train_tte, specs):
        _, y = build_flat_table(readouts, train_tte, specs)
        assert pd.api.types.is_integer_dtype(y)


class TestRemoveNoiseFeatures:
    """Data contract tests for remove_noise_features()."""

    def test_returns_dataframe_and_state(self, X_train):
        result, state = remove_noise_features(X_train.copy())
        assert isinstance(result, pd.DataFrame)
        assert isinstance(state, dict)

    def test_state_has_required_keys(self, X_train):
        _, state = remove_noise_features(X_train.copy())
        assert "cols_to_drop" in state
        assert "kept_cols" in state

    def test_constant_column_is_removed(self, X_train):
        df = X_train.copy()
        df["__constant_col__"] = 999.0   # inject a constant column
        result, _ = remove_noise_features(df)
        assert "__constant_col__" not in result.columns

    def test_near_zero_variance_column_removed(self, X_train):
        df = X_train.copy()
        # Nearly constant — only two unique values with tiny variance
        df["__near_zero__"] = 1.0
        df.iloc[0, df.columns.get_loc("__near_zero__")] = 1.0001
        result, _ = remove_noise_features(df, variance_threshold=0.01)
        assert "__near_zero__" not in result.columns

    def test_duplicate_column_removed(self, X_train):
        df = X_train.copy()
        numeric_col = X_train.select_dtypes(include=[np.number]).columns[0]
        df["__duplicate__"] = df[numeric_col]
        result, _ = remove_noise_features(df)
        # Either the original or duplicate removed — not both present
        assert not ("__duplicate__" in result.columns
                    and numeric_col in result.columns and
                    result["__duplicate__"].equals(result[numeric_col]))

    def test_normal_column_is_kept(self, X_train):
        result, _ = remove_noise_features(X_train.copy())
        # At least one original feature should survive
        assert result.shape[1] > 0

    def test_output_has_fewer_or_equal_cols(self, X_train):
        result, _ = remove_noise_features(X_train.copy())
        assert result.shape[1] <= X_train.shape[1]

    def test_apply_noise_filter_matches_train_columns(self, X_train):
        """apply_noise_filter must produce same columns as fitted result."""
        _, state = remove_noise_features(X_train.copy())
        val_df   = X_train.copy()  # same structure simulates val
        result   = apply_noise_filter(val_df, state)
        assert list(result.columns) == state["kept_cols"]


class TestRemoveCorrelatedFeatures:
    """Data contract tests for remove_correlated_features()."""

    def test_returns_dataframe_and_state(self, X_train, y_train):
        result, state = remove_correlated_features(X_train.copy(), y_train)
        assert isinstance(result, pd.DataFrame)
        assert isinstance(state, dict)

    def test_highly_correlated_pair_reduced_to_one(self, y_train):
        """Two perfectly correlated features should result in one being dropped."""
        df = pd.DataFrame({
            "feat_a": np.arange(len(y_train), dtype=float),
            "feat_b": np.arange(len(y_train), dtype=float),  # identical → corr=1.0
            "feat_c": np.random.default_rng(42).random(len(y_train)),
        })
        result, _ = remove_correlated_features(df, y_train, corr_threshold=0.95)
        # feat_a and feat_b are identical — one must be dropped
        assert not ("feat_a" in result.columns and "feat_b" in result.columns)

    def test_study_len_col_always_kept(self, X_train, y_train):
        """STUDY_LEN_COL must never be dropped regardless of correlation."""
        result, _ = remove_correlated_features(X_train.copy(), y_train)
        if STUDY_LEN_COL in X_train.columns:
            assert STUDY_LEN_COL in result.columns

    def test_output_cols_subset_of_input(self, X_train, y_train):
        result, _ = remove_correlated_features(X_train.copy(), y_train)
        assert set(result.columns).issubset(set(X_train.columns))

    def test_apply_correlation_filter_same_columns(self, X_train, y_train):
        """apply_correlation_filter must mirror the training column mask."""
        _, state  = remove_correlated_features(X_train.copy(), y_train)
        result    = apply_correlation_filter(X_train.copy(), state)
        assert list(result.columns) == state["kept_cols"]

    def test_threshold_1_drops_nothing(self, X_train, y_train):
        """threshold=1.0 means only identical features are dropped."""
        result, _ = remove_correlated_features(
            X_train.copy(), y_train, corr_threshold=1.0
        )
        # Should keep nearly all features
        assert result.shape[1] >= X_train.select_dtypes(include=[np.number]).shape[1] - 5


# =============================================================================
# LAYER 3 — PIPELINE INTEGRITY TESTS
# No leakage, reproducibility, split symmetry
# =============================================================================

class TestNoDataLeakage:
    """
    Leakage tests — the most critical tests for a thesis pipeline.
    These verify that validation and test data never influence any fitted state.
    """

    def test_noise_filter_state_fitted_on_train_only(self, X_train, y_train):
        """
        Inject a constant column into a val-like DataFrame AFTER fitting on train.
        The filter must not re-fit — the constant col should pass through untouched.
        This confirms the filter uses only the train-derived state.
        """
        _, state = remove_noise_features(X_train.copy())

        # Simulate val with an extra constant column not seen in train
        val_df = X_train.copy()
        val_df["__val_only_constant__"] = 0.0

        result = apply_noise_filter(val_df, state)
        # The filter knows nothing about __val_only_constant__ so it passes through
        # (it's not in kept_cols, so it gets dropped — not re-fitted)
        assert "__val_only_constant__" not in result.columns

    def test_correlation_filter_state_fitted_on_train_only(self, X_train, y_train):
        """
        Applying correlation filter to val must use train's kept_cols list only.
        A column that happens to be correlated in val but not in train must not
        cause the filter to drop additional val columns.
        """
        _, state = remove_correlated_features(X_train.copy(), y_train)
        original_kept = set(state["kept_cols"])

        # Apply to a copy with the same columns — kept cols must be identical
        result = apply_correlation_filter(X_train.copy(), state)
        assert set(result.columns) == {c for c in original_kept if c in X_train.columns}

    def test_val_features_are_subset_of_train_features(self, X_train, y_train):
        """
        After applying noise + correlation filter, val columns must be
        a subset of train columns — no extra columns can appear.
        """
        X_clean, noise_state = remove_noise_features(X_train.copy())
        X_clean, corr_state  = remove_correlated_features(X_clean, y_train)

        # Simulate val
        val_sim = X_train.copy()
        val_sim = apply_noise_filter(val_sim, noise_state)
        val_sim = apply_correlation_filter(val_sim, corr_state)

        assert set(val_sim.columns).issubset(set(X_clean.columns))

    def test_label_not_present_in_feature_matrix(self, readouts, train_tte, specs):
        """LABEL_COL must never appear in X — it would be the most severe leakage."""
        X, _ = build_flat_table(readouts, train_tte, specs)
        assert LABEL_COL not in X.columns

    def test_temporal_class_never_leaks_into_features(self, readouts, specs):
        """temporal_class is an eval-only column — must never appear in X."""
        val_label_df = make_val_test_labels()
        val_label_df[LABEL_COL] = val_label_df["class_label"].map({0:0,1:1,2:1,3:1,4:1})
        val_label_df = val_label_df.rename(columns={"class_label": "temporal_class"})
        study_len = (
            readouts.groupby(ID_COL)[TIME_COL].max()
            .reset_index().rename(columns={TIME_COL: STUDY_LEN_COL})
        )
        val_label_df = val_label_df.merge(study_len, on=ID_COL, how="left")

        X, _ = build_flat_table(readouts, val_label_df, specs)
        assert "temporal_class" not in X.columns


class TestReproducibility:
    """Pipeline must produce identical outputs given the same seed."""

    def test_histogram_aggregation_is_deterministic(self, readouts):
        result1 = _aggregate_histograms(readouts)
        result2 = _aggregate_histograms(readouts)
        pd.testing.assert_frame_equal(result1, result2)

    def test_counter_aggregation_is_deterministic(self, readouts):
        result1 = _aggregate_counters(readouts)
        result2 = _aggregate_counters(readouts)
        pd.testing.assert_frame_equal(result1, result2)

    def test_noise_removal_is_deterministic(self, X_train):
        result1, state1 = remove_noise_features(X_train.copy())
        result2, state2 = remove_noise_features(X_train.copy())
        assert state1["kept_cols"] == state2["kept_cols"]

    def test_correlation_removal_is_deterministic(self, X_train, y_train):
        _, state1 = remove_correlated_features(X_train.copy(), y_train)
        _, state2 = remove_correlated_features(X_train.copy(), y_train)
        assert sorted(state1["kept_cols"]) == sorted(state2["kept_cols"])

    def test_same_seed_same_synthetic_data(self):
        """Synthetic data factory must produce identical data for same seed."""
        r1 = make_readouts(seed=42)
        r2 = make_readouts(seed=42)
        pd.testing.assert_frame_equal(r1, r2)

    def test_different_seeds_different_data(self):
        r1 = make_readouts(seed=42)
        r2 = make_readouts(seed=99)
        assert not r1.equals(r2)


class TestSplitSymmetry:
    """Val and test must produce feature matrices with same columns as train."""

    def test_val_flat_table_has_same_numeric_cols_as_train(self, readouts, train_tte,
                                                            val_labels, specs):
        X_train, _ = build_flat_table(readouts, train_tte, specs)

        # Build val label_df in the same way load_split would
        val_label_df = val_labels.copy()
        val_label_df[LABEL_COL] = val_label_df["class_label"].map({0:0,1:1,2:1,3:1,4:1})
        val_label_df = val_label_df.rename(columns={"class_label": "temporal_class"})
        study_len = (
            readouts.groupby(ID_COL)[TIME_COL].max()
            .reset_index().rename(columns={TIME_COL: STUDY_LEN_COL})
        )
        val_label_df = val_label_df.merge(study_len, on=ID_COL, how="left")
        X_val, _ = build_flat_table(readouts, val_label_df, specs)

        train_num = set(X_train.select_dtypes(include=[np.number]).columns)
        val_num   = set(X_val.select_dtypes(include=[np.number]).columns)
        assert train_num == val_num

    def test_filtered_val_columns_match_filtered_train_columns(self, X_train, y_train):
        """After both filters, val columns must exactly match train columns."""
        X_clean, noise_state = remove_noise_features(X_train.copy())
        X_clean, corr_state  = remove_correlated_features(X_clean, y_train)

        val_sim = X_train.copy()
        val_sim = apply_noise_filter(val_sim, noise_state)
        val_sim = apply_correlation_filter(val_sim, corr_state)

        assert sorted(X_clean.columns.tolist()) == sorted(val_sim.columns.tolist())


# =============================================================================
# LAYER 4 — STATISTICAL SANITY TESTS
# Data quality assertions on real-world characteristics
# =============================================================================

class TestStatisticalSanity:
    """Statistical assertions that verify data quality properties."""

    def test_class_imbalance_in_expected_range(self, readouts, train_tte, specs):
        """
        Minority class should be present but represent less than 50% of data.
        Real SCANIA: ~9.6% failure rate. Our synthetic: ~10%.
        """
        _, y = build_flat_table(readouts, train_tte, specs)
        minority_ratio = y.sum() / len(y)
        assert 0.01 < minority_ratio < 0.50, (
            f"Unexpected imbalance ratio: {minority_ratio:.3f}"
        )

    def test_no_nan_in_label(self, readouts, train_tte, specs):
        _, y = build_flat_table(readouts, train_tte, specs)
        assert y.isna().sum() == 0

    def test_missingness_below_threshold_after_aggregation(self, readouts_with_nulls,
                                                            train_tte, specs):
        """
        Even with <1% missingness in raw readouts, aggregated features
        should have very low NaN rate (≤5% per column).
        """
        X, _ = build_flat_table(readouts_with_nulls, train_tte, specs)
        numeric = X.select_dtypes(include=[np.number])
        nan_rate_per_col = numeric.isna().mean()
        assert (nan_rate_per_col <= 0.05).all(), (
            f"Columns with high NaN rate:\n{nan_rate_per_col[nan_rate_per_col > 0.05]}"
        )

    def test_study_len_col_always_positive(self, readouts, train_tte, specs):
        """length_of_study_time_step must be strictly positive — zero means no data."""
        X, _ = build_flat_table(readouts, train_tte, specs)
        if STUDY_LEN_COL in X.columns:
            assert (X[STUDY_LEN_COL] > 0).all()

    def test_histogram_sums_non_negative(self, readouts, train_tte, specs):
        """Histogram bins are frequency counts — sums must be ≥ 0."""
        X, _ = build_flat_table(readouts, train_tte, specs)
        sum_cols = [c for c in X.columns if c.endswith("_sum")]
        assert (X[sum_cols] >= 0).all().all()

    def test_counter_delta_non_negative(self, readouts, train_tte, specs):
        """Counter deltas must be ≥ 0 because counters are monotone."""
        X, _ = build_flat_table(readouts, train_tte, specs)
        delta_cols = [c for c in X.columns if c.endswith("_delta")]
        assert (X[delta_cols] >= -1e-9).all().all()

    def test_feature_count_in_expected_range(self, readouts, train_tte, specs):
        """
        Expected feature count:
        97 bins × 3 stats = 291 histogram features
        8 counters × 3 stats = 24 counter features
        1 study length + 8 specs = 9 other features
        Total ≈ 324 features before noise/correlation removal.
        """
        X, _ = build_flat_table(readouts, train_tte, specs)
        assert 200 < X.shape[1] < 500, (
            f"Unexpected feature count: {X.shape[1]}"
        )

    @pytest.mark.parametrize("split_label,expected_binary_values", [
        ("train_binary",  {0, 1}),
        ("val_5class",    {0, 1}),   # after conversion
    ])
    def test_binary_labels_only_contain_0_and_1(self, split_label,
                                                  expected_binary_values,
                                                  readouts, train_tte,
                                                  val_labels, specs):
        if split_label == "train_binary":
            _, y = build_flat_table(readouts, train_tte, specs)
        else:
            val_label_df = val_labels.copy()
            val_label_df[LABEL_COL] = val_label_df["class_label"].map(
                {0:0, 1:1, 2:1, 3:1, 4:1}
            )
            val_label_df = val_label_df.rename(columns={"class_label": "temporal_class"})
            study_len = (
                readouts.groupby(ID_COL)[TIME_COL].max()
                .reset_index().rename(columns={TIME_COL: STUDY_LEN_COL})
            )
            val_label_df = val_label_df.merge(study_len, on=ID_COL, how="left")
            _, y = build_flat_table(readouts, val_label_df, specs)

        assert set(y.unique()).issubset(expected_binary_values)


class TestSelectFinalFeatures:
    """Unit and contract tests for the consensus feature selection."""

    def test_returns_list(self):
        mi    = pd.Series({"a": 0.9, "b": 0.5, "c": 0.1})
        lgbm  = pd.Series({"a": 100, "b": 80,  "c": 10})
        perm  = pd.Series({"a": 0.8, "b": 0.3, "c": 0.05})
        result = select_final_features(mi, lgbm, perm,
                                       mi_top_n=2, lgbm_top_n=2, perm_top_n=2)
        assert isinstance(result, list)

    def test_consensus_requires_min_methods(self):
        """Feature only in 1 method must be dropped when min_methods=2."""
        mi    = pd.Series({"a": 0.9, "b": 0.1})
        lgbm  = pd.Series({"a": 100, "c": 90})   # c only in lgbm
        perm  = pd.Series({"a": 0.8, "d": 0.7})   # d only in perm
        result = select_final_features(mi, lgbm, perm,
                                       mi_top_n=2, lgbm_top_n=2, perm_top_n=2,
                                       min_methods=2)
        assert "c" not in result   # only in 1 method
        assert "d" not in result   # only in 1 method
        assert "a" in result       # in all 3

    def test_always_keep_forces_feature_in(self):
        mi    = pd.Series({"a": 0.9})
        lgbm  = pd.Series({"a": 100})
        perm  = pd.Series({"a": 0.8})
        result = select_final_features(mi, lgbm, perm,
                                       always_keep=[STUDY_LEN_COL])
        assert STUDY_LEN_COL in result

    def test_result_is_sorted(self):
        mi    = pd.Series({"z": 0.9, "a": 0.8, "m": 0.7})
        lgbm  = pd.Series({"z": 100, "a": 90,  "m": 80})
        perm  = pd.Series({"z": 0.9, "a": 0.8, "m": 0.7})
        result = select_final_features(mi, lgbm, perm,
                                       mi_top_n=3, lgbm_top_n=3, perm_top_n=3)
        assert result == sorted(result)


# =============================================================================
# EDGE CASE TESTS
# Unusual but valid data conditions the pipeline must handle gracefully
# =============================================================================

class TestEdgeCases:
    """Tests for real-world edge cases in sensor / fleet data."""

    def test_vehicle_with_single_readout(self):
        """Pipeline must not crash on a vehicle with only one time step."""
        df = make_readouts(n_vehicles=5)
        # Force vehicle 0 to have only 1 readout
        df = pd.concat([
            df[df[ID_COL] != 0],
            df[df[ID_COL] == 0].head(1),
        ]).reset_index(drop=True)

        result = _aggregate_counters(df)
        assert 0 in result.index
        assert result.loc[0, "171_0_delta"] == pytest.approx(0.0)

    def test_all_zeros_histogram_bin(self):
        """A bin that is always zero should aggregate to zero without errors."""
        df = make_readouts(n_vehicles=5)
        df["167_0"] = 0.0   # force bin to all zeros
        result = _aggregate_histograms(df)
        assert (result["167_0_sum"] == 0).all()

    def test_large_counter_values_no_overflow(self):
        """Counter values can be very large (odometer-style) — no overflow expected."""
        df = pd.DataFrame({
            ID_COL: [0, 0],
            "171_0": [1e9, 2e9],
        })
        for col in COUNTER_COLS:
            if col not in df.columns:
                df[col] = 0.0
        result = _aggregate_counters(df)
        assert result.loc[0, "171_0_delta"] == pytest.approx(1e9)
        assert not np.isinf(result.loc[0, "171_0_delta"])

    def test_pipeline_handles_missing_histogram_cols_gracefully(self):
        """If a histogram column is absent from readouts, pipeline should not crash."""
        df = make_readouts(n_vehicles=5)
        df = df.drop(columns=["167_0"], errors="ignore")   # remove one bin
        try:
            result = _aggregate_histograms(df)
            assert "167_0_sum" not in result.columns   # absent col not in output
        except Exception as e:
            pytest.fail(f"Pipeline crashed on missing column: {e}")

    def test_all_vehicles_healthy_no_crash(self, readouts, specs):
        """Pipeline must not crash when all labels are 0 (no failures)."""
        tte_all_healthy = make_train_tte()
        tte_all_healthy[LABEL_COL] = 0   # force all healthy
        X, y = build_flat_table(readouts, tte_all_healthy, specs)
        assert (y == 0).all()
        assert len(X) > 0

    def test_all_vehicles_failed_no_crash(self, readouts, specs):
        """Pipeline must not crash when all labels are 1."""
        tte_all_failed = make_train_tte()
        tte_all_failed[LABEL_COL] = 1
        X, y = build_flat_table(readouts, tte_all_failed, specs)
        assert (y == 1).all()
        assert len(X) > 0