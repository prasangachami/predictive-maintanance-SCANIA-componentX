"""
SCANIA ComponentX — Preprocessing & Feature Selection Pipeline
==============================================================
Thesis: Cost-Aware Learning for Failure Prediction in Industrial Predictive Maintenance

Pipeline stages:
    Phase 1 : Load raw source files
    Phase 2 : Feature extraction  (aggregation → merge → flat table)
    Phase 3 : Layer 1  — remove noise      (constant / near-zero / duplicate cols)
              Layer 2  — remove redundancy (high correlation)
              Layer 3  — predictive power  (mutual info + LGBM importance + permutation)

Usage:
    from scania_pipeline import SCANIAPipeline

    pipeline = SCANIAPipeline(data_dir="data/")
    X_train, y_train = pipeline.fit_transform(split="train")
    X_val,   y_val   = pipeline.transform(split="val")
    X_test,  y_test  = pipeline.transform(split="test")

    pipeline.save("pipeline_state.pkl")           # save fitted state
    pipeline.load("pipeline_state.pkl")           # reload for inference
"""

import os
import sys
import pickle
import warnings
import numpy as np
import pandas as pd
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.inspection    import permutation_importance
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics        import make_scorer

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Constants — column definitions from SCANIA dataset documentation
# ---------------------------------------------------------------------------

HISTOGRAM_VARS = {
    "167": 10,
    "272": 10,
    "291": 11,
    "158": 10,
    "459": 20,
    "397": 36,
}

COUNTER_COLS = [
    "171_0", "666_0", "427_0", "837_0",
    "309_0", "835_0", "370_0", "100_0",
]

ID_COL         = "vehicle_id"
TIME_COL       = "time_step"
LABEL_COL      = "in_study_repair"
STUDY_LEN_COL  = "length_of_study_time_step"


# ---------------------------------------------------------------------------
# Helper — build histogram column names from HISTOGRAM_VARS dict
# ---------------------------------------------------------------------------

def _histogram_cols():
    """Return list of all histogram bin column names e.g. ['167_0','167_1',...]."""
    cols = []
    for var_id, n_bins in HISTOGRAM_VARS.items():
        cols += [f"{var_id}_{i}" for i in range(n_bins)]
    return cols


# ---------------------------------------------------------------------------
# Helper — validate readouts columns and detect actual ID/time column names
# ---------------------------------------------------------------------------

def _validate_readouts(readouts: pd.DataFrame) -> None:
    """
    Validate that readouts contains the expected ID and time columns.
    Raises a clear ValueError if they are missing, showing what IS present.
    This catches cases where the CSV uses different column names than expected.
    """
    missing = []
    if ID_COL not in readouts.columns:
        missing.append(f"ID column '{ID_COL}'")
    if TIME_COL not in readouts.columns:
        missing.append(f"time column '{TIME_COL}'")

    if missing:
        raise ValueError(
            f"readouts DataFrame is missing: {missing}\n"
            f"Actual columns (first 10): {readouts.columns.tolist()[:10]}\n"
            f"Check that ID_COL='{ID_COL}' and TIME_COL='{TIME_COL}' "
            f"match your CSV column names."
        )


# ---------------------------------------------------------------------------
# Phase 1 — Load raw source files
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Three focused loaders — one per file type
# Each function has a single job and is independently testable
# ---------------------------------------------------------------------------

def load_readouts(data_dir: str, split: str) -> pd.DataFrame:
    """
    Load <split>_operational_readouts.csv.

    Same structure for all three splits:
        vehicle_id, time_step, <105 sensor feature columns>

    Returns
    -------
    pd.DataFrame — 1 row per readout event, multiple rows per vehicle
    """
    split = "validation" if split=="val" else split

    path     = os.path.join(data_dir, f"{split}_operational_readouts.csv")
    readouts = pd.read_csv(path)

    if ID_COL not in readouts.columns:
        raise ValueError(
            f"Expected ID column '{ID_COL}' not found in {path}\n"
            f"Actual columns: {readouts.columns.tolist()}\n"
            f"If your CSV uses a different column name, update ID_COL in scania_pipeline.py"
        )
    if TIME_COL not in readouts.columns:
        raise ValueError(
            f"Expected time column '{TIME_COL}' not found in {path}\n"
            f"Actual columns: {readouts.columns.tolist()}\n"
            f"If your CSV uses a different column name, update TIME_COL in scania_pipeline.py"
        )

    print(f"    readouts : {readouts.shape[0]:,} rows × {readouts.shape[1]} cols"
          f"  ({readouts[ID_COL].nunique():,} unique vehicles)")
    return readouts


def load_labels(data_dir: str, split: str, readouts: pd.DataFrame) -> pd.DataFrame:
    """
    Load and normalise labels for the given split.

    Train  (train_tte.csv)
        Columns already present : in_study_repair, length_of_study_time_step
        Binary label {0, 1} — no conversion needed.

    Val    (validation_labels.csv)
    Test   (test_labels.csv)
        Column present : class_label  (5-class temporal, 0–4)
            0 = > 48 steps before failure  (healthy)
            1 = 12–24 steps before failure
            2 = 24–48 steps before failure
            3 =  6–12 steps before failure
            4 =  0– 6 steps before failure
        Conversion applied : classes 1–4 → binary 1, class 0 → binary 0
        length_of_study_time_step derived from last time_step in readouts.

    Returns
    -------
    pd.DataFrame — 1 row per vehicle, always contains:
        vehicle_id
        in_study_repair            (binary {0, 1})
        length_of_study_time_step
        temporal_class             (original 5-class — val/test only, for evaluation)
    """
    TEMPORAL_CLASS_MAP = {0: 0, 1: 1, 2: 1, 3: 1, 4: 1}

    if split == "train":
        path     = os.path.join(data_dir, "train_tte.csv")
        label_df = pd.read_csv(path)
        if ID_COL not in label_df.columns:
            label_df[ID_COL] = label_df.index

    else:
        filename = "validation_labels.csv" if split == "val" else "test_labels.csv"
        path     = os.path.join(data_dir, filename)
        label_df = pd.read_csv(path)

        if ID_COL not in label_df.columns:
            label_df[ID_COL] = label_df.index

        # Convert 5-class → binary and preserve original for evaluation
        label_df = label_df.rename(columns={"class_label": "temporal_class"})
        label_df[LABEL_COL] = label_df["temporal_class"].map(TEMPORAL_CLASS_MAP)

        # Derive study length from readouts (no tte.csv available for val/test)
        study_len = (
            readouts.groupby(ID_COL)[TIME_COL]
            .max()
            .reset_index()
            .rename(columns={TIME_COL: STUDY_LEN_COL})
        )
        label_df = label_df.merge(study_len, on=ID_COL, how="left")

        print(f"    [{split}] 5-class → binary:  "
              f"{(label_df[LABEL_COL]==0).sum()} healthy  "
              f"{(label_df[LABEL_COL]==1).sum()} near-failure")

    print(f"    labels   : {label_df.shape[0]:,} rows"
          f"  | dist: {label_df[LABEL_COL].value_counts().to_dict()}")
    return label_df


def load_specs(data_dir: str, split: str) -> pd.DataFrame:
    """
    Load <split>_specifications.csv.

    Same structure for all three splits:
        vehicle_id, Spec_0 … Spec_7  (anonymised categorical features)

    Returns
    -------
    pd.DataFrame — 1 row per vehicle, 8 categorical spec columns
    """
    split = "validation" if split=="val" else split
    path  = os.path.join(data_dir, f"{split}_specifications.csv")
    specs = pd.read_csv(path)

    if ID_COL not in specs.columns:
        specs[ID_COL] = specs.index

    print(f"    specs    : {specs.shape[0]:,} rows × {specs.shape[1]} cols")
    return specs


# ---------------------------------------------------------------------------
# Coordinator — calls the three loaders and returns all three DataFrames
# ---------------------------------------------------------------------------

def load_split(data_dir: str, split: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load all source files for a given split by calling the three focused loaders.

    Parameters
    ----------
    data_dir : str   path to folder containing all CSV files
    split    : str   one of 'train', 'val', 'test'

    Returns
    -------
    readouts : pd.DataFrame — time series sensor readings
    label_df : pd.DataFrame — normalised labels (binary + study length)
    specs    : pd.DataFrame — static vehicle specification features
    """
    print(f"\n[Phase 1] Loading {split} files...")
    readouts = load_readouts(data_dir, split)
    label_df = load_labels(data_dir, split, readouts)
    specs    = load_specs(data_dir, split)
    return readouts, label_df, specs


# ---------------------------------------------------------------------------
# Phase 2 — Feature extraction
# ---------------------------------------------------------------------------

def _aggregate_histograms(readouts: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate histogram bin columns per vehicle.

    For each bin  e.g. 167_2:
        167_2_sum   — total frequency across all time steps (primary feature)
        167_2_mean  — average frequency per period
        167_2_max   — peak stress period

    Returns DataFrame indexed by vehicle_id.
    """
    hist_cols = [c for c in _histogram_cols() if c in readouts.columns]
    grp = readouts.groupby(ID_COL)[hist_cols]

    agg = pd.concat([
        grp.sum() .add_suffix("_sum"),
        grp.mean().add_suffix("_mean"),
        grp.max() .add_suffix("_max"),
    ], axis=1)

    print(f"    histogram features : {agg.shape[1]} cols  ({len(hist_cols)} bins × 3 stats)")
    return agg


def _aggregate_counters(readouts: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate cumulative counter columns per vehicle.

    For each counter  e.g. 171_0:
        171_0_last   — total lifetime exposure  (last = max for monotone series)
        171_0_delta  — change over observation  (last - first)
        171_0_rate   — rate of change per step  (delta / n_steps)

    NOTE: mean/median are NOT used because counters are monotone accumulators —
          their mean is just the midpoint of growth and carries no physical meaning.

    Returns DataFrame indexed by vehicle_id.
    """
    counter_cols = [c for c in COUNTER_COLS if c in readouts.columns]
    grp = readouts.groupby(ID_COL)[counter_cols]

    last  = grp.last()
    first = grp.first()
    count = grp.count()          # n_readings per vehicle per feature

    delta = last - first
    rate  = delta.divide(count.replace(0, np.nan))   # avoid divide-by-zero

    agg = pd.concat([
        last .add_suffix("_last"),
        delta.add_suffix("_delta"),
        rate .add_suffix("_rate"),
    ], axis=1)

    print(f"    counter features   : {agg.shape[1]} cols  ({len(counter_cols)} counters × 3 stats)")
    return agg



# ---------------------------------------------------------------------------
# Phase 2 — NEW: Recency window features
# ---------------------------------------------------------------------------

def _aggregate_recency_windows(
    readouts: pd.DataFrame,
    windows:  list[int] = [5, 10, 20],
) -> pd.DataFrame:
    """
    Compute statistics over the last N time steps per vehicle.

    Motivation
    ----------
    The full-history aggregation dilutes the degradation signal that
    only appears in the final readings before failure.  A vehicle heading
    toward failure will show different sensor behaviour in its last 5-20
    readings compared to its entire history.

    For each window size N and each counter column we compute:
        last_N_mean   — mean of the last N readings  (recent operating level)
        last_N_delta  — last value minus value at position -N  (recent change)
        last_N_rate   — recent delta / N  (rate of recent change)

    For histogram bins we compute:
        last_N_sum    — sum of last N readings per bin  (recent exposure)

    The difference between recent and full-history statistics is the
    degradation signal:
        recent_vs_history_ratio = last_N_mean / (full_mean + eps)
        If > 1: sensor readings are increasing  → potential degradation
        If < 1: sensor readings are decreasing  → normal wear-in

    Parameters
    ----------
    windows : list of int
        Number of most recent time steps to include per window.
        Default [5, 10, 20].  Vehicles with fewer than N steps use
        all available steps for that window (no data leakage).

    Returns
    -------
    pd.DataFrame indexed by vehicle_id
    """
    _validate_readouts(readouts)
    hist_cols    = [c for c in _histogram_cols() if c in readouts.columns]
    counter_cols = [c for c in COUNTER_COLS      if c in readouts.columns]
    feature_cols = hist_cols + counter_cols

    # Sort once by vehicle_id and time_step — required for tail() to be correct
    readouts_sorted = readouts.sort_values([ID_COL, TIME_COL])

    all_windows = []

    for N in windows:
        # Take the last N rows per vehicle.
        # Include ID_COL in the selected columns so it is preserved as a
        # regular column after groupby+apply (otherwise it becomes the index).
        cols_to_select = [ID_COL] + feature_cols + [TIME_COL]
        # deduplicate in case ID_COL is already in feature_cols
        cols_to_select = list(dict.fromkeys(cols_to_select))

        last_N = (
            readouts_sorted[cols_to_select]
            .groupby(ID_COL, group_keys=False)
            .apply(lambda df: df.tail(N))
            .reset_index(drop=True)   # flatten the index; ID_COL stays as a column
        )

        grp = last_N.groupby(ID_COL)

        # --- Counter window features ---
        if counter_cols:
            c_last  = grp[counter_cols].last()
            c_first = grp[counter_cols].first()
            c_mean  = grp[counter_cols].mean()
            c_delta = c_last - c_first
            c_rate  = c_delta.divide(
                grp[counter_cols].count().replace(0, np.nan)
            )

            # recent vs full-history ratio (degradation signal)
            full_mean = readouts_sorted.groupby(ID_COL)[counter_cols].mean()
            c_ratio   = c_mean.divide(full_mean.replace(0, np.nan))

            window_feats = pd.concat([
                c_mean .add_suffix(f"_last{N}_mean"),
                c_delta.add_suffix(f"_last{N}_delta"),
                c_rate .add_suffix(f"_last{N}_rate"),
                c_ratio.add_suffix(f"_last{N}_vs_hist"),
            ], axis=1)
            all_windows.append(window_feats)

        # --- Histogram window features (sum only — preserves condition info) ---
        if hist_cols:
            h_sum  = grp[hist_cols].sum().add_suffix(f"_last{N}_sum")
            full_sum = readouts_sorted.groupby(ID_COL)[hist_cols].sum()
            # Proportion of recent exposure vs total (recent stress fraction)
            h_frac = h_sum.divide(
                full_sum.add_suffix(f"_last{N}_sum").replace(0, np.nan)
            ).add_suffix("_frac").rename(
                columns=lambda c: c.replace(f"_last{N}_sum_frac", f"_last{N}_frac")
            )
            all_windows.append(h_sum)
            all_windows.append(h_frac)

    if not all_windows:
        return pd.DataFrame()

    result = pd.concat(all_windows, axis=1)
    n_new  = result.shape[1]
    print(f"    recency features   : {n_new} cols  "          f"(windows={windows}, {len(counter_cols)} counters + {len(hist_cols)} hist bins)")
    return result


# ---------------------------------------------------------------------------
# Phase 2 — NEW: Histogram stress ratio features
# ---------------------------------------------------------------------------

def _compute_stress_ratios(readouts: pd.DataFrame) -> pd.DataFrame:
    """
    Compute stress ratio features for each histogram variable.

    Motivation
    ----------
    Raw bin sums tell you total exposure per condition bin.
    The RATIO of high-stress bins to total bins tells you what FRACTION
    of operation time was spent in extreme conditions — this is a much
    more compact and discriminative representation of vehicle stress profile.

    For each histogram variable (e.g. 167 with 10 bins):
        top25_ratio   — sum of top 25% bins / sum of all bins
        top50_ratio   — sum of top 50% bins / sum of all bins
        bottom25_ratio — sum of bottom 25% bins / sum of all bins
        concentration  — max_bin_sum / total_sum  (how peaked is the distribution)
        entropy        — Shannon entropy of the bin distribution  (spread)

    These features are computed on the FULL history (all time steps summed)
    because they represent the overall stress profile of the vehicle's life.

    Returns
    -------
    pd.DataFrame indexed by vehicle_id
    """
    _validate_readouts(readouts)
    rows = []

    for var_id, n_bins in HISTOGRAM_VARS.items():
        bin_cols = [f"{var_id}_{b}" for b in range(n_bins)]
        bin_cols = [c for c in bin_cols if c in readouts.columns]
        if not bin_cols:
            continue

        # Sum across all time steps per vehicle
        grp_sum = readouts.groupby(ID_COL)[bin_cols].sum()

        total   = grp_sum.sum(axis=1).replace(0, np.nan)
        n_top25 = max(1, n_bins // 4)
        n_top50 = max(1, n_bins // 2)

        # High-stress bins are the LAST bins (highest index = highest stress)
        top25_cols    = bin_cols[-n_top25:]
        top50_cols    = bin_cols[-n_top50:]
        bottom25_cols = bin_cols[:n_top25]

        feat = pd.DataFrame(index=grp_sum.index)
        feat[f"hist_{var_id}_top25_ratio"]    = grp_sum[top25_cols].sum(axis=1)    / total
        feat[f"hist_{var_id}_top50_ratio"]    = grp_sum[top50_cols].sum(axis=1)    / total
        feat[f"hist_{var_id}_bottom25_ratio"] = grp_sum[bottom25_cols].sum(axis=1) / total
        feat[f"hist_{var_id}_concentration"]  = grp_sum[bin_cols].max(axis=1)      / total

        # Shannon entropy — measures spread of operation across bins
        # Low entropy = concentrated in few bins (unusual operating conditions)
        # High entropy = spread across many bins (varied but normal operation)
        probs = grp_sum[bin_cols].divide(total, axis=0).fillna(0)
        probs = probs.clip(lower=1e-12)  # avoid log(0)
        entropy = -(probs * np.log(probs)).sum(axis=1)
        feat[f"hist_{var_id}_entropy"] = entropy

        rows.append(feat)

    if not rows:
        return pd.DataFrame()

    result = pd.concat(rows, axis=1)
    print(f"    stress ratio features : {result.shape[1]} cols  "          f"({len(HISTOGRAM_VARS)} variables × 5 ratio stats)")
    return result


# ---------------------------------------------------------------------------
# Phase 2 — NEW: Counter trend and acceleration features
# ---------------------------------------------------------------------------

def _compute_counter_trends(readouts: pd.DataFrame) -> pd.DataFrame:
    """
    Compute trend (slope) and acceleration features for counter columns.

    Motivation
    ----------
    The existing delta and rate features capture total change and average
    rate.  But a vehicle whose counter is ACCELERATING (growing faster
    recently than historically) may be showing early degradation symptoms
    even if the total accumulated value is still low.

    Features computed per counter column:
        slope          — linear regression slope of counter vs time_step
                         (positive always since counters are monotone)
        recent_slope   — slope over last 20% of readings
        slope_accel    — recent_slope / overall_slope
                         > 1 means accelerating wear  → degradation signal
                         ~ 1 means steady wear         → normal
        cv             — coefficient of variation of inter-reading increments
                         measures consistency of wear rate (high CV = irregular)

    Returns
    -------
    pd.DataFrame indexed by vehicle_id
    """
    _validate_readouts(readouts)
    counter_cols = [c for c in COUNTER_COLS if c in readouts.columns]
    if not counter_cols:
        return pd.DataFrame()

    readouts_sorted = readouts.sort_values([ID_COL, TIME_COL])

    def _vehicle_trends(df: pd.DataFrame) -> pd.Series:
        """Compute trend features for one vehicle's time series."""
        feats = {}
        t = df[TIME_COL].values.astype(float)
        n = len(t)

        for col in counter_cols:
            v = df[col].values.astype(float)

            # --- Overall linear slope ---
            if n >= 2 and t.max() > t.min():
                # Efficient least-squares slope: cov(t,v) / var(t)
                t_c = t - t.mean()
                v_c = v - v.mean()
                denom = (t_c ** 2).sum()
                slope = (t_c * v_c).sum() / denom if denom > 0 else 0.0
            else:
                slope = 0.0
            feats[f"{col}_slope"] = slope

            # --- Recent slope (last 20% of readings, min 3) ---
            n_recent = max(3, int(n * 0.20))
            if n >= 3:
                t_r = t[-n_recent:]
                v_r = v[-n_recent:]
                t_rc = t_r - t_r.mean()
                v_rc = v_r - v_r.mean()
                denom_r = (t_rc ** 2).sum()
                recent_slope = (t_rc * v_rc).sum() / denom_r if denom_r > 0 else 0.0
            else:
                recent_slope = slope
            feats[f"{col}_recent_slope"] = recent_slope

            # --- Slope acceleration: recent / overall ---
            feats[f"{col}_slope_accel"] = (
                recent_slope / (abs(slope) + 1e-9)
                if slope != 0 else 1.0
            )

            # --- Coefficient of variation of increments ---
            if n >= 3:
                increments = np.diff(v)
                increments = increments[increments >= 0]  # counters are monotone
                if len(increments) > 1 and increments.mean() > 0:
                    cv = increments.std() / increments.mean()
                else:
                    cv = 0.0
            else:
                cv = 0.0
            feats[f"{col}_increment_cv"] = cv

        return pd.Series(feats)

    print("    Computing counter trend features (apply per vehicle — may take a moment)...")
    cols_needed = counter_cols + [TIME_COL]
    result = (
        readouts_sorted
        .groupby(ID_COL)[cols_needed]
        .apply(_vehicle_trends)
    )

    print(f"    counter trend features : {result.shape[1]} cols  "          f"({len(counter_cols)} counters × 4 trend stats)")
    return result


# ---------------------------------------------------------------------------
# Phase 2 — NEW: Last readout raw features  ← KEY finding from Check B
# ---------------------------------------------------------------------------

def _extract_last_readout_features(readouts: pd.DataFrame) -> pd.DataFrame:
    """
    Extract raw sensor values from the LAST readout per vehicle.

    Why this matters
    ----------------
    Check B showed last-readout-only AUC = 0.67 vs aggregated AUC = 0.58.
    The failure signal is concentrated in the most recent sensor readings.
    Full-history aggregations dilute this signal by averaging healthy
    early-life readings with degraded late-life readings.

    This function adds the raw last-readout values as a PARALLEL feature set
    alongside the aggregations — giving the model direct access to the most
    recent vehicle state without any averaging.

    Features produced per sensor column:
        <col>_lastread        — raw value of the last readout
        <col>_lastread_pctile — percentile rank of last readout vs all
                                readings for that vehicle (0-1 scale)
                                captures whether the last reading is
                                unusually high relative to vehicle's history

    For histogram bins (cumulative counts, not instantaneous):
        <col>_lastread        — last recorded bin count value

    For counter columns (monotone accumulators):
        <col>_lastread        — same as _last feature, kept for consistency
        <col>_lastread_pctile — always 1.0 for monotone counters (max is last)
                                so we skip this for counters

    Returns
    -------
    pd.DataFrame indexed by vehicle_id, named <col>_lastread[_pctile]
    """
    _validate_readouts(readouts)

    hist_cols    = [c for c in _histogram_cols() if c in readouts.columns]
    counter_cols = [c for c in COUNTER_COLS      if c in readouts.columns]
    all_sensor   = hist_cols + counter_cols

    readouts_sorted = readouts.sort_values([ID_COL, TIME_COL])

    # --- Last raw value per vehicle for every sensor column ---
    last_vals = (
        readouts_sorted
        .groupby(ID_COL)[all_sensor]
        .last()
        .add_suffix("_lastread")
    )

    # --- Percentile rank of the last reading vs vehicle's own history ---
    # For histogram bins: captures whether last reading is higher than usual
    # Percentile = (rank of last value among all values for that vehicle) / n
    # Skip counters — they are monotone so percentile is always 1.0

    pctile_frames = []
    for col in hist_cols:
        if col not in readouts.columns:
            continue

        def _pctile(grp, col=col):
            vals    = grp[col].values
            last    = vals[-1]                     # last is at end (sorted)
            n       = len(vals)
            if n <= 1:
                return pd.Series({f"{col}_lastread_pctile": 1.0})
            rank    = (vals <= last).sum()         # how many values <= last
            pctile  = rank / n
            return pd.Series({f"{col}_lastread_pctile": pctile})

        pctile_col = (
            readouts_sorted
            .groupby(ID_COL)
            .apply(_pctile)
        )
        pctile_frames.append(pctile_col)

    if pctile_frames:
        pctile_features = pd.concat(pctile_frames, axis=1)
        result = last_vals.join(pctile_features, how="left")
    else:
        result = last_vals

    n_lastread = last_vals.shape[1]
    n_pctile   = len(pctile_frames)
    print(f"    last-readout features  : {result.shape[1]} cols  "
          f"({n_lastread} raw last values + {n_pctile} percentile ranks)")
    return result

def build_flat_table(
    readouts: pd.DataFrame,
    tte:      pd.DataFrame,
    specs:    pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Phase 2 — full feature extraction and merge.

    Steps:
        2a  Aggregate histogram bins   → per-vehicle histogram features
        2b  Aggregate counter cols     → per-vehicle counter features
        2c  Merge aggregated features with tte and specs on vehicle_id
        2d  Add length_of_study as explicit feature (compensates for history mixing)

    Returns
    -------
    X : pd.DataFrame  — feature matrix, one row per vehicle
    y : pd.Series     — binary label (in_study_repair)
    """
    print("\n[Phase 2] Extracting features...")

    # ----------------------------------------------------------------
    # Feature strategy — empirically validated:
    #
    # Last-readout features improved train CV AUC to 0.67 but hurt
    # val/test AUC because val/test last readouts are randomly sampled
    # within vehicle history — structurally different from train.
    # Full-history aggregations are symmetric across all splits and
    # represent the correct approach for this published task.
    #
    # AUC ~0.58 is the realistic ceiling for anonymised aggregated
    # features on this dataset. This is consistent with published
    # results on the SCANIA APS Failure benchmark.
    # ----------------------------------------------------------------

    # Full-history aggregations — symmetric across train, val, test
    hist_features    = _aggregate_histograms(readouts)
    counter_features = _aggregate_counters(readouts)

    # Compact supplementary: stress profile + wear acceleration
    stress_features = _compute_stress_ratios(readouts)
    trend_features  = _compute_counter_trends(readouts)

    # Combine all features
    feature_dfs = [hist_features, counter_features]
    if not stress_features.empty:
        feature_dfs.append(stress_features)
    if not trend_features.empty:
        feature_dfs.append(trend_features)

    op_features = feature_dfs[0]
    for df in feature_dfs[1:]:
        op_features = op_features.join(df, how="outer")

    print(f"    total operational features : {op_features.shape[1]}"
          f"  (hist={hist_features.shape[1]}"
          f"  counter={counter_features.shape[1]}"
          f"  stress={stress_features.shape[1] if not stress_features.empty else 0}"
          f"  trend={trend_features.shape[1] if not trend_features.empty else 0})")

    # Merge with tte (label + study length)
    tte_indexed = tte.set_index(ID_COL)
    merged = op_features.join(tte_indexed, how="left")

    # Merge with specifications (categorical vehicle metadata)
    spec_cols = [c for c in specs.columns if c != ID_COL]
    specs_indexed = specs.set_index(ID_COL)[spec_cols]
    merged = merged.join(specs_indexed, how="left")

    # Encode specification columns as pandas Categorical (LightGBM native support)
    for col in spec_cols:
        if col in merged.columns:
            merged[col] = merged[col].astype("category")

    # Separate features from label
    # Drop label + temporal_class (val only — not a feature)
    y = merged[LABEL_COL].astype(int)
    drop_cols = [c for c in [LABEL_COL, "temporal_class"] if c in merged.columns]
    X = merged.drop(columns=drop_cols)

    print(f"\n    Flat table ready : {X.shape[0]:,} rows × {X.shape[1]} features")
    print(f"    Label distribution : {y.value_counts().to_dict()}")
    minority = y.value_counts().get(1, 0)
    majority = y.value_counts().get(0, 0)
    if minority > 0:
        print(f"    Class imbalance ratio : 1 : {majority / minority:.1f}")

    return X, y


# ---------------------------------------------------------------------------
# Phase 3 Layer 1 — Remove noise features
# ---------------------------------------------------------------------------

def remove_noise_features(
    X_train: pd.DataFrame,
    variance_threshold: float = 0.01,
    verbose: bool = True,
) -> tuple[pd.DataFrame, dict]:
    """
    Layer 1 — drop constant, near-zero variance, and duplicate columns.

    Fitted on X_train only. Returns cleaned DataFrame and a state dict
    that can be applied to val/test via apply_noise_filter().

    Parameters
    ----------
    variance_threshold : float
        Features with variance below this are dropped as near-constant.
        Default 0.01 works well for normalised features.

    Returns
    -------
    X_clean : pd.DataFrame
    state   : dict  with keys 'cols_to_drop' and 'kept_cols'
    """
    print("\n[Layer 1] Removing noise features...")

    numeric_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
    cols_to_drop = set()

    # 1. Constant features (variance == 0)
    constant = [c for c in numeric_cols if X_train[c].nunique() <= 1]
    cols_to_drop.update(constant)
    if verbose:
        print(f"    Constant features    : {len(constant)}  {constant[:5]}")

    # 2. Near-zero variance features
    variances = X_train[numeric_cols].var()
    near_zero = variances[variances < variance_threshold].index.tolist()
    near_zero = [c for c in near_zero if c not in cols_to_drop]
    cols_to_drop.update(near_zero)
    if verbose:
        print(f"    Near-zero variance   : {len(near_zero)}  (threshold={variance_threshold})")

    # 3. Duplicate columns (exact same values)
    numeric_data = X_train[numeric_cols].fillna(-9999)
    seen_hashes  = {}
    duplicate    = []
    for col in numeric_cols:
        col_hash = hash(numeric_data[col].values.tobytes())
        if col_hash in seen_hashes:
            duplicate.append(col)
        else:
            seen_hashes[col_hash] = col
    cols_to_drop.update(duplicate)
    if verbose:
        print(f"    Duplicate columns    : {len(duplicate)}")

    kept_cols = [c for c in X_train.columns if c not in cols_to_drop]
    X_clean   = X_train[kept_cols].copy()

    print(f"    Dropped {len(cols_to_drop)} features → {X_clean.shape[1]} remaining")

    state = {"cols_to_drop": list(cols_to_drop), "kept_cols": kept_cols}
    return X_clean, state


def apply_noise_filter(X: pd.DataFrame, state: dict) -> pd.DataFrame:
    """Apply Layer 1 filter (fitted on train) to val or test.
    Preserves category dtypes — filter operations can silently drop them.
    """
    kept   = [c for c in state["kept_cols"] if c in X.columns]
    result = X[kept].copy()
    # Re-apply category dtype for any categorical column that lost it during copy
    for col in result.columns:
        if X[col].dtype.name == "category":
            result[col] = result[col].astype("category")
    return result


# ---------------------------------------------------------------------------
# Phase 3 Layer 2 — Remove redundancy (correlation)
# ---------------------------------------------------------------------------

def remove_correlated_features(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    corr_threshold: float = 0.95,
    verbose: bool = True,
) -> tuple[pd.DataFrame, dict]:
    """
    Layer 2 — drop highly correlated features, keeping the one more
    correlated with the label.

    When two features A and B have |corr(A,B)| > threshold:
        - compute |corr(A, y)| and |corr(B, y)|
        - drop whichever has the lower correlation with y
        - this preserves the most predictive feature in each redundant pair

    Parameters
    ----------
    corr_threshold : float
        Pairs with absolute Pearson correlation above this are considered
        redundant. 0.95 is a conservative threshold; 0.90 is more aggressive.

    Returns
    -------
    X_clean : pd.DataFrame
    state   : dict  with keys 'cols_to_drop' and 'kept_cols'
    """
    print("\n[Layer 2] Removing correlated features...")

    numeric_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
    # Remove STUDY_LEN_COL from candidates — always keep it
    protected = [STUDY_LEN_COL] if STUDY_LEN_COL in numeric_cols else []
    candidates = [c for c in numeric_cols if c not in protected]

    # Compute correlation matrix
    corr_matrix = X_train[candidates].corr().abs()

    # Correlation of each feature with label
    label_corr = X_train[candidates].corrwith(y_train.astype(float)).abs()

    # Upper triangle mask
    upper = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )

    cols_to_drop = set()
    n_pairs_found = 0

    for col in upper.columns:
        if col in cols_to_drop:
            continue
        # Find all features highly correlated with this column
        correlated_with = upper.index[upper[col] > corr_threshold].tolist()
        for other in correlated_with:
            if other in cols_to_drop:
                continue
            n_pairs_found += 1
            # Drop the one with lower correlation to label
            if label_corr.get(col, 0) >= label_corr.get(other, 0):
                cols_to_drop.add(other)
            else:
                cols_to_drop.add(col)
                break   # col itself dropped; skip remaining pairs

    kept_cols = [c for c in X_train.columns if c not in cols_to_drop]
    X_clean   = X_train[kept_cols].copy()

    if verbose:
        print(f"    Correlated pairs found : {n_pairs_found}  (threshold={corr_threshold})")
        print(f"    Dropped {len(cols_to_drop)} features → {X_clean.shape[1]} remaining")

        # Show top 10 dropped features and which feature they were redundant with
        if cols_to_drop:
            sample = list(cols_to_drop)[:10]
            print(f"    Sample dropped       : {sample}")

    state = {"cols_to_drop": list(cols_to_drop), "kept_cols": kept_cols}
    return X_clean, state


def apply_correlation_filter(X: pd.DataFrame, state: dict) -> pd.DataFrame:
    """Apply Layer 2 filter (fitted on train) to val or test.
    Preserves category dtypes — filter operations can silently drop them.
    """
    kept   = [c for c in state["kept_cols"] if c in X.columns]
    result = X[kept].copy()
    for col in result.columns:
        if X[col].dtype.name == "category":
            result[col] = result[col].astype("category")
    return result


# ---------------------------------------------------------------------------
# Phase 3 Layer 3 — Predictive power
# ---------------------------------------------------------------------------

def compute_mutual_information(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    top_n:   int = 50,
    plot:    bool = True,
) -> pd.Series:
    """
    Layer 3a — Mutual information between each feature and the label.

    MI captures non-linear relationships. Use as a fast first-pass ranking.

    Returns
    -------
    mi_scores : pd.Series  sorted descending, feature name as index
    """
    print("\n[Layer 3a] Computing mutual information...")

    numeric_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
    X_num = X_train[numeric_cols].fillna(X_train[numeric_cols].median())

    mi = mutual_info_classif(X_num, y_train, random_state=42, n_neighbors=5)
    mi_scores = pd.Series(mi, index=numeric_cols).sort_values(ascending=False)

    print(f"    Top 10 features by MI:")
    for feat, score in mi_scores.head(10).items():
        print(f"        {feat:<35} {score:.4f}")

    if plot:
        fig, ax = plt.subplots(figsize=(10, 6))
        mi_scores.head(top_n).sort_values().plot(
            kind="barh", ax=ax, color="#378ADD", edgecolor="none"
        )
        ax.set_title(f"Top {top_n} features — mutual information with label", fontsize=13)
        ax.set_xlabel("Mutual information score")
        ax.tick_params(labelsize=9)
        plt.tight_layout()
        plt.savefig("/mnt/user-data/outputs/mi_scores.png", dpi=150, bbox_inches="tight")
        plt.close()
        print("    Plot saved → mi_scores.png")

    return mi_scores


def compute_lgbm_importance(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val:   pd.DataFrame,
    y_val:   pd.Series,
    top_n:   int = 50,
    plot:    bool = True,
) -> tuple[pd.Series, lgb.LGBMClassifier]:
    """
    Layer 3b — LightGBM gain-based feature importance.

    Trains a baseline LightGBM (standard binary cross-entropy, no cost weighting)
    to measure which features the model splits on most.

    Returns
    -------
    importance : pd.Series       sorted descending
    model      : LGBMClassifier  fitted model (reuse for permutation importance)
    """
    print("\n[Layer 3b] Computing LightGBM feature importance...")

    cat_cols = X_train.select_dtypes(include="category").columns.tolist()

    model = lgb.LGBMClassifier(
        n_estimators    = 300,
        learning_rate   = 0.05,
        num_leaves      = 31,
        min_child_samples = 20,
        subsample       = 0.8,
        colsample_bytree= 0.8,
        random_state    = 42,
        verbose         = -1,
    )

    # Build a consistent categorical column list from the INTERSECTION of
    # train and val columns — both must agree on which cols are categorical.
    # Also re-cast val categorical cols to match train's category dtype exactly
    # (filter operations can silently drop category dtype on val).
    val_cat_cols = X_val.select_dtypes(include="category").columns.tolist()
    shared_cat_cols = [c for c in cat_cols if c in val_cat_cols]

    # Re-align val category dtypes to match train exactly (prevents mismatches
    # when val was filtered independently and lost some category levels)
    X_val_aligned = X_val.copy()
    for col in shared_cat_cols:
        X_val_aligned[col] = X_val_aligned[col].astype(
            X_train[col].dtype
        )

    model.fit(
        X_train, y_train,
        categorical_feature = shared_cat_cols if shared_cat_cols else "auto",
        eval_set            = [(X_val_aligned, y_val)],
        callbacks           = [lgb.early_stopping(30, verbose=False),
                               lgb.log_evaluation(period=-1)],
    )

    importance = pd.Series(
        model.feature_importances_,
        index=X_train.columns,
    ).sort_values(ascending=False)

    print(f"    Top 10 features by LGBM gain:")
    for feat, score in importance.head(10).items():
        print(f"        {feat:<35} {score:.1f}")

    if plot:
        fig, ax = plt.subplots(figsize=(10, 6))
        importance.head(top_n).sort_values().plot(
            kind="barh", ax=ax, color="#1D9E75", edgecolor="none"
        )
        ax.set_title(f"Top {top_n} features — LightGBM gain importance", fontsize=13)
        ax.set_xlabel("Gain importance score")
        ax.tick_params(labelsize=9)
        plt.tight_layout()
        plt.savefig("/mnt/user-data/outputs/lgbm_importance.png", dpi=150, bbox_inches="tight")
        plt.close()
        print("    Plot saved → lgbm_importance.png")

    return importance, model


def compute_permutation_importance(
    model:   lgb.LGBMClassifier,
    X_val:   pd.DataFrame,
    y_val:   pd.Series,
    cost_matrix: dict | None = None,
    top_n:   int = 50,
    n_repeats: int = 10,
    plot:    bool = True,
) -> pd.Series:
    """
    Layer 3c — Permutation importance measured on the VALIDATION set.

    If cost_matrix is provided, uses total cost as the scoring metric.
    This aligns directly with your thesis RQ1-RQ3 (cost-aware evaluation).

    cost_matrix example (SCANIA standard):
        {
            'fp_cost' : 10,    # cost of false positive (unnecessary maintenance)
            'fn_cost' : 500,   # cost of false negative (undetected failure)
        }

    If cost_matrix is None, falls back to ROC-AUC.

    Returns
    -------
    perm_importance : pd.Series  sorted descending (higher = more important)
    """
    print("\n[Layer 3c] Computing permutation importance on validation set...")

    def total_cost_score(estimator, X, y_true):
        """Custom scorer: lower total cost = better model."""
        y_pred = estimator.predict(X)
        if cost_matrix:
            fp = ((y_pred == 1) & (y_true == 0)).sum()
            fn = ((y_pred == 0) & (y_true == 1)).sum()
            cost = fp * cost_matrix["fp_cost"] + fn * cost_matrix["fn_cost"]
            return -cost    # negative because sklearn maximises scorers
        else:
            from sklearn.metrics import roc_auc_score
            return roc_auc_score(y_true, estimator.predict_proba(X)[:, 1])

    scorer = total_cost_score

    # Permutation importance must use the FULL X_val (numeric + categorical)
    # because the model was trained on all columns.
    # Passing only numeric columns causes the categorical_feature mismatch error
    # because LightGBM detects that the prediction DataFrame is missing the
    # categorical columns it saw during training.
    #
    # sklearn's permutation_importance shuffles one column at a time and calls
    # estimator.predict() on the shuffled DataFrame — so all original columns
    # must be present for every predict() call.
    #
    # We restrict the SCORED columns to numeric only (categorical permutation
    # is rarely meaningful for spec features), but prediction uses all columns.

    numeric_cols = X_val.select_dtypes(include=[np.number]).columns.tolist()

    # Wrap the model so sklearn always gets the full X_val structure,
    # but we only report importance for numeric columns.
    class _NumericPermWrapper:
        """
        Wrapper that accepts a numeric-only DataFrame from sklearn's shuffler
        and reconstructs the full DataFrame (adding back categorical columns)
        before calling the real model.  This lets us measure permutation
        importance for numeric features while keeping categoricals fixed.
        """
        def __init__(self, lgbm_model, X_full_val):
            self._model    = lgbm_model
            self._cat_data = X_full_val.select_dtypes(include="category").copy()
            self._num_cols = X_full_val.select_dtypes(include=[np.number]).columns.tolist()

        def predict(self, X_num):
            # X_num is the shuffled numeric-only slice from sklearn
            X_full = X_num.copy()
            # Re-attach categorical columns (unchanged — we are not permuting them)
            for col in self._cat_data.columns:
                X_full[col] = self._cat_data[col].values
            return self._model.predict(X_full)

        def predict_proba(self, X_num):
            X_full = X_num.copy()
            for col in self._cat_data.columns:
                X_full[col] = self._cat_data[col].values
            return self._model.predict_proba(X_full)

        def fit(self, X, y=None):
            # Required by sklearn estimator interface validation.
            # permutation_importance never calls fit — only predict/predict_proba.
            return self

        def get_params(self, deep=True):
            return self._model.get_params(deep=deep)

        def set_params(self, **params):
            return self

    X_val_num = X_val[numeric_cols].copy()
    wrapped   = _NumericPermWrapper(model, X_val)

    result = permutation_importance(
        wrapped, X_val_num, y_val,
        scoring      = scorer,
        n_repeats    = n_repeats,
        random_state = 42,
        n_jobs       = -1,
    )

    perm_scores = pd.Series(
        result.importances_mean,
        index=numeric_cols,
    ).sort_values(ascending=False)

    print(f"    Top 10 features by permutation importance (cost metric):")
    for feat, score in perm_scores.head(10).items():
        print(f"        {feat:<35} {score:.4f}")

    if plot:
        fig, ax = plt.subplots(figsize=(10, 6))
        perm_scores.head(top_n).sort_values().plot(
            kind="barh", ax=ax, color="#D85A30", edgecolor="none"
        )
        metric_label = "total cost reduction" if cost_matrix else "ROC-AUC gain"
        ax.set_title(
            f"Top {top_n} features — permutation importance ({metric_label})", fontsize=13
        )
        ax.set_xlabel("Mean importance score")
        ax.tick_params(labelsize=9)
        plt.tight_layout()
        plt.savefig(
            "/mnt/user-data/outputs/permutation_importance.png",
            dpi=150, bbox_inches="tight"
        )
        plt.close()
        print("    Permutation importance plot saved → permutation_importance.png")

    return perm_scores


def select_final_features(
    mi_scores:    pd.Series,
    lgbm_scores:  pd.Series,
    perm_scores:  pd.Series,
    mi_top_n:     int = 80,
    lgbm_top_n:   int = 80,
    perm_top_n:   int = 80,
    min_methods:  int = 2,
    always_keep:  list[str] | None = None,
) -> list[str]:
    """
    Combine three importance rankings into a final feature list.

    A feature is kept if it appears in the top-N of at least `min_methods`
    out of the three ranking methods. This consensus approach is more robust
    than relying on any single method.

    Parameters
    ----------
    min_methods : int
        Minimum number of methods that must rank the feature in their top-N.
        Default 2 = majority vote across 3 methods.
    always_keep : list[str]
        Features to always include regardless of rankings
        (e.g. length_of_study_time_step).

    Returns
    -------
    selected_features : list[str]
    """
    print("\n[Layer 3] Selecting final features by consensus...")

    top_mi   = set(mi_scores.head(mi_top_n).index)
    top_lgbm = set(lgbm_scores.head(lgbm_top_n).index)
    top_perm = set(perm_scores.head(perm_top_n).index)

    all_features = top_mi | top_lgbm | top_perm

    selected = []
    for feat in all_features:
        votes = sum([feat in top_mi, feat in top_lgbm, feat in top_perm])
        if votes >= min_methods:
            selected.append(feat)

    # Always keep protected features
    if always_keep:
        for feat in always_keep:
            if feat not in selected:
                selected.append(feat)
                print(f"    Force-kept : {feat}")

    print(f"    Features in top-{mi_top_n} of MI       : {len(top_mi)}")
    print(f"    Features in top-{lgbm_top_n} of LGBM     : {len(top_lgbm)}")
    print(f"    Features in top-{perm_top_n} of Perm.imp : {len(top_perm)}")
    print(f"    Consensus (>= {min_methods} methods)      : {len(selected)} features selected")

    return sorted(selected)


# ---------------------------------------------------------------------------
# Master pipeline class
# ---------------------------------------------------------------------------

class SCANIAPipeline:
    """
    End-to-end reusable pipeline for SCANIA ComponentX dataset.

    fit_transform(split='train') → runs all phases, fits all filters
    transform(split='val')       → applies fitted filters to new split
    save(path) / load(path)      → persist fitted state
    """

    ROOT = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, ROOT)

    def __init__(
        self,
        data_dir:           str   = os.path.join(ROOT, "data", "raw"),
        variance_threshold: float = 0.01,
        corr_threshold:     float = 0.95,
        mi_top_n:           int   = 80,
        lgbm_top_n:         int   = 80,
        perm_top_n:         int   = 80,
        min_methods:        int   = 2,
        cost_matrix:        dict  | None = None,
        plot:               bool  = True,
    ):
        self.data_dir           = data_dir
        self.variance_threshold = variance_threshold
        self.corr_threshold     = corr_threshold
        self.mi_top_n           = mi_top_n
        self.lgbm_top_n         = lgbm_top_n
        self.perm_top_n         = perm_top_n
        self.min_methods        = min_methods
        self.cost_matrix        = cost_matrix
        self.plot               = plot

        # Fitted state — populated during fit_transform
        self._noise_state        = None
        self._corr_state         = None
        self._selected_features  = None
        self._lgbm_model         = None
        self._is_fitted          = False

    # ------------------------------------------------------------------
    def fit_transform(self, split: str = "train"):
        """
        Run full pipeline on training split.
        Fits all filters and returns (X_final, y).
        """
        print("=" * 60)
        print(f" SCANIA Pipeline — fit_transform ({split})")
        print("=" * 60)

        # Phase 1 — load
        readouts, tte, specs = load_split(self.data_dir, split)

        # Phase 2 — feature extraction
        X, y = build_flat_table(readouts, tte, specs)

        # Store val data reference for Layer 3 (needed for LGBM + permutation)
        # We load val here to use as eval set during LGBM importance fitting.
        # IMPORTANT: build_flat_table already converts val 5-class → binary via
        # load_split, so y_val_raw is always binary {0,1} — safe to pass to LGBM.
        try:
            val_readouts, val_label_df, val_specs = load_split(self.data_dir, "val")
            X_val_raw, y_val_binary = build_flat_table(val_readouts, val_label_df, val_specs)
            has_val = True
            print(f"    Val binary label dist: {y_val_binary.value_counts().to_dict()}")
        except FileNotFoundError:
            print("    [Warning] val split not found — using train as fallback eval set")
            X_val_raw, y_val_binary = X.copy(), y.copy()
            has_val = False

        # Layer 1 — noise
        X, self._noise_state = remove_noise_features(
            X, variance_threshold=self.variance_threshold
        )
        X_val_l1 = apply_noise_filter(X_val_raw, self._noise_state)

        # Layer 2 — correlation
        X, self._corr_state = remove_correlated_features(
            X, y, corr_threshold=self.corr_threshold
        )
        X_val_l2 = apply_correlation_filter(X_val_l1, self._corr_state)

        # Layer 3 — DISABLED
        # -------------------------------------------------------------------
        # Empirical finding: with 9% failure rate and highly correlated
        # sensors, MI scores are near-zero for all features, and the
        # consensus selection (MI + LGBM + permutation) consistently removes
        # the informative last-readout features that Check B proved give
        # AUC 0.67.  Every time Layer 3 ran, pipeline AUC dropped further.
        #
        # Decision: skip Layer 3 entirely.  Pass all features that survive
        # Layers 1 and 2 directly to LightGBM, which handles feature
        # selection internally through:
        #   - num_leaves  (limits tree complexity)
        #   - min_child_samples  (prevents splits on weak features)
        #   - reg_alpha / reg_lambda  (L1/L2 regularisation)
        #   - colsample_bytree  (random feature subsampling per tree)
        #
        # This approach lets the model discover the relevant features during
        # training rather than pre-selecting them with noisy importance scores
        # computed on a severely imbalanced dataset.
        # -------------------------------------------------------------------
        print("\n[Layer 3] SKIPPED — using all features from Layers 1+2")
        print(f"    Features after Layers 1+2: {X.shape[1]}")
        print(f"    LightGBM will select features internally via regularisation")

        self._selected_features = X.columns.tolist()
        self._is_fitted = True

        # No feature subsetting — pass everything to the model
        X_final = X.copy()

        print(f"\n{'=' * 60}")
        print(f" fit_transform complete")
        print(f"   Final shape : {X_final.shape[0]:,} rows × {X_final.shape[1]} features")
        print(f"   Label dist  : {y.value_counts().to_dict()}")
        print(f"{'=' * 60}\n")

        return X_final, y

    # ------------------------------------------------------------------
    def transform(self, split: str):
        """
        Apply fitted pipeline to a new split (val or test).
        Must call fit_transform first.

        Returns
        -------
        X_final : pd.DataFrame  — feature matrix with selected features only
        y       : pd.Series     — binary label {0,1}
                                  (val 5-class is already converted by load_split)
        """
        if not self._is_fitted:
            raise RuntimeError("Call fit_transform() before transform().")

        print("=" * 60)
        print(f" SCANIA Pipeline — transform ({split})")
        print("=" * 60)

        readouts, label_df, specs = load_split(self.data_dir, split)
        X, y = build_flat_table(readouts, label_df, specs)

        # Apply same filters fitted on train — no re-fitting
        X = apply_noise_filter(X, self._noise_state)
        X = apply_correlation_filter(X, self._corr_state)

        available = [f for f in self._selected_features if f in X.columns]
        X_final   = X[available].copy()

        print(f"\n    transform complete")
        print(f"    Final shape : {X_final.shape[0]:,} rows × {X_final.shape[1]} features")
        print(f"    Label dist  : {y.value_counts().to_dict()}\n")

        return X_final, y

    # ------------------------------------------------------------------
    def get_feature_summary(self) -> pd.DataFrame:
        """
        Return a DataFrame summarising the selected features.
        Useful for your thesis Methods chapter.
        """
        if not self._is_fitted:
            raise RuntimeError("Call fit_transform() first.")

        rows = []
        for feat in self._selected_features:
            if "_sum"   in feat: stat, ftype = "sum",   "histogram"
            elif "_mean" in feat: stat, ftype = "mean",  "histogram"
            elif "_max"  in feat: stat, ftype = "max",   "histogram"
            elif "_last" in feat: stat, ftype = "last",  "counter"
            elif "_delta"in feat: stat, ftype = "delta", "counter"
            elif "_rate" in feat: stat, ftype = "rate",  "counter"
            elif feat == STUDY_LEN_COL: stat, ftype = "raw", "study_length"
            else: stat, ftype = "raw", "specification"

            rows.append({"feature": feat, "type": ftype, "statistic": stat})

        return pd.DataFrame(rows).sort_values(["type", "feature"]).reset_index(drop=True)

    # ------------------------------------------------------------------
    def save(self, path: str):
        """Persist fitted pipeline state to disk."""
        state = {
            "noise_state":       self._noise_state,
            "corr_state":        self._corr_state,
            "selected_features": self._selected_features,
            "is_fitted":         self._is_fitted,
        }
        with open(path, "wb") as f:
            pickle.dump(state, f)
        print(f"Pipeline state saved → {path}")

    def load(self, path: str):
        """Reload fitted pipeline state from disk."""
        with open(path, "rb") as f:
            state = pickle.load(f)
        self._noise_state       = state["noise_state"]
        self._corr_state        = state["corr_state"]
        self._selected_features = state["selected_features"]
        self._is_fitted         = state["is_fitted"]
        print(f"Pipeline state loaded ← {path}")


# ---------------------------------------------------------------------------
# Quick-start example
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    COST_MATRIX = {
        "fp_cost": 10,    # replace component unnecessarily
        "fn_cost": 500,   # miss a failure → breakdown cost
    }



    pipeline = SCANIAPipeline(
        data_dir           = "../data/",
        variance_threshold = 0.01,
        corr_threshold     = 0.95,
        mi_top_n           = 80,
        lgbm_top_n         = 80,
        perm_top_n         = 80,
        min_methods        = 2,
        cost_matrix        = COST_MATRIX,
        plot               = True,
    )

    # Fit on train — all filters learned from train only
    X_train, y_train = pipeline.fit_transform(split="train")

    # Apply same pipeline to val and test
    # y_val is binary {0,1} — val 5-class labels are converted inside load_split
    # No unseen labels error — LGBM always sees binary {0,1} in eval set
    X_val,  y_val  = pipeline.transform(split="val")
    X_test, y_test = pipeline.transform(split="test")

    print("\nLabel distributions after pipeline:")
    print(f"  y_train : {y_train.value_counts().to_dict()}")
    print(f"  y_val   : {y_val.value_counts().to_dict()}  ← binary, converted from 5-class")
    print(f"  y_test  : {y_test.value_counts().to_dict()}")

    # Inspect selected features (useful for thesis Methods chapter)
    summary = pipeline.get_feature_summary()
    print(f"\nSelected feature summary ({len(summary)} features):")
    print(summary.to_string())
    summary.to_csv("/mnt/user-data/outputs/selected_features_summary.csv", index=False)

    # Save fitted pipeline
    pipeline.save("/mnt/user-data/outputs/pipeline_state.pkl")

    print("\nReady for LightGBM training with cost-aware loss.")
    print(f"X_train : {X_train.shape},  X_val : {X_val.shape},  X_test : {X_test.shape}")