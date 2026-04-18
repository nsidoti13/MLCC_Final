#!/usr/bin/env python3
"""
scripts/train_model.py
======================
End-to-end model training using available local data:
  - MODIS active fire CSV  → labels
  - CAWeather stations CSV → lagged weather features
  - CAPDSI CSV             → drought index feature
  - H3 grid               → spatial framework

Memory-efficient strategy
-------------------------
The full (cell, date) grid is ~64 M rows — too large for RAM.
Instead we:
  1. Build ALL positive cell-days from MODIS (label=1).
  2. Sample NEG_RATIO × |positives| random negative cell-days (label=0).
  3. Compute station-level rolling features once, then join to sample.
This keeps the working dataset at ~2–5 M rows.

New features (Phase 1)
----------------------
- Fire history features: days_since_last_fire, cell_fire_count_1yr,
  neighbor_fire_7d, neighbor_fire_30d
- Spatial features: cell_lat, cell_lon, dist_to_coast_km, is_inland
- Weather-derived features: temp_range, vpd_proxy, hot_dry_windy,
  consecutive_dry_days
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("train_model")

# ── Config ────────────────────────────────────────────────────────────────────
CONFIG = yaml.safe_load(open(PROJECT_ROOT / "configs" / "config.yaml"))
BBOX        = CONFIG["grid"]["bbox"]
RESOLUTION  = CONFIG["grid"]["resolution"]
WINDOW      = CONFIG["temporal"]["prediction_window_days"]   # 7
LAG_SHORT   = CONFIG["temporal"]["lag_days_short"]           # 7
LAG_LONG    = CONFIG["temporal"]["lag_days_long"]            # 30
TRAIN_END   = "2021-12-31"
VAL_START   = "2022-01-01";  VAL_END   = "2022-12-31"
TEST_START  = "2023-01-01";  TEST_END  = "2023-12-31"
NEG_RATIO   = 20             # negative samples per positive
RANDOM_SEED = 42

# ── Paths ─────────────────────────────────────────────────────────────────────
MODIS_CSV   = PROJECT_ROOT / "data/raw/modis/MODISFireData.csv"
WEATHER_CSV = PROJECT_ROOT / "data/raw/weather_stations/CAWeather.csv"
PDSI_CSV    = PROJECT_ROOT / "data/raw/drought/CAPDSI.csv"
LANDFIRE_DIR = PROJECT_ROOT / "data/raw/landfire"
OUT_DIR     = PROJECT_ROOT / "outputs"
(OUT_DIR / "models").mkdir(parents=True, exist_ok=True)
(OUT_DIR / "reports").mkdir(parents=True, exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 — Build H3 grid
# ══════════════════════════════════════════════════════════════════════════════
def build_grid():
    from src.preprocessing.align import build_california_grid
    log.info("Building H3 grid (resolution=%d)…", RESOLUTION)
    gdf = build_california_grid(BBOX, RESOLUTION)
    log.info("Grid: %d cells", len(gdf))
    return gdf


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2 — Build positive cell-days and sample negatives
# ══════════════════════════════════════════════════════════════════════════════
def build_sample(grid_df):
    from src.labeling.labels import load_modis_csv

    log.info("Loading MODIS fire data…")
    fire_df = load_modis_csv(MODIS_CSV, h3_resolution=RESOLUTION, bbox=BBOX)

    # Filter to pipeline date range (include val+test for evaluation)
    fire_df = fire_df[fire_df["fire_date"] <= pd.Timestamp(TEST_END)]
    cell_ids = grid_df["cell_id"].values
    rng = np.random.default_rng(RANDOM_SEED)

    # ── Positives: expand each ignition into its 7-day prediction window ──
    log.info("Building positive cell-days (window=%d days)…", WINDOW)
    pos_records = []
    for _, row in fire_df.iterrows():
        cid, fdate = row["cell_id"], row["fire_date"]
        if cid not in set(cell_ids):
            continue
        for offset in range(WINDOW):
            forecast_date = fdate - pd.Timedelta(days=offset)
            pos_records.append({"cell_id": cid, "date": forecast_date, "label": 1})

    pos_df = pd.DataFrame(pos_records).drop_duplicates(["cell_id", "date"])
    pos_df["label"] = 1
    log.info("Positives: %d cell-days", len(pos_df))

    # ── Negatives: random sample from grid × date range ──────────────────
    n_neg = len(pos_df) * NEG_RATIO
    date_range = pd.date_range("2015-01-01", TEST_END, freq="D")

    sampled_cells = rng.choice(cell_ids, size=n_neg, replace=True)
    sampled_dates = rng.choice(date_range, size=n_neg, replace=True)

    neg_df = pd.DataFrame({"cell_id": sampled_cells, "date": pd.to_datetime(sampled_dates), "label": 0})

    # Remove any accidentally positive pairs
    pos_set = set(zip(pos_df["cell_id"], pos_df["date"]))
    mask = [
        (r.cell_id, r.date) not in pos_set
        for r in neg_df.itertuples(index=False)
    ]
    neg_df = neg_df[mask].reset_index(drop=True)
    log.info("Negatives: %d cell-days (ratio ~%.0fx)", len(neg_df), len(neg_df) / len(pos_df))

    sample = pd.concat([pos_df, neg_df], ignore_index=True)
    sample = sample.sort_values(["cell_id", "date"]).reset_index(drop=True)
    log.info("Total sample: %d rows (%.4f%% positive)", len(sample), sample["label"].mean() * 100)
    return sample


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3 — Weather features (station rolling + lagged)
# ══════════════════════════════════════════════════════════════════════════════
def build_weather_features(sample_df, grid_df):
    import h3
    from scipy.spatial import cKDTree
    from src.data.ingest import load_caweather

    log.info("Loading CAWeather station data…")
    wx = load_caweather(WEATHER_CSV)

    # ── Assign each grid cell to nearest station ──────────────────────────
    stations = wx[["station", "latitude", "longitude"]].drop_duplicates("station").reset_index(drop=True)
    station_coords = np.deg2rad(stations[["latitude", "longitude"]].values)
    tree = cKDTree(station_coords)

    cell_ids = grid_df["cell_id"].values
    cell_coords = np.deg2rad(np.array([h3.h3_to_geo(c) for c in cell_ids]))
    _, nearest_idx = tree.query(cell_coords, k=1)
    cell_station_map = pd.DataFrame({
        "cell_id": cell_ids,
        "station": stations.iloc[nearest_idx]["station"].values,
    })

    # ── Compute rolling features per station ─────────────────────────────
    log.info("Computing rolling weather features per station…")
    wx = wx.sort_values(["station", "date"])
    weather_vars = ["awnd", "prcp", "tmax", "tmin"]

    # Forward-fill gaps within each station (≤7 day gaps)
    wx[weather_vars] = wx.groupby("station")[weather_vars].ffill(limit=7)

    roll_features = []
    for sta, grp in wx.groupby("station"):
        grp = grp.set_index("date").sort_index()
        feat = pd.DataFrame(index=grp.index)
        feat["station"] = sta

        for var in weather_vars:
            s = grp[var]
            feat[f"{var}_roll{LAG_SHORT}_mean"] = s.rolling(LAG_SHORT, min_periods=1).mean()
            feat[f"{var}_roll{LAG_LONG}_mean"]  = s.rolling(LAG_LONG,  min_periods=1).mean()
            if var == "prcp":
                feat[f"prcp_roll{LAG_LONG}_sum"] = s.rolling(LAG_LONG, min_periods=1).sum()
            if var == "tmax":
                feat[f"tmax_roll{LAG_SHORT}_max"] = s.rolling(LAG_SHORT, min_periods=1).max()

        # Also keep same-day values as "current conditions"
        for var in weather_vars:
            feat[var] = grp[var]

        roll_features.append(feat.reset_index().rename(columns={"date": "date"}))

    station_features = pd.concat(roll_features, ignore_index=True)
    log.info("Station feature table: %d rows", len(station_features))

    # ── Join to sample via cell → station → date ─────────────────────────
    sample_with_station = sample_df.merge(cell_station_map, on="cell_id", how="left")
    sample_with_wx = sample_with_station.merge(
        station_features,
        on=["station", "date"],
        how="left",
    ).drop(columns=["station"])

    wx_cols = [c for c in sample_with_wx.columns if c not in ("cell_id", "date", "label")]
    missing_pct = sample_with_wx[wx_cols].isna().mean().mean() * 100
    log.info("Weather join complete. Missing values: %.1f%%", missing_pct)
    return sample_with_wx


# ══════════════════════════════════════════════════════════════════════════════
# STEP 4 — Drought (PDSI) feature
# ══════════════════════════════════════════════════════════════════════════════
def add_pdsi_feature(df):
    from src.data.ingest import load_capdsi, expand_capdsi_to_daily

    log.info("Adding PDSI drought feature…")
    pdsi_monthly = load_capdsi(PDSI_CSV)
    pdsi_daily = expand_capdsi_to_daily(pdsi_monthly)
    df = df.merge(pdsi_daily, on="date", how="left")
    df["pdsi"] = df["pdsi"].ffill().bfill()
    return df


# ══════════════════════════════════════════════════════════════════════════════
# STEP 5 — Temporal features (sin/cos seasonality)
# ══════════════════════════════════════════════════════════════════════════════
def add_temporal_features(df):
    log.info("Adding temporal features…")
    doy = df["date"].dt.dayofyear
    df["doy_sin"]   = np.sin(2 * np.pi * doy / 365.25)
    df["doy_cos"]   = np.cos(2 * np.pi * doy / 365.25)
    month = df["date"].dt.month
    df["month_sin"] = np.sin(2 * np.pi * month / 12)
    df["month_cos"] = np.cos(2 * np.pi * month / 12)
    df["year"]      = df["date"].dt.year
    return df


# ══════════════════════════════════════════════════════════════════════════════
# STEP 5b — Fire history features
# ══════════════════════════════════════════════════════════════════════════════
def build_fire_history_features(sample_df, fire_df):
    """
    Add fire-history features using only past fire data (no leakage).

    Features added
    --------------
    days_since_last_fire  : days elapsed since most recent prior fire in cell
    cell_fire_count_1yr   : fire count in this cell in past 365 days
    neighbor_fire_7d      : ring-1 H3 neighbors with fire in past 7 days
    neighbor_fire_30d     : ring-1 H3 neighbors with fire in past 30 days

    Leakage prevention: for a sample (cell_id, date), we only look at fire
    events strictly BEFORE date (i.e. fire_date < date).
    """
    import bisect
    import h3

    log.info("Building fire history features…")

    # ── Build per-cell sorted fire-date lists ────────────────────────────
    fire_dates_by_cell: dict = {}
    for row in fire_df.itertuples(index=False):
        cid = row.cell_id
        fdate = row.fire_date
        if cid not in fire_dates_by_cell:
            fire_dates_by_cell[cid] = []
        fire_dates_by_cell[cid].append(fdate)

    # Sort each cell's fire history
    for cid in fire_dates_by_cell:
        fire_dates_by_cell[cid].sort()

    # ── Build per-cell sorted neighbor-fire-date lists ───────────────────
    # Expand each fire event to all ring-1 neighbors
    neighbor_fire_dates: dict = {}
    for cid, dates in fire_dates_by_cell.items():
        neighbors = h3.k_ring(cid, 1)  # includes cid itself
        for nb in neighbors:
            if nb not in neighbor_fire_dates:
                neighbor_fire_dates[nb] = []
            neighbor_fire_dates[nb].extend(dates)

    # Sort neighbor fire lists
    for nb in neighbor_fire_dates:
        neighbor_fire_dates[nb].sort()

    # ── Compute features for each row ───────────────────────────────────
    n = len(sample_df)
    days_since_last = np.full(n, 9999.0)
    fire_count_1yr  = np.zeros(n, dtype=np.int32)
    nbr_fire_7d     = np.zeros(n, dtype=np.int32)
    nbr_fire_30d    = np.zeros(n, dtype=np.int32)

    sample_reset = sample_df.reset_index(drop=True)

    for i, row in enumerate(sample_reset.itertuples(index=False)):
        cid   = row.cell_id
        date  = row.date       # pd.Timestamp

        # ── Cell-level features ───────────────────────────────────────
        if cid in fire_dates_by_cell:
            dates_list = fire_dates_by_cell[cid]
            # Strictly before sample date: use bisect_left on date
            idx_before = bisect.bisect_left(dates_list, date)
            if idx_before > 0:
                last_fire = dates_list[idx_before - 1]
                days_since_last[i] = (date - last_fire).days

                # Count fires in past 365 days
                cutoff_1yr = date - pd.Timedelta(days=365)
                idx_1yr = bisect.bisect_left(dates_list, cutoff_1yr)
                fire_count_1yr[i] = idx_before - idx_1yr

        # ── Neighbor features ─────────────────────────────────────────
        if cid in neighbor_fire_dates:
            nb_dates = neighbor_fire_dates[cid]
            # 7-day window strictly before date
            cutoff_7d  = date - pd.Timedelta(days=7)
            idx_end    = bisect.bisect_left(nb_dates, date)
            idx_7d     = bisect.bisect_left(nb_dates, cutoff_7d)
            nbr_fire_7d[i] = idx_end - idx_7d

            # 30-day window strictly before date
            cutoff_30d = date - pd.Timedelta(days=30)
            idx_30d    = bisect.bisect_left(nb_dates, cutoff_30d)
            nbr_fire_30d[i] = idx_end - idx_30d

    result = sample_df.copy()
    result["days_since_last_fire"] = days_since_last
    result["cell_fire_count_1yr"]  = fire_count_1yr
    result["neighbor_fire_7d"]     = nbr_fire_7d
    result["neighbor_fire_30d"]    = nbr_fire_30d

    log.info(
        "Fire history features added. days_since_last_fire median=%.0f, "
        "cell_fire_count_1yr mean=%.2f, neighbor_fire_7d mean=%.3f",
        np.median(days_since_last),
        fire_count_1yr.mean(),
        nbr_fire_7d.mean(),
    )
    return result


# ══════════════════════════════════════════════════════════════════════════════
# STEP 5c — Spatial features
# ══════════════════════════════════════════════════════════════════════════════
def build_spatial_features(sample_df, grid_df):
    """
    Add spatial features derived from cell centroid coordinates.

    Features added
    --------------
    cell_lat        : centroid latitude of H3 cell
    cell_lon        : centroid longitude of H3 cell
    dist_to_coast_km: approximate Haversine distance to nearest Pacific coast
    is_inland       : 1 if dist_to_coast_km > 80, else 0
    """
    import math

    log.info("Building spatial features…")

    # ── Haversine distance (km) ──────────────────────────────────────────
    def haversine_km(lat1, lon1, lat2, lon2):
        R = 6371.0
        dlat = math.radians(lat2 - lat1)
        dlon = math.radians(lon2 - lon1)
        a = (math.sin(dlat / 2) ** 2
             + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2))
             * math.sin(dlon / 2) ** 2)
        return R * 2 * math.asin(math.sqrt(a))

    # Representative CA coast points (lon ≈ -124.5, varying lat)
    COAST_POINTS = [
        (32.5, -117.2),   # San Diego coast
        (33.7, -118.3),   # Los Angeles coast
        (35.0, -120.6),   # Morro Bay
        (36.6, -121.9),   # Monterey
        (37.8, -122.5),   # San Francisco
        (38.3, -123.0),   # Bodega Bay
        (39.5, -123.8),   # Fort Bragg
        (40.8, -124.2),   # Eureka
        (41.5, -124.1),   # Crescent City
    ]

    # ── Build cell centroid lookup from grid_df ─────────────────────────
    cell_lat_map = dict(zip(grid_df["cell_id"], grid_df["lat"]))
    cell_lon_map = dict(zip(grid_df["cell_id"], grid_df["lon"]))

    # Compute unique cells in sample
    unique_cells = sample_df["cell_id"].unique()
    lat_arr    = np.array([cell_lat_map.get(c, np.nan) for c in unique_cells])
    lon_arr    = np.array([cell_lon_map.get(c, np.nan) for c in unique_cells])

    dist_arr = np.array([
        min(haversine_km(lat, lon, clat, clon) for clat, clon in COAST_POINTS)
        if not (np.isnan(lat) or np.isnan(lon)) else np.nan
        for lat, lon in zip(lat_arr, lon_arr)
    ])

    cell_spatial = pd.DataFrame({
        "cell_id":          unique_cells,
        "cell_lat":         lat_arr,
        "cell_lon":         lon_arr,
        "dist_to_coast_km": dist_arr,
    })
    cell_spatial["is_inland"] = (cell_spatial["dist_to_coast_km"] > 80).astype(int)

    result = sample_df.merge(cell_spatial, on="cell_id", how="left")
    log.info(
        "Spatial features added. dist_to_coast_km: mean=%.1f, median=%.1f, "
        "is_inland fraction=%.2f",
        result["dist_to_coast_km"].mean(),
        result["dist_to_coast_km"].median(),
        result["is_inland"].mean(),
    )
    return result


# ══════════════════════════════════════════════════════════════════════════════
# STEP 5d — Vegetation / fuel features (LANDFIRE)
# ══════════════════════════════════════════════════════════════════════════════
def build_vegetation_features(sample_df, grid_df):
    """
    Add static LANDFIRE vegetation and fuel features per cell.

    Features added
    --------------
    evt           : Existing Vegetation Type code (categorical)
    fbfm40        : Fire Behavior Fuel Model 40 code (categorical)
    canopy_cover  : Canopy cover percentage (0–100)
    """
    from src.data.landfire import build_vegetation_features as _build_veg

    log.info("Building vegetation features (LANDFIRE)…")
    veg_df = _build_veg(grid_df, tif_dir=None)   # downloads if needed
    result = sample_df.merge(veg_df, on="cell_id", how="left")
    result["evt"]          = result["evt"].fillna(0).astype(int)
    result["fbfm40"]       = result["fbfm40"].fillna(0).astype(int)
    result["canopy_cover"] = result["canopy_cover"].fillna(0.0)
    log.info(
        "Vegetation features added. evt unique=%d, fbfm40 unique=%d, canopy_cover mean=%.1f",
        result["evt"].nunique(), result["fbfm40"].nunique(), result["canopy_cover"].mean(),
    )
    return result


# ══════════════════════════════════════════════════════════════════════════════
# STEP 5e — Weather-derived features
# ══════════════════════════════════════════════════════════════════════════════
def build_derived_weather_features(df):
    """
    Compute derived features from existing weather columns.

    Features added
    --------------
    temp_range          : tmax - tmin (diurnal temperature range)
    vpd_proxy           : tmax * 0.6 (when tmin < tmax*0.3) else tmax - tmin
    hot_dry_windy       : composite fire-weather flag (1 = dangerous conditions)
    consecutive_dry_days: proxy for dry streak based on prcp_roll30_sum
    """
    log.info("Building derived weather features…")

    df = df.copy()

    # ── temp_range ───────────────────────────────────────────────────────
    if "tmax" in df.columns and "tmin" in df.columns:
        df["temp_range"] = df["tmax"] - df["tmin"]
    else:
        df["temp_range"] = np.nan

    # ── vpd_proxy ────────────────────────────────────────────────────────
    if "tmax" in df.columns and "tmin" in df.columns:
        low_humidity_mask = df["tmin"] < (df["tmax"] * 0.3)
        df["vpd_proxy"] = np.where(
            low_humidity_mask,
            df["tmax"] * 0.6,
            df["tmax"] - df["tmin"],
        )
    else:
        df["vpd_proxy"] = np.nan

    # ── hot_dry_windy ────────────────────────────────────────────────────
    # Composite flag: (hot rolling tmax) AND (dry rolling prcp) AND (windy rolling awnd)
    hot_col  = f"tmax_roll{LAG_SHORT}_mean"
    prcp_col = f"prcp_roll{LAG_LONG}_sum"
    wind_col = f"awnd_roll{LAG_SHORT}_mean"

    if all(c in df.columns for c in (hot_col, prcp_col, wind_col)):
        hot_thresh  = df[hot_col].quantile(0.75)
        wind_thresh = df[wind_col].quantile(0.60)
        hot   = (df[hot_col]  > hot_thresh).astype(int)
        dry   = (df[prcp_col] < 5.0).astype(int)
        windy = (df[wind_col] > wind_thresh).astype(int)
        df["hot_dry_windy"] = hot * dry * windy
    else:
        df["hot_dry_windy"] = 0

    # ── consecutive_dry_days (proxy) ─────────────────────────────────────
    if prcp_col in df.columns:
        df["consecutive_dry_days"] = (df[prcp_col] < 5.0).astype(int) * 30
    else:
        df["consecutive_dry_days"] = 0

    new_cols = ["temp_range", "vpd_proxy", "hot_dry_windy", "consecutive_dry_days"]
    log.info(
        "Derived weather features added: %s",
        {c: f"{df[c].mean():.3f}" for c in new_cols if c in df.columns},
    )
    return df


# ══════════════════════════════════════════════════════════════════════════════
# STEP 6 — Train / Val / Test split + train model
# ══════════════════════════════════════════════════════════════════════════════
def train(df):
    import lightgbm as lgb
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline

    FEATURE_COLS = [c for c in df.columns if c not in ("cell_id", "date", "label")]
    log.info("Feature columns (%d): %s", len(FEATURE_COLS), FEATURE_COLS)

    train_df = df[df["date"] <= pd.Timestamp(TRAIN_END)]
    val_df   = df[(df["date"] >= pd.Timestamp(VAL_START)) & (df["date"] <= pd.Timestamp(VAL_END))]
    test_df  = df[(df["date"] >= pd.Timestamp(TEST_START)) & (df["date"] <= pd.Timestamp(TEST_END))]

    log.info("Split sizes — train: %d | val: %d | test: %d", len(train_df), len(val_df), len(test_df))

    X_train = train_df[FEATURE_COLS].fillna(0)
    y_train = train_df["label"]
    X_val   = val_df[FEATURE_COLS].fillna(0)
    y_val   = val_df["label"]
    X_test  = test_df[FEATURE_COLS].fillna(0)
    y_test  = test_df["label"]

    pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
    log.info("scale_pos_weight = %.1f", pos_weight)

    # ── Baseline: Logistic Regression ────────────────────────────────────
    log.info("Training Logistic Regression baseline…")
    lr = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(C=0.1, max_iter=1000, class_weight="balanced", solver="saga")),
    ])
    lr.fit(X_train, y_train)
    lr_proba = lr.predict_proba(X_test)[:, 1]

    # ── Primary: LightGBM ────────────────────────────────────────────────
    log.info("Training LightGBM…")
    lgb_params = {
        **CONFIG["models"]["lgbm"],
        "scale_pos_weight": pos_weight,
        "random_state": RANDOM_SEED,
    }
    lgb_params.pop("n_jobs", None)

    model = lgb.LGBMClassifier(**lgb_params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[
            lgb.early_stopping(CONFIG["models"]["training"]["early_stopping_rounds"], verbose=False),
            lgb.log_evaluation(period=50),
        ],
    )
    lgb_proba = model.predict_proba(X_test)[:, 1]

    return model, lr, lgb_proba, lr_proba, y_test, FEATURE_COLS


# ══════════════════════════════════════════════════════════════════════════════
# STEP 7 — Evaluate
# ══════════════════════════════════════════════════════════════════════════════
def evaluate(lgb_proba, lr_proba, y_test, model, feature_cols):
    from src.evaluation.metrics import evaluate as eval_metrics

    log.info("=== LightGBM Results ===")
    lgb_metrics = eval_metrics(y_test.values, lgb_proba, top_k=CONFIG["evaluation"]["top_k"])

    log.info("=== Logistic Regression Baseline ===")
    lr_metrics = eval_metrics(y_test.values, lr_proba, top_k=CONFIG["evaluation"]["top_k"])

    # Feature importance
    import pandas as pd
    fi = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False)
    log.info("Top 10 features:\n%s", fi.head(10).to_string())

    # Save feature importance
    fi.to_csv(OUT_DIR / "reports" / "feature_importance.csv")

    return lgb_metrics


# ══════════════════════════════════════════════════════════════════════════════
# STEP 8 — Save model
# ══════════════════════════════════════════════════════════════════════════════
def save_model(model, lr_model, lgb_proba, lr_proba, y_test):
    import pickle
    model_path = OUT_DIR / "models" / "lgbm_model.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    log.info("Model saved → %s", model_path)

    lr_path = OUT_DIR / "models" / "lr_model.pkl"
    with open(lr_path, "wb") as f:
        pickle.dump(lr_model, f)
    log.info("LR model saved → %s", lr_path)

    preds_path = OUT_DIR / "reports" / "test_predictions.parquet"
    pd.DataFrame({
        "y_test": y_test.values,
        "lgb_proba": lgb_proba,
        "lr_proba": lr_proba,
    }).to_parquet(preds_path, index=False)
    log.info("Test predictions saved → %s", preds_path)


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import warnings; warnings.filterwarnings("ignore")

    log.info("━━━ Wildfire Ignition Model — Training Pipeline ━━━")

    grid_df  = build_grid()

    # Build sample and load fire_df (reuse load_modis_csv for fire history)
    from src.labeling.labels import load_modis_csv
    fire_df = load_modis_csv(MODIS_CSV, h3_resolution=RESOLUTION, bbox=BBOX)
    fire_df = fire_df[fire_df["fire_date"] <= pd.Timestamp(TEST_END)]

    sample   = build_sample(grid_df)
    sample   = build_weather_features(sample, grid_df)
    sample   = add_pdsi_feature(sample)
    sample   = add_temporal_features(sample)

    n_features_before = len([c for c in sample.columns if c not in ("cell_id", "date", "label")])
    log.info("Feature count BEFORE new features: %d", n_features_before)

    # Phase 1 — new features
    sample   = build_fire_history_features(sample, fire_df)
    sample   = build_spatial_features(sample, grid_df)
    sample   = build_derived_weather_features(sample)
    sample   = build_vegetation_features(sample, grid_df)

    n_features_after = len([c for c in sample.columns if c not in ("cell_id", "date", "label")])
    log.info("Feature count AFTER new features: %d (+%d)", n_features_after, n_features_after - n_features_before)

    model, lr, lgb_proba, lr_proba, y_test, feat_cols = train(sample)
    metrics = evaluate(lgb_proba, lr_proba, y_test, model, feat_cols)
    save_model(model, lr, lgb_proba, lr_proba, y_test)

    log.info("━━━ Done ━━━")
