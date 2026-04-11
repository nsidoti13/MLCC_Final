"""
src/features/engineer.py
========================
Build the full feature matrix from multiple input DataFrames.

Feature groups
--------------
1. Forecast features       — Max/min/sum of 7-day weather forecast fields
2. Lagged rolling features — Rolling means/sums over past 7 and 30 days
3. Static features         — Terrain, vegetation, fuels (time-invariant)
4. Human features          — Roads, powerlines, population density
5. Spatial features        — Neighbor fire counts, dryness index, wind alignment
6. Temporal features       — sin/cos encoding of day-of-year and month

All features are joined onto the (cell_id, date) panel DataFrame so that
the final output has exactly one row per (cell_id, date) pair.
"""

from __future__ import annotations

import logging
from typing import List, Optional

import h3
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def build_features(
    weather_df: pd.DataFrame,
    static_df: pd.DataFrame,
    fire_history_df: pd.DataFrame,
    grid_df: pd.DataFrame,
    human_df: Optional[pd.DataFrame] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    lag_short: int = 7,
    lag_long: int = 30,
    h3_resolution: int = 6,
) -> pd.DataFrame:
    """
    Build the complete feature DataFrame for model training or inference.

    Parameters
    ----------
    weather_df:
        Daily weather panel indexed by (cell_id, date).  Expected columns:
        tmp2m, rh2m, wnd10m, apcp.
    static_df:
        Static cell attributes (one row per cell_id).  Expected columns:
        elevation_m, slope_deg, aspect_deg, fuel_model, canopy_cover_pct,
        canopy_bulk_density, canopy_base_height, vegetation_type.
    fire_history_df:
        Historical fire records with columns cell_id and fire_date.
        Used to compute spatial neighbor fire features.
    grid_df:
        Base H3 grid with column cell_id (and optionally lat, lon).
    human_df:
        Optional human-infrastructure features (one row per cell_id).
        Expected columns: road_density_km_per_km2, dist_to_powerline_km,
        pop_density_per_km2.
    start_date:
        First date to include in output (ISO string).  Defaults to earliest
        date in weather_df.
    end_date:
        Last date to include in output (ISO string).  Defaults to latest date.
    lag_short:
        Short rolling window in days (default 7).
    lag_long:
        Long rolling window in days (default 30).
    h3_resolution:
        H3 resolution used throughout the pipeline (default 6).

    Returns
    -------
    pd.DataFrame
        Wide feature DataFrame with columns:
        cell_id, date, <all feature columns>.
        No NaN values (any remaining are filled with column median).
    """
    # ------------------------------------------------------------------
    # 0. Build the (cell_id, date) panel skeleton
    # ------------------------------------------------------------------
    panel = _build_panel(grid_df, weather_df, start_date, end_date)

    # ------------------------------------------------------------------
    # 1. Forecast features
    # ------------------------------------------------------------------
    forecast_feats = _compute_forecast_features(weather_df)
    panel = panel.merge(forecast_feats, on=["cell_id", "date"], how="left")

    # ------------------------------------------------------------------
    # 2. Lagged rolling features
    # ------------------------------------------------------------------
    lagged_feats = _compute_lagged_features(weather_df, lag_short, lag_long)
    panel = panel.merge(lagged_feats, on=["cell_id", "date"], how="left")

    # ------------------------------------------------------------------
    # 3. Static features
    # ------------------------------------------------------------------
    static_clean = _prepare_static_features(static_df)
    panel = panel.merge(static_clean, on="cell_id", how="left")

    # ------------------------------------------------------------------
    # 4. Human features
    # ------------------------------------------------------------------
    if human_df is not None:
        human_clean = _prepare_human_features(human_df)
        panel = panel.merge(human_clean, on="cell_id", how="left")
    else:
        # Insert zero-filled placeholders so downstream code is consistent
        for col in ["road_density_km_per_km2", "dist_to_powerline_km", "pop_density_per_km2"]:
            panel[col] = 0.0
        logger.warning("build_features: human_df not provided; human feature columns set to 0.")

    # ------------------------------------------------------------------
    # 5. Spatial neighbor features
    # ------------------------------------------------------------------
    spatial_feats = _compute_spatial_features(
        panel, fire_history_df, h3_resolution
    )
    panel = panel.merge(spatial_feats, on=["cell_id", "date"], how="left")

    # ------------------------------------------------------------------
    # 6. Temporal (seasonality) features
    # ------------------------------------------------------------------
    panel = _add_temporal_features(panel)

    # ------------------------------------------------------------------
    # 7. Fill any remaining NaN with column median
    # ------------------------------------------------------------------
    numeric_cols = panel.select_dtypes(include=[np.number]).columns.tolist()
    for col in numeric_cols:
        if panel[col].isnull().any():
            panel[col] = panel[col].fillna(panel[col].median())

    panel = panel.sort_values(["cell_id", "date"]).reset_index(drop=True)
    logger.info(
        "build_features: final feature matrix shape = %s.", panel.shape
    )
    return panel


# ---------------------------------------------------------------------------
# 0. Panel skeleton
# ---------------------------------------------------------------------------


def _build_panel(
    grid_df: pd.DataFrame,
    weather_df: pd.DataFrame,
    start_date: Optional[str],
    end_date: Optional[str],
) -> pd.DataFrame:
    """Create the full (cell_id, date) panel from the grid and date range."""
    cell_ids = grid_df["cell_id"].unique().tolist()

    weather_df = weather_df.copy()
    weather_df["date"] = pd.to_datetime(weather_df["date"])

    sd = pd.to_datetime(start_date) if start_date else weather_df["date"].min()
    ed = pd.to_datetime(end_date) if end_date else weather_df["date"].max()

    date_range = pd.date_range(start=sd, end=ed, freq="D")
    idx = pd.MultiIndex.from_product([cell_ids, date_range], names=["cell_id", "date"])
    panel = pd.DataFrame(index=idx).reset_index()
    logger.info(
        "_build_panel: %d cells × %d dates = %d rows.",
        len(cell_ids),
        len(date_range),
        len(panel),
    )
    return panel


# ---------------------------------------------------------------------------
# 1. Forecast features
# ---------------------------------------------------------------------------


def _compute_forecast_features(weather_df: pd.DataFrame) -> pd.DataFrame:
    """
    Summarise the 7-day forecast window into scalar features per (cell_id, date).

    We assume weather_df has one row per (cell_id, date) with the forecast
    values for that day already aligned to the cell.  The "forecast" features
    are simply the values for the current and next 6 days aggregated to max/min/sum.

    In production these would come from CFSv2 horizon-specific fields.  Here
    we compute a rolling forward window of the observed weather as a proxy.

    Output columns:
        tmp2m_max_7d, rh2m_min_7d, wnd10m_max_7d, apcp_sum_7d
    """
    df = weather_df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["cell_id", "date"])

    cols_exist = {
        "tmp2m": "tmp2m_max_7d",
        "rh2m": "rh2m_min_7d",
        "wnd10m": "wnd10m_max_7d",
        "apcp": "apcp_sum_7d",
    }
    # Keep only columns that exist
    available = {k: v for k, v in cols_exist.items() if k in df.columns}

    if not available:
        logger.warning("_compute_forecast_features: none of expected columns found in weather_df.")
        return df[["cell_id", "date"]].drop_duplicates()

    result_parts = []
    for cell_id, group in df.groupby("cell_id"):
        g = group.set_index("date").sort_index()
        part = pd.DataFrame(index=g.index)
        part.index.name = "date"

        if "tmp2m" in available:
            # Max temperature over next 7 days (lead = next 7 days)
            part["tmp2m_max_7d"] = g["tmp2m"].shift(-6).rolling(7, min_periods=1).max()
        if "rh2m" in available:
            part["rh2m_min_7d"] = g["rh2m"].shift(-6).rolling(7, min_periods=1).min()
        if "wnd10m" in available:
            part["wnd10m_max_7d"] = g["wnd10m"].shift(-6).rolling(7, min_periods=1).max()
        if "apcp" in available:
            part["apcp_sum_7d"] = g["apcp"].shift(-6).rolling(7, min_periods=1).sum()

        part["cell_id"] = cell_id
        result_parts.append(part.reset_index())

    return pd.concat(result_parts, ignore_index=True)


# ---------------------------------------------------------------------------
# 2. Lagged rolling features
# ---------------------------------------------------------------------------


def _compute_lagged_features(
    weather_df: pd.DataFrame,
    lag_short: int = 7,
    lag_long: int = 30,
) -> pd.DataFrame:
    """
    Compute past rolling weather statistics to capture dryness / heat stress.

    Rolling windows look only BACKWARD (no future leakage):
    - Short window (default 7 days): recent conditions
    - Long window (default 30 days): accumulated drought / seasonal dryness

    Output columns:
        tmp2m_roll7_mean, tmp2m_roll30_mean,
        rh2m_roll7_mean,  rh2m_roll30_mean,
        apcp_roll30_sum,  wnd10m_roll7_mean
    """
    df = weather_df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["cell_id", "date"])

    result_parts = []
    for cell_id, group in df.groupby("cell_id"):
        g = group.set_index("date").sort_index()
        part = pd.DataFrame(index=g.index)
        part.index.name = "date"

        if "tmp2m" in g.columns:
            # shift(1) ensures current day is not included (strictly past)
            part["tmp2m_roll7_mean"] = (
                g["tmp2m"].shift(1).rolling(lag_short, min_periods=1).mean()
            )
            part["tmp2m_roll30_mean"] = (
                g["tmp2m"].shift(1).rolling(lag_long, min_periods=1).mean()
            )

        if "rh2m" in g.columns:
            part["rh2m_roll7_mean"] = (
                g["rh2m"].shift(1).rolling(lag_short, min_periods=1).mean()
            )
            part["rh2m_roll30_mean"] = (
                g["rh2m"].shift(1).rolling(lag_long, min_periods=1).mean()
            )

        if "apcp" in g.columns:
            part["apcp_roll30_sum"] = (
                g["apcp"].shift(1).rolling(lag_long, min_periods=1).sum()
            )

        if "wnd10m" in g.columns:
            part["wnd10m_roll7_mean"] = (
                g["wnd10m"].shift(1).rolling(lag_short, min_periods=1).mean()
            )

        part["cell_id"] = cell_id
        result_parts.append(part.reset_index())

    return pd.concat(result_parts, ignore_index=True)


# ---------------------------------------------------------------------------
# 3. Static features
# ---------------------------------------------------------------------------


def _prepare_static_features(static_df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare static (time-invariant) features for merging onto the panel.

    Encodes categorical fuel_model and vegetation_type as integer codes.
    Returns one row per cell_id.
    """
    df = static_df.copy()

    expected_cols = [
        "cell_id", "elevation_m", "slope_deg", "aspect_deg",
        "fuel_model", "canopy_cover_pct", "canopy_bulk_density",
        "canopy_base_height", "vegetation_type",
    ]

    # Keep only columns that are present
    available = [c for c in expected_cols if c in df.columns]
    df = df[available].copy()

    # Encode categoricals as integers
    for cat_col in ["fuel_model", "vegetation_type"]:
        if cat_col in df.columns:
            df[cat_col] = pd.Categorical(df[cat_col]).codes

    # Aspect: encode as sin/cos to handle the 0°/360° discontinuity
    if "aspect_deg" in df.columns:
        aspect_rad = np.deg2rad(df["aspect_deg"])
        df["aspect_sin"] = np.sin(aspect_rad)
        df["aspect_cos"] = np.cos(aspect_rad)
        df = df.drop(columns=["aspect_deg"])

    return df.drop_duplicates(subset=["cell_id"])


# ---------------------------------------------------------------------------
# 4. Human features
# ---------------------------------------------------------------------------


def _prepare_human_features(human_df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare human-infrastructure features for merging onto the panel.

    Applies log1p transform to skewed density columns.
    Returns one row per cell_id.
    """
    df = human_df.copy()

    for skewed_col in ["road_density_km_per_km2", "pop_density_per_km2"]:
        if skewed_col in df.columns:
            df[skewed_col] = np.log1p(df[skewed_col])

    return df.drop_duplicates(subset=["cell_id"])


# ---------------------------------------------------------------------------
# 5. Spatial neighbor features
# ---------------------------------------------------------------------------


def _compute_spatial_features(
    panel: pd.DataFrame,
    fire_history_df: pd.DataFrame,
    h3_resolution: int = 6,
    neighbor_rings: int = 1,
    fire_lookback_days: int = 7,
) -> pd.DataFrame:
    """
    Compute spatial neighbor features for each (cell_id, date).

    Features computed:
    - ``neighbor_fire_count_7d``: Number of H3 neighbors that had a fire in the
      past ``fire_lookback_days`` days.  Only *past* fires are counted
      (date < current date), so there is no future leakage.
    - ``neighbor_dryness_index``: Placeholder — in production this would be the
      average KBDI (Keetch-Byram Drought Index) across neighbors.
    - ``wind_alignment_score``: Placeholder — in production this encodes how
      well the prevailing wind direction aligns with the nearest ignition source.

    Parameters
    ----------
    panel:
        (cell_id, date) panel DataFrame.
    fire_history_df:
        Historical fire events with cell_id and fire_date columns.
    h3_resolution:
        H3 grid resolution.
    neighbor_rings:
        Number of H3 k-rings to include as neighbors (default 1 = direct neighbors).
    fire_lookback_days:
        How many past days to scan for neighbor fires.
    """
    fire = fire_history_df.copy()
    fire["fire_date"] = pd.to_datetime(fire["fire_date"])

    # Pre-build neighbor lookup for each unique cell
    unique_cells = panel["cell_id"].unique().tolist()
    neighbor_map: dict = {}
    for cid in unique_cells:
        try:
            neighbors = h3.k_ring(cid, neighbor_rings) - {cid}
        except Exception:
            neighbors = set()
        neighbor_map[cid] = neighbors

    # Build a set of (cell_id, date) → fire for fast lookup
    # We need: for each (cell_id, date), count fires in neighbor cells
    # in [date - lookback, date - 1]
    fire_set: set = set(zip(fire["cell_id"].tolist(), fire["fire_date"].dt.date.tolist()))

    records = []
    for _, row in panel[["cell_id", "date"]].drop_duplicates().iterrows():
        cid = row["cell_id"]
        current_date = pd.Timestamp(row["date"])

        neighbors = neighbor_map.get(cid, set())
        count = 0
        for nid in neighbors:
            for offset in range(1, fire_lookback_days + 1):
                check_date = (current_date - pd.Timedelta(days=offset)).date()
                if (nid, check_date) in fire_set:
                    count += 1
                    break  # count each neighbor at most once

        records.append({
            "cell_id": cid,
            "date": current_date,
            "neighbor_fire_count_7d": count,
            "neighbor_dryness_index": 0.0,   # placeholder
            "wind_alignment_score": 0.0,      # placeholder
        })

    result = pd.DataFrame(records)
    logger.info("_compute_spatial_features: computed for %d (cell, date) pairs.", len(result))
    return result


# ---------------------------------------------------------------------------
# 6. Temporal (seasonality) features
# ---------------------------------------------------------------------------


def _add_temporal_features(panel: pd.DataFrame) -> pd.DataFrame:
    """
    Add cyclic sin/cos encodings of day-of-year and month.

    Cyclic encoding wraps the calendar so that day 1 and day 365 are adjacent
    in feature space — preventing the model from treating the year boundary as
    a large discontinuity.

    Columns added:
        doy_sin, doy_cos   (day of year, period = 365.25)
        month_sin, month_cos (month, period = 12)
        week_of_year       (integer, for tree-based models)
    """
    df = panel.copy()
    df["date"] = pd.to_datetime(df["date"])

    doy = df["date"].dt.day_of_year.astype(float)
    month = df["date"].dt.month.astype(float)

    df["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    df["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
    df["month_sin"] = np.sin(2 * np.pi * month / 12.0)
    df["month_cos"] = np.cos(2 * np.pi * month / 12.0)
    df["week_of_year"] = df["date"].dt.isocalendar().week.astype(int)

    return df


# ---------------------------------------------------------------------------
# Feature list helper
# ---------------------------------------------------------------------------


def get_feature_columns(df: pd.DataFrame) -> List[str]:
    """
    Return the list of feature columns in a built feature DataFrame.

    Excludes metadata columns (cell_id, date, label).
    """
    exclude = {"cell_id", "date", "label"}
    return [c for c in df.columns if c not in exclude]
