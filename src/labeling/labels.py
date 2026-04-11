"""
src/labeling/labels.py
======================
Build the (cell_id, date, label) target DataFrame used for model training.

Leakage-prevention design
--------------------------
The central risk in this pipeline is **temporal data leakage**: assigning a
label that depends on information the model could not have had at prediction
time.

We follow these rules strictly:

1. **Forward-only labelling window**: For a row with date ``t``, the label is 1
   if *any* ignition occurs in [t, t + window_days - 1].  The label for date
   ``t`` is NEVER influenced by events before ``t`` or after ``t + window - 1``.

2. **First ignition only**: Once a cell burns, subsequent spread events in
   overlapping windows are NOT counted as new ignitions.  We record only the
   first detected ignition per fire event.

3. **Grid generation precedes labeling**: The grid of (cell_id, date) pairs is
   built independently of the fire data.  Cells that never burn still appear
   in the dataset as negative examples—there is no filtering on outcomes.

4. **No leaking future fire counts into features**: The labeling function does
   not add any fire-count features.  Feature engineering is a separate stage.

5. **Date fence**: When splitting by year, the label window at the train/val
   boundary can spill into the validation period.  Callers must trim the last
   ``window_days - 1`` rows of the training split before fitting.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# MODIS CSV ingestion
# ---------------------------------------------------------------------------


def load_modis_csv(
    csv_path: str | Path,
    h3_resolution: int = 6,
    bbox: Optional[dict] = None,
    confidence_threshold: int = 30,
    vegetation_only: bool = True,
) -> pd.DataFrame:
    """
    Load a MODIS active-fire CSV (FIRMS format) and convert point detections
    to H3 cell IDs, ready for use with ``build_label_dataframe``.

    Expected CSV columns (standard FIRMS MODIS C6.1 format)
    --------------------------------------------------------
    latitude    — WGS-84 latitude of fire pixel centre
    longitude   — WGS-84 longitude of fire pixel centre
    acq_date    — Acquisition date (YYYY-MM-DD)
    confidence  — Detection confidence 0–100
    type        — Fire type: 0=vegetation, 1=volcano, 2=static, 3=offshore

    Parameters
    ----------
    csv_path:
        Path to the MODIS fire CSV (e.g. ``data/raw/modis/MODISFireData.csv``).
    h3_resolution:
        H3 grid resolution to use when converting lat/lon to cell IDs.
        Should match the resolution used throughout the pipeline (default 6).
    bbox:
        Spatial filter as ``{lon_min, lat_min, lon_max, lat_max}``.
        Defaults to California bounding box if None.
    confidence_threshold:
        Minimum MODIS confidence (0–100) to retain a detection (default 30).
        MODIS standard: low <30, nominal 30–80, high >80.
    vegetation_only:
        If True (default), keep only type=0 (presumed vegetation fires).
        Excludes volcanoes, static sources, and offshore detections.

    Returns
    -------
    pd.DataFrame
        Columns: ``cell_id`` (H3 string), ``fire_date`` (datetime64).
        Deduplicated to one row per (cell_id, fire_date).
    """
    try:
        import h3
    except ImportError as exc:
        raise ImportError("h3 is required: pip install h3") from exc

    _DEFAULT_BBOX = {"lon_min": -124.48, "lat_min": 32.53, "lon_max": -114.13, "lat_max": 42.01}
    if bbox is None:
        bbox = _DEFAULT_BBOX

    logger.info("Loading MODIS CSV: %s", csv_path)
    df = pd.read_csv(csv_path, parse_dates=["acq_date"])

    # Spatial filter
    mask = (
        df["latitude"].between(bbox["lat_min"], bbox["lat_max"])
        & df["longitude"].between(bbox["lon_min"], bbox["lon_max"])
    )
    df = df[mask].copy()
    logger.info("After bbox filter: %d detections", len(df))

    # Vegetation fires only
    if vegetation_only:
        df = df[df["type"] == 0].copy()
        logger.info("After vegetation filter (type=0): %d detections", len(df))

    # Confidence filter
    df = df[df["confidence"] >= confidence_threshold].copy()
    logger.info("After confidence>=%d filter: %d detections", confidence_threshold, len(df))

    # Convert lat/lon → H3 cell ID
    df["cell_id"] = df.apply(
        lambda row: h3.geo_to_h3(row["latitude"], row["longitude"], h3_resolution),
        axis=1,
    )
    df = df.rename(columns={"acq_date": "fire_date"})[["cell_id", "fire_date"]]

    # Deduplicate: multiple MODIS pixels can fall in the same H3 cell on the same day
    before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    logger.info(
        "After H3 deduplication: %d → %d ignition events (resolution %d).",
        before, len(df), h3_resolution,
    )

    return df


# ---------------------------------------------------------------------------
# Main public function
# ---------------------------------------------------------------------------


def build_label_dataframe(
    fire_df: pd.DataFrame,
    grid_df: pd.DataFrame,
    window: int = 7,
    date_col: str = "date",
    cell_col: str = "cell_id",
    fire_date_col: str = "fire_date",
    fire_cell_col: str = "cell_id",
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> pd.DataFrame:
    """
    Create the (cell_id, date, label) target DataFrame.

    For every (cell_id, date) pair in the full spatio-temporal grid, the label
    is 1 if the cell has at least one *first* ignition in the half-open
    interval [date, date + window_days).

    Only *first ignitions* are counted.  If a cell has two fire records within
    the same 7-day window, it counts as a single positive.  Spread fires
    (ignitions in a cell that already burned within the last 30 days) are
    excluded; this approximates "new ignition vs. active spread".

    Parameters
    ----------
    fire_df:
        DataFrame of observed fire ignitions.  Must contain at least:
        - ``fire_date_col``: date of detected ignition (coerced to datetime).
        - ``fire_cell_col``: H3 cell ID where the ignition occurred.
    grid_df:
        GeoDataFrame (or plain DataFrame) of all burnable cells.  Must have a
        ``cell_id`` column.
    window:
        Forward prediction window in days (default 7).
        Label = 1 iff ignition in [t, t + window - 1].
    date_col:
        Output column name for the forecast date.
    cell_col:
        Output column name for the cell ID.
    fire_date_col:
        Column in ``fire_df`` containing ignition dates.
    fire_cell_col:
        Column in ``fire_df`` containing cell IDs.
    start_date:
        First date to include in the output grid (ISO string).
        Defaults to the earliest fire date minus 30 days.
    end_date:
        Last date to include in the output grid (ISO string).
        Defaults to the latest fire date.

    Returns
    -------
    pd.DataFrame
        Columns: cell_id, date, label (0/1).
        One row per (cell_id, date) combination.
        Sorted by cell_id then date.

    Raises
    ------
    ValueError
        If required columns are missing from ``fire_df`` or ``grid_df``.
    """
    # ------------------------------------------------------------------
    # Validate inputs
    # ------------------------------------------------------------------
    required_fire_cols = {fire_date_col, fire_cell_col}
    missing_fire = required_fire_cols - set(fire_df.columns)
    if missing_fire:
        raise ValueError(f"build_label_dataframe: fire_df missing columns {missing_fire}")

    if cell_col not in grid_df.columns:
        raise ValueError(f"build_label_dataframe: grid_df missing column '{cell_col}'")

    # ------------------------------------------------------------------
    # Prepare fire data
    # ------------------------------------------------------------------
    fire = fire_df[[fire_date_col, fire_cell_col]].copy()
    fire[fire_date_col] = pd.to_datetime(fire[fire_date_col])
    fire = fire.rename(columns={fire_date_col: "fire_date", fire_cell_col: "cell_id"})
    fire = fire.sort_values(["cell_id", "fire_date"]).reset_index(drop=True)

    # ------------------------------------------------------------------
    # Remove spread events: keep only *first* ignition per cell per
    # 30-day cooldown window.
    #
    # Leakage note: we use past fire history to filter spread.
    # This is safe because we are looking BACKWARD (at history) not forward.
    # ------------------------------------------------------------------
    fire = _filter_first_ignitions(fire, cooldown_days=30)

    # ------------------------------------------------------------------
    # Build the full (cell_id, date) spatio-temporal grid
    # ------------------------------------------------------------------
    cell_ids = grid_df[cell_col].unique().tolist()

    if start_date is None:
        start_date = (fire["fire_date"].min() - pd.Timedelta(days=30)).strftime("%Y-%m-%d")
    if end_date is None:
        end_date = fire["fire_date"].max().strftime("%Y-%m-%d")

    date_range = pd.date_range(start=start_date, end=end_date, freq="D")

    logger.info(
        "Building label grid: %d cells × %d dates = %d rows.",
        len(cell_ids),
        len(date_range),
        len(cell_ids) * len(date_range),
    )

    # Use a MultiIndex for efficiency — avoids a cartesian merge
    idx = pd.MultiIndex.from_product([cell_ids, date_range], names=[cell_col, date_col])
    label_df = pd.DataFrame(index=idx).reset_index()
    label_df["label"] = 0

    # ------------------------------------------------------------------
    # Assign labels (vectorised forward-window lookup)
    #
    # For each ignition event (cell, fire_date), set label=1 for all dates
    # t such that fire_date ∈ [t, t + window - 1], i.e.
    #     fire_date - (window - 1) ≤ t ≤ fire_date
    #
    # Leakage note: we compute the *earliest forecast date* that would
    # legitimately predict this ignition.  A forecast on date t can see the
    # ignition only if the ignition occurs within the future window [t, t+6].
    # Setting label=1 for t ∈ [fire_date - 6, fire_date] is correct.
    # No information from after the forecast date leaks into the label.
    # ------------------------------------------------------------------
    label_df = _assign_labels_vectorised(label_df, fire, window, cell_col, date_col)

    # Sort for deterministic output
    label_df = label_df.sort_values([cell_col, date_col]).reset_index(drop=True)

    pos_rate = label_df["label"].mean()
    logger.info(
        "Label construction complete: %d rows, %.4f%% positive rate.",
        len(label_df),
        pos_rate * 100,
    )
    return label_df


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _filter_first_ignitions(
    fire: pd.DataFrame,
    cooldown_days: int = 30,
) -> pd.DataFrame:
    """
    Retain only *first* ignitions per cell by applying a cooldown window.

    If a cell had a detected fire within the past ``cooldown_days`` days, the
    subsequent detection is treated as fire *spread* and dropped.

    This is computed using only past information (for each row, we look at
    earlier rows for the same cell), so there is no forward leakage.

    Parameters
    ----------
    fire:
        DataFrame with columns ``cell_id``, ``fire_date``, sorted by
        (cell_id, fire_date).
    cooldown_days:
        Minimum number of days between independent ignitions.

    Returns
    -------
    pd.DataFrame
        Filtered fire DataFrame with spread events removed.
    """
    fire = fire.sort_values(["cell_id", "fire_date"]).copy()

    keep_flags = []
    last_ignition: dict = {}  # cell_id → last retained fire_date

    for _, row in fire.iterrows():
        cid = row["cell_id"]
        fdate = row["fire_date"]

        if cid not in last_ignition:
            # First ever ignition in this cell — always keep
            keep_flags.append(True)
            last_ignition[cid] = fdate
        else:
            days_since = (fdate - last_ignition[cid]).days
            if days_since >= cooldown_days:
                keep_flags.append(True)
                last_ignition[cid] = fdate
            else:
                # Within cooldown — treat as spread, not new ignition
                keep_flags.append(False)

    filtered = fire[keep_flags].reset_index(drop=True)
    dropped = len(fire) - len(filtered)
    logger.info(
        "_filter_first_ignitions: removed %d spread events (%d → %d fire records).",
        dropped,
        len(fire),
        len(filtered),
    )
    return filtered


def _assign_labels_vectorised(
    label_df: pd.DataFrame,
    fire: pd.DataFrame,
    window: int,
    cell_col: str,
    date_col: str,
) -> pd.DataFrame:
    """
    Vectorised label assignment using a merge-asof approach.

    For each ignition event at (cell_id, fire_date), any forecast date t in
    [fire_date - (window-1), fire_date] should receive label=1.

    We avoid iterating over every cell-date pair by:
    1. Building a per-cell set of ignition dates.
    2. For each ignition, marking the [t_min, t_max] window of forecast dates.
    3. Using a merge on (cell_id, date) to stamp label=1.

    Parameters
    ----------
    label_df:
        Full (cell_id, date, label=0) DataFrame.
    fire:
        Filtered fire DataFrame with columns cell_id, fire_date.
    window:
        Prediction window in days.
    cell_col, date_col:
        Column names in label_df.

    Returns
    -------
    pd.DataFrame
        label_df with label column updated.
    """
    if fire.empty:
        logger.warning("_assign_labels_vectorised: fire DataFrame is empty; all labels = 0.")
        return label_df

    # Explode each ignition into all forecast dates that would predict it
    # forecast date t predicts ignition on fire_date iff:
    #   t ≤ fire_date ≤ t + window - 1
    #   ⟺  fire_date - (window-1) ≤ t ≤ fire_date
    positive_records = []
    for _, row in fire.iterrows():
        cid = row["cell_id"]
        fire_date = row["fire_date"]
        # Range of forecast dates [t_min, fire_date] that legitimately predict
        # this ignition.  NOTE: t must be ≤ fire_date (no future leakage).
        t_min = fire_date - pd.Timedelta(days=window - 1)
        t_max = fire_date  # inclusive

        forecast_dates = pd.date_range(start=t_min, end=t_max, freq="D")
        for fd in forecast_dates:
            positive_records.append({cell_col: cid, date_col: fd})

    if not positive_records:
        return label_df

    positive_df = pd.DataFrame(positive_records).drop_duplicates()
    positive_df["label"] = 1

    # Merge back, preferring label=1 over label=0
    merged = label_df.drop(columns=["label"]).merge(
        positive_df,
        on=[cell_col, date_col],
        how="left",
    )
    merged["label"] = merged["label"].fillna(0).astype(int)

    return merged


# ---------------------------------------------------------------------------
# Utility: fence the boundary between splits
# ---------------------------------------------------------------------------


def trim_label_boundary(
    label_df: pd.DataFrame,
    split_end_date: str,
    window: int = 7,
    date_col: str = "date",
) -> pd.DataFrame:
    """
    Remove the last ``window - 1`` days from a training split.

    At the train/val boundary, labels for the last ``window - 1`` training
    days depend on events in the validation period.  Including these rows
    would introduce subtle leakage.  Drop them before fitting.

    Parameters
    ----------
    label_df:
        Label DataFrame for the training period.
    split_end_date:
        Inclusive end date of the training split (ISO string).
    window:
        Prediction window (default 7).
    date_col:
        Date column name.

    Returns
    -------
    pd.DataFrame
        Trimmed label DataFrame.
    """
    end = pd.Timestamp(split_end_date)
    cutoff = end - pd.Timedelta(days=window - 1)
    trimmed = label_df[label_df[date_col] <= cutoff].copy()
    dropped = len(label_df) - len(trimmed)
    logger.info(
        "trim_label_boundary: removed %d rows near split boundary (cutoff %s).",
        dropped,
        cutoff.strftime("%Y-%m-%d"),
    )
    return trimmed
