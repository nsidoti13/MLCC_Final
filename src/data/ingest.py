"""
src/data/ingest.py
==================
Ingestion functions for locally-available datasets.

These functions load raw CSV files, clean and parse them, and return
DataFrames in a normalised format ready for the preprocessing stage.

Datasets handled
----------------
- CAPDSI   : California Palmer Drought Severity Index (monthly statewide)
- CAWeather: NOAA GHCN-D weather station observations (daily, 88 CA stations)
- MODIS    : Active fire detections (see src/labeling/labels.py for label use)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# CAPDSI — California Palmer Drought Severity Index
# ---------------------------------------------------------------------------


def load_capdsi(
    csv_path: str | Path,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> pd.DataFrame:
    """
    Load the California Palmer Drought Severity Index (PDSI) monthly CSV.

    **Source**: NOAA National Centers for Environmental Information (NCEI)
    California statewide PDSI, monthly averages.

    The PDSI ranges roughly from -10 (extreme drought) to +10 (extreme wet).
    Values below -2 indicate moderate-to-severe drought conditions that
    strongly correlate with elevated fire risk.

    Raw file format
    ---------------
    The file has a two-row header; actual data begins on row 2::

        #  California Palmer Drought Severity Index (PDSI)
        Date,Value
        201501,-4.84
        ...

    Parameters
    ----------
    csv_path:
        Path to ``CAPDSI.csv``.
    start_date:
        Optional ISO date string to filter the start (e.g. ``"2015-01-01"``).
    end_date:
        Optional ISO date string to filter the end (e.g. ``"2023-12-31"``).

    Returns
    -------
    pd.DataFrame
        Columns:
        - ``date``  : first day of each month (datetime64)
        - ``pdsi``  : Palmer Drought Severity Index (float)
        Sorted by date ascending.
    """
    df = pd.read_csv(csv_path, skiprows=1, names=["date", "pdsi"])

    # Drop any residual header rows that crept in
    df = df[df["date"].astype(str).str.match(r"^\d{6}$")].copy()
    df["pdsi"] = pd.to_numeric(df["pdsi"], errors="coerce")
    df = df.dropna(subset=["pdsi"])

    # Parse YYYYMM → first day of month
    df["date"] = pd.to_datetime(df["date"].astype(str), format="%Y%m")

    if start_date:
        df = df[df["date"] >= pd.Timestamp(start_date)]
    if end_date:
        df = df[df["date"] <= pd.Timestamp(end_date)]

    df = df.sort_values("date").reset_index(drop=True)

    logger.info(
        "CAPDSI: loaded %d monthly records (%s → %s). "
        "PDSI range: [%.2f, %.2f].",
        len(df),
        df["date"].min().strftime("%Y-%m"),
        df["date"].max().strftime("%Y-%m"),
        df["pdsi"].min(),
        df["pdsi"].max(),
    )
    return df


def expand_capdsi_to_daily(pdsi_df: pd.DataFrame) -> pd.DataFrame:
    """
    Forward-fill the monthly PDSI to a daily time series.

    Each day inherits the PDSI value of the month it falls in.  This makes
    the drought index joinable with the daily (cell_id, date) feature table.

    Parameters
    ----------
    pdsi_df:
        Output of ``load_capdsi``.

    Returns
    -------
    pd.DataFrame
        Columns: ``date`` (daily), ``pdsi``.
    """
    if pdsi_df.empty:
        return pdsi_df

    start = pdsi_df["date"].min()
    end = pdsi_df["date"].max() + pd.offsets.MonthEnd(1)
    daily_dates = pd.date_range(start=start, end=end, freq="D")

    daily = pd.DataFrame({"date": daily_dates})
    # Merge on month start, then forward-fill within each month
    daily["month_start"] = daily["date"].values.astype("datetime64[M]")
    pdsi_df = pdsi_df.copy()
    pdsi_df["month_start"] = pdsi_df["date"].values.astype("datetime64[M]")
    daily = daily.merge(pdsi_df[["month_start", "pdsi"]], on="month_start", how="left")
    daily = daily.drop(columns=["month_start"]).sort_values("date")
    daily["pdsi"] = daily["pdsi"].ffill()

    logger.info("CAPDSI expanded to %d daily rows.", len(daily))
    return daily.reset_index(drop=True)


# ---------------------------------------------------------------------------
# CAWeather — NOAA GHCN-D weather station observations
# ---------------------------------------------------------------------------


def load_caweather(
    csv_path: str | Path,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    min_stations_per_cell: int = 1,
) -> pd.DataFrame:
    """
    Load NOAA GHCN-D daily weather station observations for California.

    **Source**: NOAA Global Historical Climatology Network – Daily (GHCN-D)
    https://www.ncei.noaa.gov/products/land-based-station/global-historical-climatology-network-daily

    Variables
    ---------
    AWND  — Average daily wind speed (m/s)
    PRCP  — Precipitation (mm)
    TMAX  — Maximum temperature (°C)
    TMIN  — Minimum temperature (°C)

    Parameters
    ----------
    csv_path:
        Path to ``CAWeather.csv``.
    start_date:
        Optional ISO start date filter.
    end_date:
        Optional ISO end date filter.
    min_stations_per_cell:
        Not used at load time; passed through for documentation purposes.
        Spatial aggregation to H3 cells is handled by
        ``interpolate_weather_to_grid``.

    Returns
    -------
    pd.DataFrame
        Columns: ``station``, ``name``, ``latitude``, ``longitude``,
        ``elevation``, ``date``, ``awnd``, ``prcp``, ``tmax``, ``tmin``.
        Missing sensor readings are left as NaN for downstream imputation.
    """
    df = pd.read_csv(csv_path, parse_dates=["DATE"])
    df.columns = [c.lower() for c in df.columns]
    df = df.rename(columns={"date": "date"})

    if start_date:
        df = df[df["date"] >= pd.Timestamp(start_date)]
    if end_date:
        df = df[df["date"] <= pd.Timestamp(end_date)]

    # Coerce numeric columns (some may arrive as strings with missing markers)
    for col in ["awnd", "prcp", "tmax", "tmin"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.sort_values(["station", "date"]).reset_index(drop=True)

    logger.info(
        "CAWeather: loaded %d station-day records | %d stations | %s → %s.",
        len(df),
        df["station"].nunique(),
        df["date"].min().strftime("%Y-%m-%d"),
        df["date"].max().strftime("%Y-%m-%d"),
    )
    return df


def interpolate_weather_to_grid(
    weather_df: pd.DataFrame,
    grid_df: pd.DataFrame,
    h3_resolution: int = 6,
    method: str = "nearest",
) -> pd.DataFrame:
    """
    Spatially assign weather station observations to H3 grid cells.

    For each (cell_id, date), the value from the nearest weather station
    with a valid reading is used.  This is an inverse-distance-weighted (IDW)
    approximation when ``method="idw"``, or a simple nearest-station lookup
    when ``method="nearest"``.

    Parameters
    ----------
    weather_df:
        Output of ``load_caweather``.
    grid_df:
        GeoDataFrame with ``cell_id`` and ``geometry`` columns (H3 cells).
    h3_resolution:
        H3 resolution — used to compute cell centroids for distance calc.
    method:
        ``"nearest"`` (default) or ``"idw"`` (inverse distance weighting).

    Returns
    -------
    pd.DataFrame
        Columns: ``cell_id``, ``date``, ``awnd``, ``prcp``, ``tmax``, ``tmin``.
        One row per (cell_id, date).
    """
    try:
        import h3
        from scipy.spatial import cKDTree
    except ImportError as exc:
        raise ImportError("scipy is required: pip install scipy") from exc

    # --- Build cell centroid lookup -----------------------------------------
    cell_ids = grid_df["cell_id"].unique()
    cell_centroids = np.array(
        [h3.h3_to_geo(cid) for cid in cell_ids]
    )  # shape (N, 2): (lat, lon)
    cell_lat = cell_centroids[:, 0]
    cell_lon = cell_centroids[:, 1]

    # Convert to radians for haversine-approximate KD-tree
    cell_coords_rad = np.deg2rad(np.column_stack([cell_lat, cell_lon]))

    # --- Build station KD-tree -----------------------------------------------
    stations = (
        weather_df[["station", "latitude", "longitude"]]
        .drop_duplicates("station")
        .reset_index(drop=True)
    )
    station_coords_rad = np.deg2rad(
        stations[["latitude", "longitude"]].values
    )
    tree = cKDTree(station_coords_rad)

    weather_vars = ["awnd", "prcp", "tmax", "tmin"]

    if method == "nearest":
        # Find the nearest station index for each cell
        _, nearest_idx = tree.query(cell_coords_rad, k=1)
        cell_to_station = pd.DataFrame({
            "cell_id": cell_ids,
            "station": stations.iloc[nearest_idx]["station"].values,
        })

        # Merge cell→station mapping with weather observations
        result = cell_to_station.merge(
            weather_df[["station", "date"] + weather_vars],
            on="station",
            how="left",
        ).drop(columns=["station"])

    elif method == "idw":
        k = min(5, len(stations))
        distances, indices = tree.query(cell_coords_rad, k=k)

        records = []
        dates = weather_df["date"].unique()

        # Pre-group weather by station for faster lookup
        station_daily = weather_df.set_index(["station", "date"])

        for cid, dists, idxs in zip(cell_ids, distances, indices):
            nearby_stations = stations.iloc[idxs]["station"].values
            # Avoid division by zero for exact matches
            weights = 1.0 / np.maximum(dists, 1e-10)
            weights /= weights.sum()

            for d in dates:
                row: dict = {"cell_id": cid, "date": d}
                for var in weather_vars:
                    vals, ws = [], []
                    for st, w in zip(nearby_stations, weights):
                        try:
                            v = station_daily.loc[(st, d), var]
                            if pd.notna(v):
                                vals.append(v)
                                ws.append(w)
                        except KeyError:
                            continue
                    row[var] = float(np.average(vals, weights=ws)) if vals else np.nan
                records.append(row)

        result = pd.DataFrame(records)

    else:
        raise ValueError(f"Unknown interpolation method: {method!r}. Use 'nearest' or 'idw'.")

    result = result.sort_values(["cell_id", "date"]).reset_index(drop=True)
    logger.info(
        "interpolate_weather_to_grid: %d cell-day rows (method=%s).",
        len(result), method,
    )
    return result
