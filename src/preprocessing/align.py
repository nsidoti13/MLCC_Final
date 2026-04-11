"""
src/preprocessing/align.py
===========================
Functions to reproject rasters and tabular data onto the H3 hexagonal grid,
align temporal resolution to a daily timestep, and handle missing values.

All functions operate on in-memory pandas / geopandas objects.  Upstream
callers are responsible for reading raw files from data/raw/ and writing
outputs to data/interim/.

H3 note
-------
We use H3 resolution 6 (≈ 4 km edge length, ≈ 36 km² area).  The H3 index
for each cell is stored as a string in the ``cell_id`` column throughout the
pipeline.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple, Union

import geopandas as gpd
import h3
import numpy as np
import pandas as pd
from pyproj import Transformer
from shapely.geometry import Point, Polygon

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Grid construction
# ---------------------------------------------------------------------------


def build_california_grid(
    bbox: Dict[str, float],
    resolution: int = 6,
    exclude_lc_codes: Optional[List[int]] = None,
    land_cover_gdf: Optional[gpd.GeoDataFrame] = None,
) -> gpd.GeoDataFrame:
    """
    Build the base H3 hex grid covering California.

    Fills the bounding box with H3 cells at the given resolution, then
    optionally removes water / dense-urban cells based on a land-cover layer.

    Parameters
    ----------
    bbox:
        Dict with keys lon_min, lat_min, lon_max, lat_max (WGS-84 degrees).
    resolution:
        H3 resolution (default 6 ≈ 4 km).
    exclude_lc_codes:
        NLCD land-cover codes to drop (e.g. 11=water, 22-24=developed).
    land_cover_gdf:
        GeoDataFrame with a ``lc_code`` column aligned to the grid.
        Required when exclude_lc_codes is non-empty.

    Returns
    -------
    gpd.GeoDataFrame
        Columns: cell_id (str), geometry (Polygon, WGS-84), lat (float), lon (float).
    """
    # Polyfill the bounding box polygon with H3 cells
    bbox_polygon = {
        "type": "Polygon",
        "coordinates": [[
            [bbox["lon_min"], bbox["lat_min"]],
            [bbox["lon_max"], bbox["lat_min"]],
            [bbox["lon_max"], bbox["lat_max"]],
            [bbox["lon_min"], bbox["lat_max"]],
            [bbox["lon_min"], bbox["lat_min"]],
        ]],
    }

    cell_ids: List[str] = list(h3.polyfill_geojson(bbox_polygon, resolution))
    logger.info("Generated %d H3 cells at resolution %d.", len(cell_ids), resolution)

    rows = []
    for cid in cell_ids:
        lat, lon = h3.h3_to_geo(cid)
        boundary = h3.h3_to_geo_boundary(cid, geo_json=True)  # returns (lon, lat) tuples
        polygon = Polygon(boundary)
        rows.append({"cell_id": cid, "lat": lat, "lon": lon, "geometry": polygon})

    gdf = gpd.GeoDataFrame(rows, crs="EPSG:4326")

    # Optionally filter out non-burnable cells
    if exclude_lc_codes and land_cover_gdf is not None:
        before = len(gdf)
        gdf = gdf.merge(
            land_cover_gdf[["cell_id", "lc_code"]],
            on="cell_id",
            how="left",
        )
        gdf = gdf[~gdf["lc_code"].isin(exclude_lc_codes)].drop(columns=["lc_code"])
        logger.info(
            "Excluded %d non-burnable cells; %d burnable cells remain.",
            before - len(gdf),
            len(gdf),
        )

    return gdf.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Weather alignment
# ---------------------------------------------------------------------------


def align_weather_to_grid(
    df: pd.DataFrame,
    resolution: int = 6,
    lat_col: str = "lat",
    lon_col: str = "lon",
    date_col: str = "date",
    value_cols: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Assign H3 cell IDs to point-based or gridded weather observations and
    aggregate to one row per (cell_id, date).

    The function handles two input formats:
    * **Point observations** (station data): each row has a lat/lon coordinate.
    * **Regular grids** (NOAA GRIB interpolated to lat/lon pairs): same structure.

    Aggregation within each H3 cell uses the mean for continuous variables
    (temperature, humidity, wind speed) and sum for precipitation.

    Parameters
    ----------
    df:
        Input DataFrame with at least lat, lon, and date columns.
    resolution:
        H3 resolution to use for cell assignment (default 6).
    lat_col, lon_col:
        Names of the latitude and longitude columns.
    date_col:
        Name of the date column.  Will be coerced to ``datetime64[ns]``.
    value_cols:
        Columns to aggregate.  If None, all numeric columns except lat/lon
        are aggregated.

    Returns
    -------
    pd.DataFrame
        Columns: cell_id, date, <value_cols...>.  One row per (cell_id, date).

    Raises
    ------
    ValueError
        If lat_col or lon_col are missing from ``df``.
    """
    required = {lat_col, lon_col, date_col}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"align_weather_to_grid: missing columns {missing}")

    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])

    # Assign H3 cell ID for each observation
    df["cell_id"] = df.apply(
        lambda row: h3.geo_to_h3(row[lat_col], row[lon_col], resolution),
        axis=1,
    )

    if value_cols is None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        value_cols = [c for c in numeric_cols if c not in (lat_col, lon_col)]

    # Determine aggregation: sum for precip, mean for everything else
    agg_dict: Dict[str, str] = {}
    for col in value_cols:
        if "precip" in col.lower() or "apcp" in col.lower() or "prcp" in col.lower():
            agg_dict[col] = "sum"
        else:
            agg_dict[col] = "mean"

    grouped = (
        df.groupby(["cell_id", date_col])
        .agg(agg_dict)
        .reset_index()
        .rename(columns={date_col: "date"})
    )

    logger.info(
        "align_weather_to_grid: %d source rows → %d (cell_id, date) rows.",
        len(df),
        len(grouped),
    )
    return grouped


# ---------------------------------------------------------------------------
# Static layer alignment
# ---------------------------------------------------------------------------


def align_static_to_grid(
    gdf: gpd.GeoDataFrame,
    resolution: int = 6,
    value_cols: Optional[List[str]] = None,
    agg_funcs: Optional[Dict[str, str]] = None,
) -> pd.DataFrame:
    """
    Spatially join a GeoDataFrame of static raster/vector features to H3 cells.

    Each feature in ``gdf`` is assigned to the H3 cell that contains its
    centroid, then features are aggregated per cell.

    Common usage:
    * LANDFIRE raster pixels (point geometry after ``rasterio.features.shapes``)
    * USGS DEM slope/aspect pixels
    * Road/powerline densities after buffering

    Parameters
    ----------
    gdf:
        Input GeoDataFrame.  Must have a CRS set.  Geometry may be Points,
        Polygons, or Lines; centroids are used for H3 assignment.
    resolution:
        H3 resolution (default 6).
    value_cols:
        Columns to retain after aggregation.  Defaults to all numeric columns.
    agg_funcs:
        Dict mapping column name → aggregation function (e.g. ``{"fuel_model": "mode"}``).
        For categorical columns, ``"mode"`` (most-frequent value) is recommended.
        Defaults to mean for all numeric columns.

    Returns
    -------
    pd.DataFrame
        Columns: cell_id, <value_cols...>.

    Raises
    ------
    ValueError
        If gdf has no CRS set.
    """
    if gdf.crs is None:
        raise ValueError("align_static_to_grid: input GeoDataFrame has no CRS set.")

    # Reproject to WGS-84 for H3 assignment
    if gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs(epsg=4326)

    centroids = gdf.geometry.centroid

    # Assign H3 cell ID
    gdf = gdf.copy()
    gdf["cell_id"] = [
        h3.geo_to_h3(pt.y, pt.x, resolution)
        for pt in centroids
    ]

    if value_cols is None:
        value_cols = gdf.select_dtypes(include=[np.number]).columns.tolist()
        value_cols = [c for c in value_cols if c != "cell_id"]

    if agg_funcs is None:
        agg_funcs = {col: "mean" for col in value_cols}

    def _safe_mode(series: pd.Series):
        """Return the most-frequent value, or NaN if empty."""
        mode_vals = series.mode()
        return mode_vals.iloc[0] if len(mode_vals) > 0 else np.nan

    agg_dispatch: Dict[str, Union[str, callable]] = {}
    for col in value_cols:
        func = agg_funcs.get(col, "mean")
        if func == "mode":
            agg_dispatch[col] = _safe_mode
        else:
            agg_dispatch[col] = func

    result = (
        gdf.groupby("cell_id")[value_cols]
        .agg(agg_dispatch)
        .reset_index()
    )

    logger.info(
        "align_static_to_grid: %d geometries → %d unique cells.",
        len(gdf),
        len(result),
    )
    return result


# ---------------------------------------------------------------------------
# Temporal alignment
# ---------------------------------------------------------------------------


def align_temporal_resolution(
    df: pd.DataFrame,
    target_freq: str = "D",
    date_col: str = "date",
    group_cols: Optional[List[str]] = None,
    value_cols: Optional[List[str]] = None,
    fill_method: str = "forward",
) -> pd.DataFrame:
    """
    Resample a time-series DataFrame to a target frequency (default daily).

    Use cases:
    * Resampling sub-daily NOAA data to daily.
    * Forward-filling gaps in static-layer time series.

    Parameters
    ----------
    df:
        Input DataFrame with a date column.
    target_freq:
        Pandas offset alias (e.g. ``"D"`` for daily, ``"W"`` for weekly).
    date_col:
        Name of the date column.
    group_cols:
        Columns that define each independent time series (e.g. ``["cell_id"]``).
        If None, all data is treated as a single series.
    value_cols:
        Columns to resample.  Defaults to all numeric columns.
    fill_method:
        How to fill gaps after resampling: ``"forward"``, ``"backward"``,
        or ``"interpolate"``.

    Returns
    -------
    pd.DataFrame
        Resampled DataFrame at target frequency.
    """
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])

    if value_cols is None:
        value_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    if group_cols is None:
        resampled = (
            df.set_index(date_col)[value_cols]
            .resample(target_freq)
            .mean()
            .reset_index()
        )
    else:
        parts = []
        for keys, group in df.groupby(group_cols):
            group_indexed = group.set_index(date_col)[value_cols].resample(target_freq).mean()
            if isinstance(keys, tuple):
                for col, val in zip(group_cols, keys):
                    group_indexed[col] = val
            else:
                group_indexed[group_cols[0]] = keys
            parts.append(group_indexed.reset_index())
        resampled = pd.concat(parts, ignore_index=True)

    # Fill gaps
    if group_cols:
        resampled = resampled.sort_values(group_cols + [date_col])
        if fill_method == "forward":
            resampled[value_cols] = resampled.groupby(group_cols)[value_cols].ffill()
        elif fill_method == "backward":
            resampled[value_cols] = resampled.groupby(group_cols)[value_cols].bfill()
        elif fill_method == "interpolate":
            resampled[value_cols] = resampled.groupby(group_cols)[value_cols].transform(
                lambda s: s.interpolate(method="time")
            )
    else:
        resampled = resampled.sort_values(date_col)
        if fill_method == "forward":
            resampled[value_cols] = resampled[value_cols].ffill()
        elif fill_method == "backward":
            resampled[value_cols] = resampled[value_cols].bfill()
        elif fill_method == "interpolate":
            resampled[value_cols] = resampled[value_cols].interpolate(method="time")

    logger.info(
        "align_temporal_resolution: %d rows → %d rows at freq='%s'.",
        len(df),
        len(resampled),
        target_freq,
    )
    return resampled


# ---------------------------------------------------------------------------
# Missing value handling
# ---------------------------------------------------------------------------


def clean_missing(
    df: pd.DataFrame,
    strategy: str = "median",
    max_missing_frac: float = 0.5,
    group_cols: Optional[List[str]] = None,
    flag_missing: bool = True,
) -> pd.DataFrame:
    """
    Handle missing values in a feature DataFrame.

    Strategy options
    ----------------
    ``"median"``      — Fill with column median (default; robust to outliers).
    ``"mean"``        — Fill with column mean.
    ``"zero"``        — Fill with 0 (appropriate for precipitation sums).
    ``"forward"``     — Forward-fill within each group (time-series imputation).
    ``"drop_rows"``   — Drop rows with *any* missing values.
    ``"drop_cols"``   — Drop columns exceeding ``max_missing_frac``.

    When ``flag_missing=True``, a boolean indicator column ``<col>_missing``
    is added for each column that had any missing values.  This allows the
    model to learn from the missingness pattern itself.

    Parameters
    ----------
    df:
        Input DataFrame.
    strategy:
        Imputation strategy (see above).
    max_missing_frac:
        Threshold for ``"drop_cols"`` strategy.  Columns with fraction of
        missing values above this threshold are dropped.
    group_cols:
        For ``"forward"`` strategy, defines the groups (e.g. ``["cell_id"]``).
    flag_missing:
        Whether to add binary indicator columns for missingness.

    Returns
    -------
    pd.DataFrame
        DataFrame with missing values handled.
    """
    df = df.copy()
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    non_numeric = [c for c in df.columns if c not in numeric_cols]

    # Identify columns with missing data
    missing_mask = df[numeric_cols].isnull()
    cols_with_missing = [c for c in numeric_cols if missing_mask[c].any()]

    if not cols_with_missing:
        logger.info("clean_missing: no missing values found.")
        return df

    # Add missingness indicator columns before imputation
    if flag_missing:
        for col in cols_with_missing:
            df[f"{col}_missing"] = missing_mask[col].astype(int)

    if strategy == "drop_cols":
        drop_cols = [
            c for c in numeric_cols
            if df[c].isnull().mean() > max_missing_frac
        ]
        df = df.drop(columns=drop_cols)
        logger.info("clean_missing (drop_cols): dropped %d columns.", len(drop_cols))
        # Re-identify after dropping
        cols_with_missing = [c for c in cols_with_missing if c not in drop_cols]

    if strategy == "drop_rows":
        before = len(df)
        df = df.dropna(subset=numeric_cols)
        logger.info("clean_missing (drop_rows): dropped %d rows.", before - len(df))
        return df

    if strategy == "forward":
        if group_cols is None:
            df[numeric_cols] = df[numeric_cols].ffill().bfill()
        else:
            df = df.sort_values(group_cols)
            df[numeric_cols] = df.groupby(group_cols)[numeric_cols].ffill().bfill().values
    elif strategy == "median":
        for col in cols_with_missing:
            if col in df.columns:
                df[col] = df[col].fillna(df[col].median())
    elif strategy == "mean":
        for col in cols_with_missing:
            if col in df.columns:
                df[col] = df[col].fillna(df[col].mean())
    elif strategy == "zero":
        df[numeric_cols] = df[numeric_cols].fillna(0)
    else:
        raise ValueError(f"clean_missing: unknown strategy '{strategy}'.")

    remaining = df[numeric_cols].isnull().sum().sum()
    logger.info(
        "clean_missing (%s): handled %d columns; %d missing values remain.",
        strategy,
        len(cols_with_missing),
        remaining,
    )
    return df


# ---------------------------------------------------------------------------
# Raster reprojection helper
# ---------------------------------------------------------------------------


def reproject_raster_to_h3(
    raster_path: str,
    resolution: int = 6,
    band: int = 1,
    stat: str = "mean",
) -> pd.DataFrame:
    """
    Aggregate a GeoTIFF raster onto an H3 grid by sampling pixel values.

    This function reads a raster file, iterates over pixels with valid data,
    assigns each pixel to its H3 cell, and aggregates using the given
    statistic.

    Requires ``rasterio`` (optional dependency — install separately if needed).

    Parameters
    ----------
    raster_path:
        Path to the input GeoTIFF (or any GDAL-readable raster).
    resolution:
        H3 resolution (default 6).
    band:
        Raster band index (1-based, default 1).
    stat:
        Aggregation statistic: ``"mean"``, ``"median"``, ``"max"``, ``"min"``,
        ``"sum"``, or ``"mode"``.

    Returns
    -------
    pd.DataFrame
        Columns: cell_id, value.  One row per H3 cell that has data.
    """
    try:
        import rasterio
        from rasterio.warp import calculate_default_transform, reproject, Resampling
    except ImportError as exc:
        raise ImportError(
            "rasterio is required for reproject_raster_to_h3. "
            "Install it with: pip install rasterio"
        ) from exc

    records: Dict[str, List[float]] = {}

    with rasterio.open(raster_path) as src:
        # Reproject to WGS-84 if necessary
        if src.crs.to_epsg() != 4326:
            transform, width, height = calculate_default_transform(
                src.crs, "EPSG:4326", src.width, src.height, *src.bounds
            )
            data = np.zeros((height, width), dtype=src.dtypes[band - 1])
            reproject(
                source=rasterio.band(src, band),
                destination=data,
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=transform,
                dst_crs="EPSG:4326",
                resampling=Resampling.bilinear,
            )
            nodata = src.nodata
            bounds_transform = transform
        else:
            data = src.read(band)
            nodata = src.nodata
            bounds_transform = src.transform

        rows_idx, cols_idx = np.where(
            (data != nodata) & np.isfinite(data) if nodata is not None else np.isfinite(data)
        )
        for r, c in zip(rows_idx, cols_idx):
            lon, lat = bounds_transform * (c + 0.5, r + 0.5)
            cell = h3.geo_to_h3(lat, lon, resolution)
            records.setdefault(cell, []).append(float(data[r, c]))

    agg_map = {
        "mean": np.mean,
        "median": np.median,
        "max": np.max,
        "min": np.min,
        "sum": np.sum,
        "mode": lambda x: float(pd.Series(x).mode().iloc[0]),
    }
    agg_func = agg_map.get(stat, np.mean)

    result_rows = [
        {"cell_id": cid, "value": agg_func(vals)}
        for cid, vals in records.items()
    ]

    logger.info(
        "reproject_raster_to_h3: aggregated %s → %d cells.",
        raster_path,
        len(result_rows),
    )
    return pd.DataFrame(result_rows)
