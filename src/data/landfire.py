"""
src/data/landfire.py
====================
Query LANDFIRE vegetation/fuel layer values at H3 cell centroids via the
ArcGIS ImageServer getSamples REST API — no download or rasterio required.

Layers queried (LF 2022)
------------------------
LF2022_EVT_CONUS    Existing Vegetation Type   (categorical int)
LF2022_FBFM40_CONUS Fire Behavior Fuel Model 40 (categorical int)
LF2022_CC_CONUS     Canopy Cover               (0–100 %)

Output
------
DataFrame: cell_id, evt, fbfm40, canopy_cover
Cached to data/raw/landfire/landfire_features.parquet after first run.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

log = logging.getLogger(__name__)

BASE_URL = "https://lfps.usgs.gov/arcgis/rest/services/Landfire_LF2022/{layer}/ImageServer/getSamples"

LAYERS = {
    "evt":          "LF2022_EVT_CONUS",
    "fbfm40":       "LF2022_FBFM40_CONUS",
    "canopy_cover": "LF2022_CC_CONUS",
}

OUT_DIR = Path(__file__).resolve().parents[2] / "data" / "raw" / "landfire"
BATCH_SIZE = 500   # points per API request (max ~1000)


def _query_layer(lats: np.ndarray, lons: np.ndarray, layer_name: str) -> np.ndarray:
    """
    Query a LANDFIRE ImageServer layer at all (lat, lon) points.
    Batches requests to stay within API limits.

    Returns np.ndarray of sampled values (float32), same length as lats.
    """
    import json as _json

    url = BASE_URL.format(layer=layer_name)
    n = len(lats)
    values = np.full(n, np.nan, dtype=np.float32)

    for start in range(0, n, BATCH_SIZE):
        end = min(start + BATCH_SIZE, n)
        points = [[float(lons[i]), float(lats[i])] for i in range(start, end)]
        geo = {"points": points, "spatialReference": {"wkid": 4326}}
        params = {
            "geometry":     _json.dumps(geo),
            "geometryType": "esriGeometryMultipoint",
            "sampleCount":  len(points),
            "f":            "json",
        }

        for attempt in range(3):
            try:
                resp = requests.post(url, data=params, timeout=60)
                resp.raise_for_status()
                data = resp.json()
                break
            except Exception as exc:
                if attempt == 2:
                    raise
                log.warning("Batch %d-%d attempt %d failed: %s — retrying…",
                            start, end, attempt + 1, exc)
                time.sleep(2 ** attempt)

        for sample in data.get("samples", []):
            idx = start + sample["locationId"]
            try:
                values[idx] = float(sample["value"])
            except (ValueError, TypeError):
                values[idx] = 0.0

        if (start // BATCH_SIZE) % 10 == 0:
            log.info("  %s: %d/%d points sampled", layer_name, end, n)

    return values


def build_vegetation_features(grid_df: pd.DataFrame,
                               tif_dir=None) -> pd.DataFrame:
    """
    Build per-cell static vegetation/fuel features from LANDFIRE ImageServer.

    Parameters
    ----------
    grid_df  : H3 grid DataFrame with columns cell_id, lat, lon
    tif_dir  : ignored (kept for API compatibility)

    Returns
    -------
    DataFrame with columns: cell_id, evt, fbfm40, canopy_cover
    """
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    parquet_path = OUT_DIR / "landfire_features.parquet"

    if parquet_path.exists():
        log.info("Loading cached LANDFIRE features from %s", parquet_path)
        return pd.read_parquet(parquet_path)

    lats = grid_df["lat"].values
    lons = grid_df["lon"].values

    result = pd.DataFrame({"cell_id": grid_df["cell_id"].values})

    for col, layer_name in LAYERS.items():
        log.info("Querying LANDFIRE layer %s (%d cells)…", layer_name, len(lats))
        vals = _query_layer(lats, lons, layer_name)
        result[col] = vals

    result["evt"]          = result["evt"].fillna(0).astype(np.int32)
    result["fbfm40"]       = result["fbfm40"].fillna(0).astype(np.int32)
    result["canopy_cover"] = result["canopy_cover"].fillna(0.0).astype(np.float32)

    result.to_parquet(parquet_path, index=False)
    log.info(
        "LANDFIRE features built and cached → %s  "
        "(evt unique=%d, fbfm40 unique=%d, canopy_cover mean=%.1f%%)",
        parquet_path,
        result["evt"].nunique(),
        result["fbfm40"].nunique(),
        result["canopy_cover"].mean(),
    )
    return result
