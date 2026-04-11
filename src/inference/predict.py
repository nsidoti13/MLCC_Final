"""
src/inference/predict.py
========================
Inference functions for running a trained wildfire ignition model on new
feature data and saving probability maps.

The output is a DataFrame with columns:
    cell_id, date, ignition_prob

Saved in two formats:
    1. Parquet — full numeric table for downstream analysis.
    2. GeoJSON — georeferenced probability map for visualisation in GIS tools.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Optional, Union

import geopandas as gpd
import h3
import numpy as np
import pandas as pd
from shapely.geometry import Polygon

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Main inference function
# ---------------------------------------------------------------------------


def run_inference(
    model: Any,
    feature_df: pd.DataFrame,
    output_path: Optional[Union[str, Path]] = None,
    batch_size: int = 100_000,
    threshold: float = 0.5,
) -> pd.DataFrame:
    """
    Run the trained wildfire ignition model on a feature DataFrame and
    save probability maps.

    The function handles arbitrarily large feature DataFrames by processing
    them in batches to avoid memory exhaustion.

    Parameters
    ----------
    model:
        A fitted model with a ``predict_proba(X)`` method that returns a
        1-D array of positive-class probabilities.  Compatible with
        LGBMModel, XGBModel, and scikit-learn estimators.
    feature_df:
        DataFrame produced by ``build_features()``.  Must contain ``cell_id``
        and ``date`` columns in addition to all feature columns.
    output_path:
        Directory where outputs are written.  Defaults to
        ``outputs/predictions/``.  The following files are created:
        - ``predictions_<date_range>.parquet`` — full probability table.
        - ``predictions_<date_range>.geojson`` — georeferenced map.
        - ``top_risk_cells_<date_range>.csv`` — top-500 highest-risk cells.
    batch_size:
        Number of rows to process per batch (default 100k).
    threshold:
        Decision threshold for binary predictions saved alongside probabilities.

    Returns
    -------
    pd.DataFrame
        Columns: cell_id, date, ignition_prob, ignition_pred (0/1).

    Raises
    ------
    ValueError
        If ``feature_df`` is missing required columns.
    RuntimeError
        If the model does not expose a ``predict_proba`` method.
    """
    if not hasattr(model, "predict_proba"):
        raise RuntimeError(
            "run_inference: model must implement predict_proba().  "
            f"Got {type(model).__name__}."
        )

    required_meta = {"cell_id", "date"}
    missing = required_meta - set(feature_df.columns)
    if missing:
        raise ValueError(f"run_inference: feature_df missing columns {missing}")

    if output_path is None:
        output_path = Path("outputs/predictions")
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Identify feature columns (exclude metadata)
    # ------------------------------------------------------------------
    meta_cols = {"cell_id", "date", "label"}
    feature_cols = [c for c in feature_df.columns if c not in meta_cols]

    logger.info(
        "run_inference: %d samples, %d features, batch_size=%d.",
        len(feature_df),
        len(feature_cols),
        batch_size,
    )

    # ------------------------------------------------------------------
    # Batch prediction
    # ------------------------------------------------------------------
    all_probs: list = []
    n_batches = (len(feature_df) + batch_size - 1) // batch_size

    for batch_idx in range(n_batches):
        start = batch_idx * batch_size
        end = min(start + batch_size, len(feature_df))
        batch = feature_df.iloc[start:end]

        X_batch = batch[feature_cols]

        try:
            proba = model.predict_proba(X_batch)
        except Exception as exc:
            logger.error("Prediction failed on batch %d: %s", batch_idx, exc)
            raise

        all_probs.append(proba)
        logger.info(
            "  Batch %d/%d complete (rows %d–%d).",
            batch_idx + 1,
            n_batches,
            start,
            end - 1,
        )

    probs = np.concatenate(all_probs)

    # ------------------------------------------------------------------
    # Build output DataFrame
    # ------------------------------------------------------------------
    predictions = pd.DataFrame(
        {
            "cell_id": feature_df["cell_id"].values,
            "date": feature_df["date"].values,
            "ignition_prob": probs,
            "ignition_pred": (probs >= threshold).astype(int),
        }
    )
    predictions["date"] = pd.to_datetime(predictions["date"])

    # Summary stats
    logger.info(
        "Inference complete: mean_prob=%.5f, pred_positive_rate=%.4f%%",
        probs.mean(),
        predictions["ignition_pred"].mean() * 100,
    )

    # ------------------------------------------------------------------
    # Date range label for filenames
    # ------------------------------------------------------------------
    date_min = predictions["date"].min().strftime("%Y%m%d")
    date_max = predictions["date"].max().strftime("%Y%m%d")
    date_tag = f"{date_min}_{date_max}"

    # ------------------------------------------------------------------
    # Save parquet
    # ------------------------------------------------------------------
    parquet_path = output_path / f"predictions_{date_tag}.parquet"
    predictions.to_parquet(parquet_path, index=False)
    logger.info("Predictions parquet saved to %s", parquet_path)

    # ------------------------------------------------------------------
    # Save GeoJSON probability map
    # ------------------------------------------------------------------
    geojson_path = output_path / f"predictions_{date_tag}.geojson"
    _save_geojson(predictions, geojson_path)

    # ------------------------------------------------------------------
    # Save top-500 risk cells CSV
    # ------------------------------------------------------------------
    top_csv_path = output_path / f"top_risk_cells_{date_tag}.csv"
    _save_top_risk_cells(predictions, top_csv_path, top_n=500)

    return predictions


# ---------------------------------------------------------------------------
# Aggregated risk map (across a date range)
# ---------------------------------------------------------------------------


def aggregate_risk_map(
    predictions: pd.DataFrame,
    agg: str = "max",
    output_path: Optional[Union[str, Path]] = None,
) -> pd.DataFrame:
    """
    Aggregate per-day probabilities to a single spatial risk map.

    Parameters
    ----------
    predictions:
        Output of ``run_inference()`` with cell_id, date, ignition_prob columns.
    agg:
        Aggregation method: ``"max"`` (default), ``"mean"``, ``"sum"``.
        ``"max"`` captures the peak risk over the period.
    output_path:
        If provided, save the aggregated map as a GeoJSON file.

    Returns
    -------
    pd.DataFrame
        Columns: cell_id, ignition_prob_<agg>.
    """
    agg_func_map = {
        "max": "max",
        "mean": "mean",
        "sum": "sum",
    }
    if agg not in agg_func_map:
        raise ValueError(f"aggregate_risk_map: unknown aggregation '{agg}'.")

    col_name = f"ignition_prob_{agg}"
    agg_df = (
        predictions.groupby("cell_id")["ignition_prob"]
        .agg(agg_func_map[agg])
        .reset_index()
        .rename(columns={"ignition_prob": col_name})
    )

    if output_path is not None:
        out_path = Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        # Build a minimal predictions DataFrame for the GeoJSON writer
        _save_geojson(
            agg_df.rename(columns={col_name: "ignition_prob"}),
            out_path,
        )

    return agg_df


# ---------------------------------------------------------------------------
# Save helpers
# ---------------------------------------------------------------------------


def _save_geojson(
    predictions: pd.DataFrame,
    path: Path,
    date_col: str = "date",
) -> None:
    """
    Write predictions as a GeoJSON FeatureCollection.

    Each feature is an H3 hexagon polygon with ``ignition_prob`` (and
    optionally ``date``) as properties.  For multi-date DataFrames, we
    aggregate to the mean probability per cell to keep the file manageable.
    """
    # If multi-date, aggregate to mean per cell for the map
    if date_col in predictions.columns and predictions[date_col].nunique() > 1:
        map_df = (
            predictions.groupby("cell_id")["ignition_prob"]
            .mean()
            .reset_index()
        )
        logger.info("GeoJSON: aggregated %d rows to %d cells (mean prob).", len(predictions), len(map_df))
    else:
        map_df = predictions[["cell_id", "ignition_prob"]].copy()

    features = []
    for _, row in map_df.iterrows():
        cid = row["cell_id"]
        try:
            boundary = h3.h3_to_geo_boundary(cid, geo_json=True)
            geom = {
                "type": "Polygon",
                "coordinates": boundary["coordinates"],
            }
        except Exception:
            continue
        feature = {
            "type": "Feature",
            "geometry": geom,
            "properties": {
                "cell_id": cid,
                "ignition_prob": float(row["ignition_prob"]),
            },
        }
        features.append(feature)

    geojson = {"type": "FeatureCollection", "features": features}
    with open(path, "w") as fh:
        json.dump(geojson, fh)
    logger.info("GeoJSON map saved to %s (%d cells)", path, len(features))


def _save_top_risk_cells(
    predictions: pd.DataFrame,
    path: Path,
    top_n: int = 500,
) -> None:
    """
    Save a CSV of the top-N highest-risk (cell_id, date) pairs.
    """
    top = predictions.nlargest(top_n, "ignition_prob").copy()
    top["lat"] = top["cell_id"].apply(lambda c: h3.h3_to_geo(c)[0])
    top["lon"] = top["cell_id"].apply(lambda c: h3.h3_to_geo(c)[1])
    top.to_csv(path, index=False)
    logger.info("Top-%d risk cells saved to %s", top_n, path)


# ---------------------------------------------------------------------------
# Load model helper
# ---------------------------------------------------------------------------


def load_model(model_path: Union[str, Path]) -> Any:
    """
    Load a serialised model from disk.

    Supports both pickle-serialised LGBMModel / XGBModel wrappers and
    raw LightGBM / XGBoost native formats.

    Parameters
    ----------
    model_path:
        Path to the model file.

    Returns
    -------
    Any
        Loaded model with a ``predict_proba`` method.

    Raises
    ------
    FileNotFoundError
        If model_path does not exist.
    """
    import pickle

    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"load_model: file not found: {path}")

    suffix = path.suffix.lower()

    if suffix == ".pkl":
        with open(path, "rb") as fh:
            model = pickle.load(fh)
        logger.info("Loaded model from pickle: %s", path)
        return model

    if suffix in (".txt", ".bst"):
        # Native LightGBM text format
        try:
            import lightgbm as lgb
            booster = lgb.Booster(model_file=str(path))
            # Wrap booster in a thin callable
            model = _BoosterWrapper(booster, framework="lgbm")
            logger.info("Loaded LightGBM booster from %s", path)
            return model
        except Exception:
            pass

        # Native XGBoost binary format
        try:
            from xgboost import XGBClassifier
            model = XGBClassifier()
            model.load_model(str(path))
            logger.info("Loaded XGBoost model from %s", path)
            return model
        except Exception as exc:
            raise RuntimeError(
                f"load_model: could not load model from {path}: {exc}"
            ) from exc

    raise ValueError(f"load_model: unsupported file format '{suffix}'.")


class _BoosterWrapper:
    """Minimal wrapper that adds predict_proba to a raw booster."""

    def __init__(self, booster: Any, framework: str) -> None:
        self._booster = booster
        self._framework = framework

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if self._framework == "lgbm":
            import lightgbm as lgb
            data = lgb.Dataset(X, free_raw_data=False)
            return self._booster.predict(X.values)
        raise NotImplementedError
