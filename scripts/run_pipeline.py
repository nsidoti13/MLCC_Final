#!/usr/bin/env python3
"""
scripts/run_pipeline.py
=======================
End-to-end wildfire ignition prediction pipeline runner.

Each stage can be skipped independently with --skip-* flags, which is useful
when re-running after a failure or when iterating on a specific stage.

Usage
-----
# Full run (all stages)
python scripts/run_pipeline.py

# Skip data download and preprocessing (use cached interim data)
python scripts/run_pipeline.py --skip-download --skip-preprocess

# Evaluate only
python scripts/run_pipeline.py --skip-download --skip-preprocess \
    --skip-labels --skip-features --skip-train

# Use XGBoost instead of LightGBM
python scripts/run_pipeline.py --model-type xgboost

# Custom config file
python scripts/run_pipeline.py --config configs/config_dev.yaml
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import pandas as pd
import yaml

# Make project root importable regardless of where the script is called from
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("run_pipeline")


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Wildfire Ignition Prediction — End-to-End Pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Stage skip flags
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip raw data download stage (all sources).",
    )
    # Granular download source skips
    parser.add_argument("--skip-gridmet",  action="store_true", help="Skip gridMET download.")
    parser.add_argument("--skip-ndfd",     action="store_true", help="Skip NDFD download.")
    parser.add_argument("--skip-era5",     action="store_true", help="Skip ERA5 download.")
    parser.add_argument("--skip-noaa",     action="store_true", help="Skip NOAA CFSv2 download.")
    parser.add_argument("--skip-modis",    action="store_true", help="Skip MODIS download.")
    parser.add_argument("--skip-landfire", action="store_true", help="Skip LANDFIRE download.")
    parser.add_argument("--skip-terrain",  action="store_true", help="Skip terrain DEM download.")
    parser.add_argument("--skip-human",    action="store_true", help="Skip human features download.")
    parser.add_argument(
        "--skip-preprocess",
        action="store_true",
        help="Skip preprocessing / alignment stage.",
    )
    parser.add_argument(
        "--skip-labels",
        action="store_true",
        help="Skip label construction stage.",
    )
    parser.add_argument(
        "--skip-features",
        action="store_true",
        help="Skip feature engineering stage.",
    )
    parser.add_argument(
        "--skip-train",
        action="store_true",
        help="Skip model training stage.",
    )
    parser.add_argument(
        "--skip-evaluate",
        action="store_true",
        help="Skip evaluation stage.",
    )
    parser.add_argument(
        "--skip-inference",
        action="store_true",
        help="Skip inference / map generation stage.",
    )

    # Configuration
    parser.add_argument(
        "--config",
        type=str,
        default=str(PROJECT_ROOT / "configs" / "config.yaml"),
        help="Path to YAML config file.",
    )
    parser.add_argument(
        "--model-type",
        type=str,
        default="lgbm",
        choices=["lgbm", "xgboost", "logistic"],
        help="Model type to train.",
    )
    parser.add_argument(
        "--start-year",
        type=int,
        default=2015,
        help="Start year for data download.",
    )
    parser.add_argument(
        "--end-year",
        type=int,
        default=2023,
        help="End year for data download.",
    )
    parser.add_argument(
        "--inference-date",
        type=str,
        default=None,
        help="Date for inference (ISO format).  If None, use test split.",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity.",
    )

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Pipeline stages
# ---------------------------------------------------------------------------


def stage_download(config: dict, args: argparse.Namespace) -> None:
    """Stage 1: Download all raw data sources."""
    from src.data.download import download_all

    logger.info("=== STAGE 1: Data Download ===")
    t0 = time.time()

    results = download_all(
        start_year=args.start_year,
        end_year=args.end_year,
        skip_gridmet=getattr(args, "skip_gridmet", False),
        skip_ndfd=getattr(args, "skip_ndfd", False),
        skip_era5=getattr(args, "skip_era5", False),
        skip_noaa=getattr(args, "skip_noaa", False),
        skip_modis=getattr(args, "skip_modis", False),
        skip_landfire=getattr(args, "skip_landfire", False),
        skip_terrain=getattr(args, "skip_terrain", False),
        skip_human=getattr(args, "skip_human", False),
    )

    total = sum(len(v) for v in results.values())
    logger.info("Download complete: %d files in %.1fs.", total, time.time() - t0)


def stage_preprocess(config: dict) -> None:
    """Stage 2: Align raw data to the H3 grid and save interim files."""
    from src.preprocessing.align import (
        align_weather_to_grid,
        align_static_to_grid,
        clean_missing,
        build_california_grid,
    )

    logger.info("=== STAGE 2: Preprocessing ===")
    t0 = time.time()

    raw_dir = Path(config["data"]["raw_dir"])
    interim_dir = Path(config["data"]["interim_dir"])
    interim_dir.mkdir(parents=True, exist_ok=True)

    bbox = config["grid"]["bbox"]
    resolution = config["grid"]["resolution"]

    # Build the base H3 grid
    logger.info("Building H3 grid (resolution=%d)...", resolution)
    grid_gdf = build_california_grid(bbox, resolution)
    grid_path = interim_dir / "ca_grid.parquet"
    grid_gdf.to_parquet(grid_path, index=False)
    logger.info("Grid saved: %d cells → %s", len(grid_gdf), grid_path)

    # Align weather data — prefer gridMET, fall back to ERA5, then NOAA CFSv2
    weather_aligned = None
    for source_name, source_dir in [
        ("gridMET", raw_dir / "gridmet"),
        ("ERA5",    raw_dir / "era5"),
        ("NOAA",    raw_dir / "noaa"),
    ]:
        if source_dir.exists() and any(source_dir.rglob("*.parquet")):
            logger.info("Aligning %s weather data…", source_name)
            weather_raw = pd.read_parquet(next(source_dir.rglob("*.parquet")))
            weather_aligned = align_weather_to_grid(weather_raw, resolution)
            weather_aligned = clean_missing(weather_aligned, strategy="forward", group_cols=["cell_id"])
            weather_aligned.to_parquet(interim_dir / "weather_aligned.parquet", index=False)
            logger.info("%s weather aligned: %d rows.", source_name, len(weather_aligned))
            break

    if weather_aligned is None:
        logger.warning(
            "No weather files found (gridMET/ERA5/NOAA).  "
            "Run download stage first or use --skip-preprocess."
        )

    logger.info("Preprocessing complete in %.1fs.", time.time() - t0)


def stage_labels(config: dict) -> pd.DataFrame:
    """Stage 3: Build (cell_id, date, label) target DataFrame."""
    from src.labeling.labels import build_label_dataframe, trim_label_boundary

    logger.info("=== STAGE 3: Label Construction ===")
    t0 = time.time()

    interim_dir = Path(config["data"]["interim_dir"])
    processed_dir = Path(config["data"]["processed_dir"])
    processed_dir.mkdir(parents=True, exist_ok=True)

    grid_path = interim_dir / "ca_grid.parquet"
    if not grid_path.exists():
        raise FileNotFoundError(
            f"Grid file not found: {grid_path}. Run preprocessing first."
        )
    grid_df = pd.read_parquet(grid_path)

    fire_path = interim_dir / "fire_ignitions.parquet"
    if not fire_path.exists():
        raise FileNotFoundError(
            f"Fire ignitions file not found: {fire_path}. Run preprocessing first."
        )
    fire_df = pd.read_parquet(fire_path)

    window = config["temporal"]["prediction_window_days"]
    train_end = config["splits"]["train"]["end"]

    label_df = build_label_dataframe(
        fire_df=fire_df,
        grid_df=grid_df,
        window=window,
        start_date=config["splits"]["train"]["start"],
        end_date=config["splits"]["test"]["end"],
    )

    # Trim boundary rows from training split to prevent leakage
    train_labels = label_df[label_df["date"] <= pd.Timestamp(train_end)]
    train_labels = trim_label_boundary(train_labels, train_end, window=window)
    train_labels.to_parquet(processed_dir / "labels_train.parquet", index=False)

    val_end = config["splits"]["val"]["end"]
    val_labels = label_df[
        (label_df["date"] >= pd.Timestamp(config["splits"]["val"]["start"]))
        & (label_df["date"] <= pd.Timestamp(val_end))
    ]
    val_labels.to_parquet(processed_dir / "labels_val.parquet", index=False)

    test_labels = label_df[label_df["date"] >= pd.Timestamp(config["splits"]["test"]["start"])]
    test_labels.to_parquet(processed_dir / "labels_test.parquet", index=False)

    logger.info(
        "Labels complete: train=%d, val=%d, test=%d rows in %.1fs.",
        len(train_labels), len(val_labels), len(test_labels), time.time() - t0,
    )
    return label_df


def stage_features(config: dict) -> None:
    """Stage 4: Engineer full feature matrix."""
    from src.features.engineer import build_features

    logger.info("=== STAGE 4: Feature Engineering ===")
    t0 = time.time()

    interim_dir = Path(config["data"]["interim_dir"])
    processed_dir = Path(config["data"]["processed_dir"])

    required = {
        "weather_aligned.parquet": "weather",
        "static_features.parquet": "static",
        "fire_ignitions.parquet": "fire_history",
        "ca_grid.parquet": "grid",
    }

    data = {}
    for fname, key in required.items():
        fpath = interim_dir / fname
        if not fpath.exists():
            raise FileNotFoundError(
                f"Required interim file missing: {fpath}. "
                "Run preprocessing first."
            )
        data[key] = pd.read_parquet(fpath)

    human_path = interim_dir / "human_features.parquet"
    human_df = pd.read_parquet(human_path) if human_path.exists() else None

    for split in ["train", "val", "test"]:
        label_path = processed_dir / f"labels_{split}.parquet"
        if not label_path.exists():
            logger.warning("Labels file not found for split '%s'; skipping.", split)
            continue

        labels = pd.read_parquet(label_path)
        start = labels["date"].min().strftime("%Y-%m-%d")
        end = labels["date"].max().strftime("%Y-%m-%d")

        features = build_features(
            weather_df=data["weather"],
            static_df=data["static"],
            fire_history_df=data["fire_history"],
            grid_df=data["grid"],
            human_df=human_df,
            start_date=start,
            end_date=end,
        )

        # Merge labels onto features
        features_labeled = features.merge(
            labels[["cell_id", "date", "label"]],
            on=["cell_id", "date"],
            how="inner",
        )

        out_path = processed_dir / f"features_{split}.parquet"
        features_labeled.to_parquet(out_path, index=False)
        logger.info(
            "Features saved for %s: %d rows, %d cols → %s",
            split, len(features_labeled), features_labeled.shape[1], out_path,
        )

    logger.info("Feature engineering complete in %.1fs.", time.time() - t0)


def stage_train(config: dict, model_type: str) -> None:
    """Stage 5: Train and save the model."""
    from src.features.engineer import get_feature_columns
    from src.modeling.trainer import train_model

    logger.info("=== STAGE 5: Model Training (%s) ===", model_type)
    t0 = time.time()

    processed_dir = Path(config["data"]["processed_dir"])
    models_dir = PROJECT_ROOT / "outputs" / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    train_path = processed_dir / "features_train.parquet"
    val_path = processed_dir / "features_val.parquet"

    if not train_path.exists():
        raise FileNotFoundError(f"Training features not found: {train_path}")

    train_df = pd.read_parquet(train_path)
    val_df = pd.read_parquet(val_path) if val_path.exists() else None

    feature_cols = get_feature_columns(train_df)
    X_train = train_df[feature_cols]
    y_train = train_df["label"]

    eval_set = None
    if val_df is not None:
        eval_set = (val_df[feature_cols], val_df["label"])

    model_params = config.get("models", {}).get(model_type, None)
    model = train_model(
        X_train=X_train,
        y_train=y_train,
        model_type=model_type,
        params=model_params,
        eval_set=eval_set,
    )

    model_path = models_dir / f"{model_type}_model.pkl"
    import pickle
    with open(model_path, "wb") as fh:
        pickle.dump(model, fh)
    logger.info("Model saved to %s in %.1fs.", model_path, time.time() - t0)


def stage_evaluate(config: dict, model_type: str) -> None:
    """Stage 6: Evaluate on validation and test sets."""
    import pickle
    from src.evaluation.metrics import evaluate
    from src.features.engineer import get_feature_columns

    logger.info("=== STAGE 6: Evaluation ===")
    t0 = time.time()

    processed_dir = Path(config["data"]["processed_dir"])
    models_dir = PROJECT_ROOT / "outputs" / "models"
    reports_dir = PROJECT_ROOT / "outputs" / "reports"

    model_path = models_dir / f"{model_type}_model.pkl"
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}. Run training first.")

    with open(model_path, "rb") as fh:
        model = pickle.load(fh)

    for split in ["val", "test"]:
        feat_path = processed_dir / f"features_{split}.parquet"
        if not feat_path.exists():
            logger.warning("Features not found for split '%s'; skipping.", split)
            continue

        feat_df = pd.read_parquet(feat_path)
        feature_cols = get_feature_columns(feat_df)
        X = feat_df[feature_cols]
        y_true = feat_df["label"].values

        y_score = model.predict_proba(X)

        metrics = evaluate(
            y_true=y_true,
            y_score=y_score,
            top_k=config["evaluation"]["top_k"],
            output_dir=str(reports_dir),
            model_name=f"{model_type}_{split}",
        )

        logger.info(
            "[%s/%s] PR-AUC=%.4f | ROC-AUC=%.4f | Recall@500=%.4f",
            model_type, split,
            metrics["pr_auc"],
            metrics["roc_auc"],
            metrics["recall_at_k"],
        )

    logger.info("Evaluation complete in %.1fs.", time.time() - t0)


def stage_inference(config: dict, model_type: str, inference_date: str) -> None:
    """Stage 7: Run inference and save probability maps."""
    import pickle
    from src.inference.predict import run_inference
    from src.features.engineer import get_feature_columns, build_features

    logger.info("=== STAGE 7: Inference ===")
    t0 = time.time()

    processed_dir = Path(config["data"]["processed_dir"])
    models_dir = PROJECT_ROOT / "outputs" / "models"
    predictions_dir = PROJECT_ROOT / "outputs" / "predictions"

    model_path = models_dir / f"{model_type}_model.pkl"
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    with open(model_path, "rb") as fh:
        model = pickle.load(fh)

    # Use test set features for inference, or a specific date if provided
    feat_path = processed_dir / "features_test.parquet"
    if not feat_path.exists():
        raise FileNotFoundError(f"Test features not found: {feat_path}")

    feat_df = pd.read_parquet(feat_path)

    if inference_date:
        feat_df = feat_df[feat_df["date"] == pd.Timestamp(inference_date)]
        if feat_df.empty:
            raise ValueError(f"No features found for inference_date={inference_date}")

    meta_cols = {"cell_id", "date", "label"}
    feature_cols = [c for c in feat_df.columns if c not in meta_cols]

    predictions = run_inference(
        model=model,
        feature_df=feat_df,
        output_path=predictions_dir,
    )

    logger.info(
        "Inference complete: %d predictions saved in %.1fs.",
        len(predictions),
        time.time() - t0,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    args = parse_args()

    # Set log level
    logging.getLogger().setLevel(getattr(logging, args.log_level))

    # Load config
    config_path = Path(args.config)
    if not config_path.exists():
        logger.error("Config file not found: %s", config_path)
        sys.exit(1)
    with open(config_path, "r") as fh:
        config = yaml.safe_load(fh)

    logger.info("Wildfire Ignition Prediction Pipeline")
    logger.info("Config: %s", config_path)
    logger.info("Model type: %s", args.model_type)

    pipeline_start = time.time()

    try:
        if not args.skip_download:
            stage_download(config, args)

        if not args.skip_preprocess:
            stage_preprocess(config)

        if not args.skip_labels:
            stage_labels(config)

        if not args.skip_features:
            stage_features(config)

        if not args.skip_train:
            stage_train(config, args.model_type)

        if not args.skip_evaluate:
            stage_evaluate(config, args.model_type)

        if not args.skip_inference:
            stage_inference(config, args.model_type, args.inference_date)

    except FileNotFoundError as exc:
        logger.error("Missing required file: %s", exc)
        logger.error("Tip: earlier pipeline stages may need to run first.")
        sys.exit(1)
    except Exception as exc:
        logger.exception("Pipeline failed: %s", exc)
        sys.exit(1)

    logger.info(
        "Pipeline complete in %.1f seconds.", time.time() - pipeline_start
    )


if __name__ == "__main__":
    main()
