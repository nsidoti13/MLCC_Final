"""
tests/test_features.py
======================
Unit tests for feature engineering in src/features/engineer.py.

Test coverage:
  1. Sin/cos seasonality encoding correctness (values, periodicity, no NaN).
  2. Rolling window features: correct direction (backward only), correct window size.
  3. No NaN in output: build_features produces a clean feature matrix.
  4. Feature column consistency: output contains all expected feature groups.
  5. Static feature merging: static features correctly joined per cell.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.features.engineer import (
    _add_temporal_features,
    _compute_lagged_features,
    _compute_forecast_features,
    _prepare_static_features,
    build_features,
    get_feature_columns,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def make_weather_df(n_cells: int = 2, n_days: int = 60) -> pd.DataFrame:
    """Synthetic weather DataFrame for testing."""
    rng = np.random.default_rng(42)
    cells = [f"cell_{i}" for i in range(n_cells)]
    dates = pd.date_range("2020-01-01", periods=n_days, freq="D")
    rows = []
    for cell in cells:
        for date in dates:
            rows.append(
                {
                    "cell_id": cell,
                    "date": date,
                    "tmp2m": 290 + rng.normal(0, 5),
                    "rh2m": 50 + rng.normal(0, 10),
                    "wnd10m": 3 + rng.exponential(2),
                    "apcp": max(0, rng.normal(1, 2)),
                }
            )
    return pd.DataFrame(rows)


def make_static_df(n_cells: int = 2) -> pd.DataFrame:
    """Synthetic static features."""
    rng = np.random.default_rng(42)
    cells = [f"cell_{i}" for i in range(n_cells)]
    return pd.DataFrame(
        {
            "cell_id": cells,
            "elevation_m": rng.uniform(0, 3000, n_cells),
            "slope_deg": rng.uniform(0, 45, n_cells),
            "aspect_deg": rng.uniform(0, 360, n_cells),
            "fuel_model": rng.choice(["GR1", "SH1", "TU1"], n_cells),
            "canopy_cover_pct": rng.uniform(0, 100, n_cells),
            "canopy_bulk_density": rng.uniform(0, 0.5, n_cells),
            "canopy_base_height": rng.uniform(0, 10, n_cells),
            "vegetation_type": rng.choice(["chaparral", "forest", "grassland"], n_cells),
        }
    )


def make_fire_history_df() -> pd.DataFrame:
    """Synthetic fire history (very sparse)."""
    return pd.DataFrame(
        {
            "cell_id": ["cell_0"],
            "fire_date": ["2020-02-15"],
        }
    )


def make_grid_df(n_cells: int = 2) -> pd.DataFrame:
    """Minimal grid DataFrame."""
    return pd.DataFrame({"cell_id": [f"cell_{i}" for i in range(n_cells)]})


# ---------------------------------------------------------------------------
# Test 1: Sin/cos seasonality encoding
# ---------------------------------------------------------------------------


class TestTemporalFeatures:
    """Verify sin/cos cyclical encoding of time features."""

    def test_doy_sin_cos_range(self):
        """doy_sin and doy_cos must be in [-1, 1]."""
        panel = pd.DataFrame(
            {
                "cell_id": ["cell_0"] * 365,
                "date": pd.date_range("2020-01-01", periods=365, freq="D"),
            }
        )
        result = _add_temporal_features(panel)
        assert result["doy_sin"].between(-1, 1).all(), "doy_sin out of [-1, 1]"
        assert result["doy_cos"].between(-1, 1).all(), "doy_cos out of [-1, 1]"

    def test_month_sin_cos_range(self):
        """month_sin and month_cos must be in [-1, 1]."""
        panel = pd.DataFrame(
            {
                "cell_id": ["cell_0"] * 365,
                "date": pd.date_range("2020-01-01", periods=365, freq="D"),
            }
        )
        result = _add_temporal_features(panel)
        assert result["month_sin"].between(-1, 1).all(), "month_sin out of [-1, 1]"
        assert result["month_cos"].between(-1, 1).all(), "month_cos out of [-1, 1]"

    def test_doy_sin_cos_periodicity(self):
        """
        Day 1 and Day 366 (i.e., start of next year) should produce similar
        (sin, cos) values — the encoding wraps around.
        """
        panel = pd.DataFrame(
            {
                "cell_id": ["cell_0", "cell_0"],
                "date": [pd.Timestamp("2020-01-01"), pd.Timestamp("2021-01-01")],
            }
        )
        result = _add_temporal_features(panel).set_index("date")
        sin_jan1_2020 = result.loc["2020-01-01", "doy_sin"]
        sin_jan1_2021 = result.loc["2021-01-01", "doy_sin"]
        # Both are Jan 1 — sin values should be very close
        assert abs(sin_jan1_2020 - sin_jan1_2021) < 0.05, (
            "Sin encoding should be nearly equal for the same calendar day "
            "across years."
        )

    def test_no_nan_in_temporal_features(self):
        """Temporal features should never produce NaN."""
        panel = pd.DataFrame(
            {
                "cell_id": ["cell_0"] * 100,
                "date": pd.date_range("2020-01-01", periods=100, freq="D"),
            }
        )
        result = _add_temporal_features(panel)
        temporal_cols = ["doy_sin", "doy_cos", "month_sin", "month_cos", "week_of_year"]
        for col in temporal_cols:
            assert result[col].notna().all(), f"NaN found in {col}"

    def test_dec31_and_jan1_are_adjacent_in_sin_space(self):
        """
        Dec 31 (DOY 365) and Jan 1 (DOY 1) should be close in (sin, cos)
        space, confirming the cyclical encoding avoids a discontinuity.
        """
        panel = pd.DataFrame(
            {
                "cell_id": ["cell_0", "cell_0"],
                "date": [pd.Timestamp("2020-12-31"), pd.Timestamp("2021-01-01")],
            }
        )
        result = _add_temporal_features(panel).set_index("date")

        sin_dec31 = result.loc["2020-12-31", "doy_sin"]
        sin_jan1 = result.loc["2021-01-01", "doy_sin"]
        cos_dec31 = result.loc["2020-12-31", "doy_cos"]
        cos_jan1 = result.loc["2021-01-01", "doy_cos"]

        euclidean_dist = np.sqrt((sin_dec31 - sin_jan1)**2 + (cos_dec31 - cos_jan1)**2)
        # Compare with mid-year distance for reference
        mid_year_panel = pd.DataFrame(
            {
                "cell_id": ["cell_0", "cell_0"],
                "date": [pd.Timestamp("2020-06-30"), pd.Timestamp("2020-12-31")],
            }
        )
        mid = _add_temporal_features(mid_year_panel).set_index("date")
        mid_dist = np.sqrt(
            (mid.loc["2020-06-30", "doy_sin"] - mid.loc["2020-12-31", "doy_sin"])**2 +
            (mid.loc["2020-06-30", "doy_cos"] - mid.loc["2020-12-31", "doy_cos"])**2
        )
        assert euclidean_dist < mid_dist, (
            "Dec 31 and Jan 1 should be closer in sin/cos space than "
            "Jun 30 and Dec 31."
        )


# ---------------------------------------------------------------------------
# Test 2: Rolling window features
# ---------------------------------------------------------------------------


class TestRollingFeatures:
    """Verify lagged rolling features use only past data."""

    def test_rolling_uses_past_data_only(self):
        """
        The rolling mean at date t must not include the value at t (shift(1)
        ensures we use strictly past data).
        """
        # Create a step function: tmp2m=0 for first 10 days, then tmp2m=100
        n_days = 20
        dates = pd.date_range("2020-01-01", periods=n_days, freq="D")
        values = [0.0] * 10 + [100.0] * 10
        df = pd.DataFrame(
            {
                "cell_id": ["cell_0"] * n_days,
                "date": dates,
                "tmp2m": values,
            }
        )

        result = _compute_lagged_features(df, lag_short=7, lag_long=7)
        result = result.set_index("date")

        # On 2020-01-11 (first day of 100.0 values), the 7-day lag mean
        # should reflect values from 2020-01-04 to 2020-01-10 — all 0.0.
        jan11 = result.loc["2020-01-11", "tmp2m_roll7_mean"]
        assert jan11 == pytest.approx(0.0, abs=1e-9), (
            f"Rolling mean on 2020-01-11 should be 0.0 (past data only), got {jan11}"
        )

    def test_rolling_window_length_short(self):
        """
        After lag_short days, the rolling mean should use exactly lag_short
        past values.
        """
        lag_short = 7
        n_days = 40
        dates = pd.date_range("2020-01-01", periods=n_days, freq="D")
        # Constant temperature = 300 K
        df = pd.DataFrame(
            {
                "cell_id": ["cell_0"] * n_days,
                "date": dates,
                "tmp2m": [300.0] * n_days,
            }
        )
        result = _compute_lagged_features(df, lag_short=lag_short, lag_long=lag_short)
        result = result.set_index("date")

        # After the first lag_short + 1 days, rolling mean should be 300.0
        stable_date = dates[lag_short + 5]
        mean_val = result.loc[stable_date, "tmp2m_roll7_mean"]
        assert mean_val == pytest.approx(300.0, abs=1e-9), (
            f"Rolling mean of constant series should be 300.0, got {mean_val}"
        )

    def test_rolling_no_nan_after_warmup(self):
        """After the warmup period, no NaN should appear in rolling features."""
        df = make_weather_df(n_cells=1, n_days=60)
        result = _compute_lagged_features(df, lag_short=7, lag_long=30)
        # After 30 warmup days, no NaN should remain
        stable = result[result["date"] >= "2020-02-01"]
        lag_cols = [c for c in result.columns if "roll" in c]
        for col in lag_cols:
            assert stable[col].notna().all(), f"NaN found in {col} after warmup period."

    def test_apcp_rolling_is_sum_not_mean(self):
        """Precipitation rolling should be summed, not averaged."""
        n_days = 40
        dates = pd.date_range("2020-01-01", periods=n_days, freq="D")
        # Constant precipitation of 1.0 mm/day
        df = pd.DataFrame(
            {
                "cell_id": ["cell_0"] * n_days,
                "date": dates,
                "apcp": [1.0] * n_days,
            }
        )
        result = _compute_lagged_features(df, lag_short=7, lag_long=30)
        result = result.set_index("date")

        stable_date = dates[35]
        apcp_val = result.loc[stable_date, "apcp_roll30_sum"]
        # Should be 30 (30 days × 1.0 mm each, shifted so we don't include current)
        assert apcp_val == pytest.approx(30.0, abs=1.0), (
            f"30-day precipitation sum of 1.0/day should be ~30.0, got {apcp_val}"
        )


# ---------------------------------------------------------------------------
# Test 3: No NaN in final feature output
# ---------------------------------------------------------------------------


class TestNoNaNInOutput:
    """Verify that build_features produces a complete feature matrix."""

    def test_no_nan_in_numeric_columns(self):
        """
        The full build_features output should have no NaN in any numeric column.
        """
        n_cells = 2
        n_days = 45  # Enough for 30-day rolling warmup

        weather = make_weather_df(n_cells=n_cells, n_days=n_days)
        static = make_static_df(n_cells=n_cells)
        fire_history = make_fire_history_df()
        grid = make_grid_df(n_cells=n_cells)

        features = build_features(
            weather_df=weather,
            static_df=static,
            fire_history_df=fire_history,
            grid_df=grid,
            human_df=None,
            lag_short=7,
            lag_long=30,
        )

        numeric_cols = features.select_dtypes(include=[np.number]).columns.tolist()
        for col in numeric_cols:
            n_nan = features[col].isna().sum()
            assert n_nan == 0, (
                f"Column '{col}' has {n_nan} NaN values in build_features output."
            )

    def test_output_has_cell_id_and_date(self):
        """Output DataFrame must contain cell_id and date columns."""
        features = build_features(
            weather_df=make_weather_df(n_cells=1, n_days=45),
            static_df=make_static_df(n_cells=1),
            fire_history_df=make_fire_history_df(),
            grid_df=make_grid_df(n_cells=1),
        )
        assert "cell_id" in features.columns, "Missing cell_id column"
        assert "date" in features.columns, "Missing date column"

    def test_get_feature_columns_excludes_metadata(self):
        """get_feature_columns must not return cell_id, date, or label."""
        features = build_features(
            weather_df=make_weather_df(n_cells=1, n_days=45),
            static_df=make_static_df(n_cells=1),
            fire_history_df=make_fire_history_df(),
            grid_df=make_grid_df(n_cells=1),
        )
        # Simulate labeled data
        features["label"] = 0
        feat_cols = get_feature_columns(features)
        assert "cell_id" not in feat_cols, "cell_id must not be in feature columns"
        assert "date" not in feat_cols, "date must not be in feature columns"
        assert "label" not in feat_cols, "label must not be in feature columns"
        assert len(feat_cols) > 0, "No feature columns found"


# ---------------------------------------------------------------------------
# Test 4: Feature column presence
# ---------------------------------------------------------------------------


class TestFeatureColumns:
    """Verify expected feature groups are present in the output."""

    def test_temporal_feature_columns_present(self):
        """All temporal encoding columns must be present."""
        features = build_features(
            weather_df=make_weather_df(n_cells=1, n_days=45),
            static_df=make_static_df(n_cells=1),
            fire_history_df=make_fire_history_df(),
            grid_df=make_grid_df(n_cells=1),
        )
        for col in ["doy_sin", "doy_cos", "month_sin", "month_cos", "week_of_year"]:
            assert col in features.columns, f"Missing temporal feature: {col}"

    def test_rolling_feature_columns_present(self):
        """Lagged rolling features must be present."""
        features = build_features(
            weather_df=make_weather_df(n_cells=1, n_days=45),
            static_df=make_static_df(n_cells=1),
            fire_history_df=make_fire_history_df(),
            grid_df=make_grid_df(n_cells=1),
        )
        expected_rolling = [
            "tmp2m_roll7_mean", "tmp2m_roll30_mean",
            "rh2m_roll7_mean", "rh2m_roll30_mean",
            "apcp_roll30_sum", "wnd10m_roll7_mean",
        ]
        for col in expected_rolling:
            assert col in features.columns, f"Missing rolling feature: {col}"

    def test_spatial_feature_columns_present(self):
        """Spatial neighbor features must be present."""
        features = build_features(
            weather_df=make_weather_df(n_cells=1, n_days=45),
            static_df=make_static_df(n_cells=1),
            fire_history_df=make_fire_history_df(),
            grid_df=make_grid_df(n_cells=1),
        )
        assert "neighbor_fire_count_7d" in features.columns, (
            "Missing spatial feature: neighbor_fire_count_7d"
        )


# ---------------------------------------------------------------------------
# Test 5: Static feature encoding
# ---------------------------------------------------------------------------


class TestStaticFeatures:
    """Verify static feature encoding and merging."""

    def test_aspect_encoded_as_sin_cos(self):
        """aspect_deg must be replaced by aspect_sin and aspect_cos."""
        static = make_static_df(n_cells=2)
        result = _prepare_static_features(static)
        assert "aspect_deg" not in result.columns, "aspect_deg should be dropped"
        assert "aspect_sin" in result.columns, "aspect_sin missing"
        assert "aspect_cos" in result.columns, "aspect_cos missing"

    def test_aspect_sin_cos_values_in_range(self):
        """aspect_sin and aspect_cos must be in [-1, 1]."""
        static = make_static_df(n_cells=10)
        result = _prepare_static_features(static)
        assert result["aspect_sin"].between(-1.0, 1.0).all(), "aspect_sin out of range"
        assert result["aspect_cos"].between(-1.0, 1.0).all(), "aspect_cos out of range"

    def test_categorical_fuel_model_encoded_as_integer(self):
        """fuel_model should be encoded as integer codes."""
        static = make_static_df(n_cells=3)
        result = _prepare_static_features(static)
        assert pd.api.types.is_integer_dtype(result["fuel_model"]), (
            "fuel_model must be encoded as integer."
        )

    def test_one_row_per_cell_in_static(self):
        """Static features must have exactly one row per unique cell_id."""
        static = make_static_df(n_cells=5)
        result = _prepare_static_features(static)
        assert result["cell_id"].nunique() == len(result), (
            "Static features must have exactly one row per cell_id."
        )
