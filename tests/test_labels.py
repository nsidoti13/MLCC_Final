"""
tests/test_labels.py
====================
Unit tests for label construction in src/labeling/labels.py.

Test coverage:
  1. No future leakage: labels at date t never see fires after t + window - 1.
  2. Only first ignition counted: repeated fires in cooldown window are filtered.
  3. Correct window size: label=1 exactly for dates in [fire_date - (window-1), fire_date].
  4. Label boundary trimming: trim_label_boundary removes the correct rows.
  5. Empty fire DataFrame produces all-zero labels.
"""

from __future__ import annotations

import pandas as pd
import pytest
import sys
from pathlib import Path

# Ensure project root is on the path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.labeling.labels import (
    build_label_dataframe,
    trim_label_boundary,
    _filter_first_ignitions,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def small_grid() -> pd.DataFrame:
    """A minimal grid with 3 cells."""
    return pd.DataFrame({"cell_id": ["cell_A", "cell_B", "cell_C"]})


@pytest.fixture()
def single_fire() -> pd.DataFrame:
    """A single ignition event at cell_A on 2020-07-15."""
    return pd.DataFrame(
        {
            "fire_date": ["2020-07-15"],
            "cell_id": ["cell_A"],
        }
    )


# ---------------------------------------------------------------------------
# Test 1: No future data leakage
# ---------------------------------------------------------------------------


class TestNoFutureLeakage:
    """Verify that label assignment never uses information from after t + window - 1."""

    def test_label_only_set_within_window(self, small_grid, single_fire):
        """
        Given a fire at 2020-07-15, the label for date t should be 1
        only when t is in [2020-07-09, 2020-07-15] (window=7).

        Crucially: dates AFTER 2020-07-15 must have label=0 because those
        forecasts could not have predicted the fire without using future data.
        """
        window = 7
        label_df = build_label_dataframe(
            fire_df=single_fire,
            grid_df=small_grid,
            window=window,
            start_date="2020-07-01",
            end_date="2020-07-31",
        )

        cell_a = label_df[label_df["cell_id"] == "cell_A"].set_index("date")

        fire_date = pd.Timestamp("2020-07-15")

        # All dates strictly after the fire date must be 0
        after_fire = cell_a[cell_a.index > fire_date]
        assert (after_fire["label"] == 0).all(), (
            "Labels after fire_date must be 0 — forecasts on those dates "
            "cannot predict a fire that already happened."
        )

    def test_label_zero_before_window(self, small_grid, single_fire):
        """
        Dates more than (window - 1) days before the fire date should be 0.
        A forecast on 2020-07-08 with a 7-day window covers [7/8, 7/14],
        which does NOT include 7/15.
        """
        window = 7
        label_df = build_label_dataframe(
            fire_df=single_fire,
            grid_df=small_grid,
            window=window,
            start_date="2020-07-01",
            end_date="2020-07-31",
        )

        cell_a = label_df[label_df["cell_id"] == "cell_A"].set_index("date")

        # 2020-07-08 is exactly (window) days before fire date — should be 0
        cutoff = pd.Timestamp("2020-07-15") - pd.Timedelta(days=window)
        before_window = cell_a[cell_a.index <= cutoff]
        assert (before_window["label"] == 0).all(), (
            f"Labels more than {window - 1} days before fire_date must be 0."
        )

    def test_label_one_at_fire_date(self, small_grid, single_fire):
        """Label on the exact fire_date should be 1 (window starts same day)."""
        window = 7
        label_df = build_label_dataframe(
            fire_df=single_fire,
            grid_df=small_grid,
            window=window,
            start_date="2020-07-01",
            end_date="2020-07-31",
        )
        cell_a = label_df[label_df["cell_id"] == "cell_A"].set_index("date")
        fire_date = pd.Timestamp("2020-07-15")
        assert cell_a.loc[fire_date, "label"] == 1, (
            "Label on the fire_date itself must be 1."
        )

    def test_unaffected_cells_always_zero(self, small_grid, single_fire):
        """Cells with no fires must have all-zero labels."""
        window = 7
        label_df = build_label_dataframe(
            fire_df=single_fire,
            grid_df=small_grid,
            window=window,
            start_date="2020-07-01",
            end_date="2020-07-31",
        )
        for cell in ["cell_B", "cell_C"]:
            cell_labels = label_df[label_df["cell_id"] == cell]["label"]
            assert (cell_labels == 0).all(), (
                f"Cell {cell} had no fires but has non-zero labels."
            )


# ---------------------------------------------------------------------------
# Test 2: Only first ignition counted
# ---------------------------------------------------------------------------


class TestFirstIgnitionOnly:
    """Verify that the cooldown filter removes spread events."""

    def test_repeated_fires_in_cooldown_filtered(self):
        """
        Two fires 5 days apart in the same cell should count as one ignition
        (cooldown = 30 days).
        """
        fire_df = pd.DataFrame(
            {
                "fire_date": ["2020-07-10", "2020-07-15"],
                "cell_id": ["cell_A", "cell_A"],
            }
        )
        filtered = _filter_first_ignitions(fire_df.assign(fire_date=pd.to_datetime(fire_df["fire_date"])), cooldown_days=30)
        assert len(filtered) == 1, (
            "Second fire within 30-day cooldown should be treated as spread and filtered."
        )
        assert filtered.iloc[0]["fire_date"] == pd.Timestamp("2020-07-10"), (
            "First chronological fire should be retained."
        )

    def test_fires_after_cooldown_both_kept(self):
        """
        Two fires 45 days apart should both be counted as independent ignitions.
        """
        fire_df = pd.DataFrame(
            {
                "fire_date": ["2020-06-01", "2020-07-16"],
                "cell_id": ["cell_A", "cell_A"],
            }
        )
        filtered = _filter_first_ignitions(
            fire_df.assign(fire_date=pd.to_datetime(fire_df["fire_date"])),
            cooldown_days=30,
        )
        assert len(filtered) == 2, (
            "Fires more than 30 days apart should both be kept."
        )

    def test_different_cells_independent(self):
        """
        Fires on consecutive days in different cells should both be retained.
        """
        fire_df = pd.DataFrame(
            {
                "fire_date": ["2020-07-10", "2020-07-11"],
                "cell_id": ["cell_A", "cell_B"],
            }
        )
        filtered = _filter_first_ignitions(
            fire_df.assign(fire_date=pd.to_datetime(fire_df["fire_date"])),
            cooldown_days=30,
        )
        assert len(filtered) == 2, (
            "Fires in different cells are independent; both should be retained."
        )

    def test_label_df_only_first_ignition_positive(self, small_grid):
        """
        With two fires in the same cell within cooldown, the label window
        should reflect only the first fire.
        """
        fire_df = pd.DataFrame(
            {
                "fire_date": ["2020-07-10", "2020-07-12"],
                "cell_id": ["cell_A", "cell_A"],
            }
        )
        grid = pd.DataFrame({"cell_id": ["cell_A"]})
        window = 7
        label_df = build_label_dataframe(
            fire_df=fire_df,
            grid_df=grid,
            window=window,
            start_date="2020-07-01",
            end_date="2020-07-31",
        )

        # Only the window around the FIRST fire (2020-07-10) should be positive
        cell_a = label_df.set_index("date")
        first_fire_window_start = pd.Timestamp("2020-07-04")  # 10 - 6

        # 2020-07-12 should NOT independently extend the window since it's
        # a spread event.  Only dates in [7/4, 7/10] should be labeled 1.
        # Dates [7/11, 7/31] should be 0.
        assert cell_a.loc["2020-07-10", "label"] == 1
        assert cell_a.loc["2020-07-31", "label"] == 0


# ---------------------------------------------------------------------------
# Test 3: Correct window size
# ---------------------------------------------------------------------------


class TestWindowSize:
    """Verify that the label=1 window is exactly `window` days wide."""

    @pytest.mark.parametrize("window", [1, 3, 7, 14])
    def test_label_window_exactly_window_days(self, window):
        """
        For a given window, exactly `window` consecutive dates should have
        label=1 for the affected cell (assuming no date-range truncation).
        """
        fire_df = pd.DataFrame(
            {
                "fire_date": ["2020-07-15"],
                "cell_id": ["cell_A"],
            }
        )
        grid = pd.DataFrame({"cell_id": ["cell_A"]})

        # Give plenty of date range so truncation is not an issue
        label_df = build_label_dataframe(
            fire_df=fire_df,
            grid_df=grid,
            window=window,
            start_date="2020-06-01",
            end_date="2020-08-31",
        )

        cell_a_labels = label_df[label_df["cell_id"] == "cell_A"]["label"]
        n_positive = cell_a_labels.sum()

        assert n_positive == window, (
            f"Expected exactly {window} positive labels for window={window}, "
            f"but got {n_positive}."
        )

    def test_window_boundary_inclusive(self):
        """
        Both the start and end of the window should be label=1.

        For window=7 and fire_date=2020-07-15:
        - 2020-07-09 (fire_date - 6) should be 1  (earliest forecast to predict)
        - 2020-07-15 (fire_date)      should be 1  (latest forecast to predict)
        """
        fire_df = pd.DataFrame(
            {"fire_date": ["2020-07-15"], "cell_id": ["cell_A"]}
        )
        grid = pd.DataFrame({"cell_id": ["cell_A"]})
        window = 7

        label_df = build_label_dataframe(
            fire_df=fire_df,
            grid_df=grid,
            window=window,
            start_date="2020-07-01",
            end_date="2020-08-01",
        )

        cell_a = label_df[label_df["cell_id"] == "cell_A"].set_index("date")
        window_start = pd.Timestamp("2020-07-09")
        window_end = pd.Timestamp("2020-07-15")

        assert cell_a.loc[window_start, "label"] == 1, "Window start date must be 1."
        assert cell_a.loc[window_end, "label"] == 1, "Window end date (fire_date) must be 1."
        assert cell_a.loc["2020-07-08", "label"] == 0, "Day before window must be 0."
        assert cell_a.loc["2020-07-16", "label"] == 0, "Day after fire_date must be 0."


# ---------------------------------------------------------------------------
# Test 4: Label boundary trimming
# ---------------------------------------------------------------------------


class TestLabelBoundaryTrimming:
    """Verify trim_label_boundary removes the correct rows."""

    def test_trim_removes_last_window_minus_1_days(self):
        """
        A training split ending 2021-12-31 with window=7 should have dates
        up to 2021-12-25 (31 - 6 days) in the trimmed output.
        """
        dates = pd.date_range("2021-01-01", "2021-12-31", freq="D")
        df = pd.DataFrame({"cell_id": "cell_A", "date": dates, "label": 0})
        window = 7
        split_end = "2021-12-31"

        trimmed = trim_label_boundary(df, split_end, window=window)

        expected_last_date = pd.Timestamp("2021-12-31") - pd.Timedelta(days=window - 1)
        assert trimmed["date"].max() == expected_last_date, (
            f"Trimmed data should end at {expected_last_date}, "
            f"got {trimmed['date'].max()}."
        )

    def test_trim_correct_row_count(self):
        """Trimmed DataFrame should have len(original) - (window-1) rows."""
        n_days = 100
        window = 7
        dates = pd.date_range("2021-01-01", periods=n_days, freq="D")
        df = pd.DataFrame({"cell_id": "cell_A", "date": dates, "label": 0})

        trimmed = trim_label_boundary(df, dates[-1].strftime("%Y-%m-%d"), window=window)
        expected_rows = n_days - (window - 1)
        assert len(trimmed) == expected_rows, (
            f"Expected {expected_rows} rows after trim, got {len(trimmed)}."
        )


# ---------------------------------------------------------------------------
# Test 5: Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Edge case and robustness tests."""

    def test_empty_fire_df_all_zero_labels(self, small_grid):
        """An empty fire DataFrame should produce all-zero labels."""
        fire_df = pd.DataFrame(columns=["fire_date", "cell_id"])
        label_df = build_label_dataframe(
            fire_df=fire_df,
            grid_df=small_grid,
            window=7,
            start_date="2020-01-01",
            end_date="2020-01-31",
        )
        assert (label_df["label"] == 0).all(), (
            "No fires should produce all-zero labels."
        )

    def test_output_columns(self, small_grid, single_fire):
        """Output DataFrame must have exactly cell_id, date, label columns."""
        label_df = build_label_dataframe(
            fire_df=single_fire,
            grid_df=small_grid,
            window=7,
            start_date="2020-07-01",
            end_date="2020-07-31",
        )
        assert set(label_df.columns) == {"cell_id", "date", "label"}, (
            f"Expected columns {{cell_id, date, label}}, got {set(label_df.columns)}."
        )

    def test_label_dtype_is_integer(self, small_grid, single_fire):
        """Label column must be integer (0/1), not float."""
        label_df = build_label_dataframe(
            fire_df=single_fire,
            grid_df=small_grid,
            window=7,
            start_date="2020-07-01",
            end_date="2020-07-31",
        )
        assert pd.api.types.is_integer_dtype(label_df["label"]), (
            "Label column must be integer dtype."
        )

    def test_missing_fire_date_column_raises(self, small_grid):
        """Missing fire_date column should raise ValueError."""
        bad_fire_df = pd.DataFrame({"wrong_col": ["2020-07-15"], "cell_id": ["cell_A"]})
        with pytest.raises(ValueError, match="missing columns"):
            build_label_dataframe(
                fire_df=bad_fire_df,
                grid_df=small_grid,
                window=7,
                start_date="2020-07-01",
                end_date="2020-07-31",
            )

    def test_missing_cell_id_in_grid_raises(self, single_fire):
        """Missing cell_id column in grid_df should raise ValueError."""
        bad_grid = pd.DataFrame({"wrong_col": ["cell_A", "cell_B"]})
        with pytest.raises(ValueError, match="missing column"):
            build_label_dataframe(
                fire_df=single_fire,
                grid_df=bad_grid,
                window=7,
                start_date="2020-07-01",
                end_date="2020-07-31",
            )
