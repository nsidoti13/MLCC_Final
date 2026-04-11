# California Wildfire Ignition Prediction

Predict the **probability of wildfire ignition** across California over the next 7 days, at H3 hexagonal cell resolution (~4 km).

- **Unit of prediction**: (cell_id, date) pair
- **Target**: binary label — did any ignition occur in this cell within [t, t+7)?
- **Output**: P(ignition) per H3 cell per day

---

## Architecture Diagram

```
Raw Data Sources
┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌─────────────┐
│  MODIS Fire  │  │ NOAA CFSv2   │  │  LANDFIRE    │  │  USGS DEM   │
│  MOD14A1     │  │  7-day fcst  │  │  Fuels/Veg   │  │  Terrain    │
└──────┬───────┘  └──────┬───────┘  └──────┬───────┘  └──────┬──────┘
       │                 │                  │                  │
       └─────────────────┴──────────────────┴──────────────────┘
                                   │
                         src/data/download.py
                                   │
                                   ▼
                        ┌──────────────────────┐
                        │  data/raw/           │
                        │  (HDF, GRIB2, TIF)   │
                        └──────────┬───────────┘
                                   │
                     src/preprocessing/align.py
                     (H3 grid, reproject, fill)
                                   │
                                   ▼
                        ┌──────────────────────┐
                        │  data/interim/       │
                        │  ca_grid.parquet     │
                        │  weather_aligned.p   │
                        │  static_features.p   │
                        └────────┬─────────────┘
                                 │
              ┌──────────────────┴──────────────────┐
              │                                     │
    src/labeling/labels.py               src/features/engineer.py
    (cell_id, date, label)               (forecast, lagged, static,
    No future leakage                     human, spatial, temporal)
              │                                     │
              └──────────────────┬──────────────────┘
                                 │
                      data/processed/
                      features_{train,val,test}.parquet
                      labels_{train,val,test}.parquet
                                 │
                                 ▼
                      src/modeling/trainer.py
                      TimeSeriesSplit CV
                      scale_pos_weight auto
                                 │
                    ┌────────────┴────────────┐
                    │                         │
            LGBMModel                    XGBModel
            (primary)                 (comparison)
                    │                         │
                    └────────────┬────────────┘
                                 │
                      src/evaluation/metrics.py
                      PR-AUC, ROC-AUC, Recall@k
                      Precision-Recall curves
                                 │
                                 ▼
                      src/inference/predict.py
                      outputs/predictions/*.parquet
                      outputs/predictions/*.geojson
                      outputs/maps/
```

---

## Setup

### Prerequisites

- Python 3.10+
- macOS / Linux (tested on macOS 14+)

### Installation

```bash
# Clone or navigate to project
cd "/path/to/MLCC_Final"

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install all dependencies
pip install -r requirements.txt
```

### Verify installation

```bash
python -c "import h3, lightgbm, xgboost, geopandas; print('All packages OK')"
```

---

## Data Sources

| Source | Product | Variables | License |
|--------|---------|-----------|---------|
| **NASA EOSDIS LPDAAC** | MODIS MOD14A1 v061 | Daily fire mask, fire radiative power | Free (requires EarthData login) |
| **NOAA NCEI** | CFSv2 operational forecasts | tmp2m, rh2m, wnd10m, apcp (7-day) | Free/public |
| **USGS LANDFIRE** | LF2022 (v2.2.0) | EVT, FBFM40, CBD, CBH, CC, CH | Free/public |
| **USGS 3DEP** | 1/3 arc-second DEM | Elevation, slope, aspect | Free/public |
| **US Census** | TIGER/Line 2023, ACS 5yr | Roads, population density | Free/public |
| **HIFLD** | Electric Transmission Lines | Powerline locations | Free/public |

### Authentication setup

```bash
# NASA EarthData (for MODIS)
echo "machine urs.earthdata.nasa.gov login YOUR_USER password YOUR_PASS" >> ~/.netrc
chmod 600 ~/.netrc

# Census API (for population data)
export CENSUS_API_KEY="your_key_here"
# Get a free key at: https://api.census.gov/data/key_signup.html
```

---

## Pipeline Walkthrough

### Stage 1: Data Download

Downloads all raw data sources. Skippable with `--skip-download` if data already exists.

```bash
python scripts/run_pipeline.py
# or selectively:
python scripts/run_pipeline.py --start-year 2020 --end-year 2023
```

**Output**: `data/raw/{modis,noaa,landfire,terrain,human}/`

### Stage 2: Preprocessing & Alignment

Reprojects all data sources onto the H3 resolution-6 grid. Handles:
- Raster reprojection to WGS-84
- Aggregation of pixels to H3 cells
- Temporal resampling to daily resolution
- Missing value imputation (median fill + missingness indicators)

**Output**: `data/interim/ca_grid.parquet`, `weather_aligned.parquet`, `static_features.parquet`

### Stage 3: Label Construction

Builds the `(cell_id, date, label)` binary target DataFrame.

Critical design choices (see `src/labeling/labels.py`):
- Only **first ignitions** counted (30-day cooldown for spread)
- Forward-window only: label at t uses fires in [t, t+6]
- Boundary trimming to prevent train/val leakage

**Output**: `data/processed/labels_{train,val,test}.parquet`

### Stage 4: Feature Engineering

Constructs six feature groups:

| Group | Features | Count |
|-------|----------|-------|
| Forecast | tmp2m_max_7d, rh2m_min_7d, wnd10m_max_7d, apcp_sum_7d | 4 |
| Lagged rolling | 7/30-day rolling means/sums | 6 |
| Static | elevation, slope, aspect (sin/cos), fuel model, vegetation, canopy | 8 |
| Human | road density, dist to powerline, population density | 3 |
| Spatial | neighbor fire count, dryness index, wind alignment | 3 |
| Temporal | doy_sin/cos, month_sin/cos, week_of_year | 5 |

**Output**: `data/processed/features_{train,val,test}.parquet`

### Stage 5: Training

Trains LightGBM (primary) or XGBoost with:
- Automatic `scale_pos_weight = neg_count / pos_count`
- TimeSeriesSplit cross-validation (3 folds) for diagnostic PR-AUC
- Early stopping on validation PR-AUC

```bash
python scripts/run_pipeline.py --skip-download --skip-preprocess --skip-labels --skip-features
# Or choose a different model:
python scripts/run_pipeline.py --skip-download ... --model-type xgboost
```

**Output**: `outputs/models/lgbm_model.pkl`

### Stage 6: Evaluation

```bash
python scripts/run_pipeline.py --skip-download --skip-preprocess \
    --skip-labels --skip-features --skip-train
```

Produces:
- `outputs/reports/lgbm_val_metrics.json`
- `outputs/reports/lgbm_val_pr_curve.png`
- `outputs/reports/lgbm_val_roc_curve.png`

### Stage 7: Inference

```bash
python scripts/run_pipeline.py --skip-download --skip-preprocess \
    --skip-labels --skip-features --skip-train --skip-evaluate
# Inference for a specific date:
python scripts/run_pipeline.py ... --inference-date 2023-08-15
```

**Output**: `outputs/predictions/predictions_*.parquet`, `*.geojson`, `top_risk_cells_*.csv`

---

## Model Performance (Placeholder)

Fill this table after running the full pipeline on real data.

| Model | Split | PR-AUC | ROC-AUC | Recall@500 | Precision@500 |
|-------|-------|--------|---------|------------|---------------|
| Logistic Regression | Val 2022 | — | — | — | — |
| LightGBM | Val 2022 | — | — | — | — |
| XGBoost | Val 2022 | — | — | — | — |
| LightGBM | Test 2023 | — | — | — | — |
| XGBoost | Test 2023 | — | — | — | — |

**Baseline** (random classifier): PR-AUC ≈ positive_rate ≈ 0.008

---

## Running Tests

```bash
source .venv/bin/activate
python -m pytest tests/ -v
```

Tests cover:
- `tests/test_labels.py` — leakage prevention, first-ignition logic, window correctness
- `tests/test_features.py` — sin/cos encoding, rolling windows, NaN-free output

---

## Directory Structure

```
MLCC_Final/
├── configs/
│   └── config.yaml              # Master configuration (grid, splits, hyperparams)
├── data/
│   ├── raw/                     # Original downloaded files (HDF, GRIB2, TIF, ZIP)
│   │   ├── modis/
│   │   ├── noaa/
│   │   ├── landfire/
│   │   ├── terrain/
│   │   └── human/
│   ├── interim/                 # H3-aligned, cleaned intermediate files
│   └── processed/               # Final feature matrices and labels
├── notebooks/
│   └── 01_eda.ipynb             # Exploratory Data Analysis
├── outputs/
│   ├── maps/                    # Risk map visualisations
│   ├── models/                  # Serialised model files
│   ├── predictions/             # Inference outputs (parquet + geojson)
│   └── reports/                 # Evaluation plots and metrics JSON
├── scripts/
│   └── run_pipeline.py          # End-to-end pipeline runner (argparse)
├── src/
│   ├── data/
│   │   └── download.py          # MODIS, NOAA, LANDFIRE, terrain, human downloads
│   ├── preprocessing/
│   │   └── align.py             # H3 grid construction, raster alignment, imputation
│   ├── labeling/
│   │   └── labels.py            # (cell_id, date, label) construction, leakage guard
│   ├── features/
│   │   └── engineer.py          # All 6 feature groups; returns wide feature DataFrame
│   ├── modeling/
│   │   └── trainer.py           # train_model(), run_cross_validation(), TimeSeriesSplit
│   ├── models/
│   │   ├── lgbm_model.py        # LGBMModel wrapper (fit/predict/save/load)
│   │   └── xgb_model.py         # XGBModel wrapper (fit/predict/save/load)
│   ├── evaluation/
│   │   └── metrics.py           # evaluate(), PR-AUC, ROC-AUC, recall@k, plots
│   └── inference/
│       └── predict.py           # run_inference(), GeoJSON/parquet output
├── tests/
│   ├── test_labels.py           # Unit tests: leakage, first ignition, window size
│   └── test_features.py         # Unit tests: sin/cos, rolling, NaN-free
├── .venv/                       # Virtual environment (not committed)
├── requirements.txt
└── README.md
```

---

## Key Design Decisions

### 1. Time-based train/val/test split (not random)

**Why**: Wildfire ignition has strong temporal autocorrelation. A random split would allow the model to "see" 2022 patterns while predicting 2021 — inflating all metrics. We use strict year cutoffs: train 2015–2021, val 2022, test 2023.

**Implication**: The last `window-1` rows of each training split are trimmed (`trim_label_boundary`) because their labels depend on the next period.

### 2. PR-AUC as primary metric

**Why**: At < 1% positive rate, a classifier that predicts "never ignite" for every cell achieves > 99% accuracy and ROC-AUC ≈ 0.5–0.6. PR-AUC forces the model to actually identify positives; a random baseline achieves PR-AUC ≈ positive_rate ≈ 0.008. Any useful model must substantially exceed this.

### 3. H3 hexagonal grid at resolution 6

**Why**: H3 hexagons are spatially uniform (no pole distortion), have consistent neighbor relationships (each cell has exactly 6 neighbors), and resolution 6 (~4 km) matches the spatial resolution of MODIS thermal anomalies and NOAA forecast grids. Square grids have inconsistent diagonal vs. cardinal distances.

### 4. scale_pos_weight instead of oversampling

**Why**: At 1% positive rate, SMOTE or random oversampling inflates the effective dataset size 100x, causing slow training and potentially unrealistic synthetic positives in feature space. `scale_pos_weight` adjusts the loss function directly — equivalent to weighted sampling but computationally free.

### 5. First-ignition-only labeling

**Why**: We predict *new* ignitions, not fire spread. If a cell burns on Day 1 and a fire spreads to it again on Day 5, the Day-5 detection is fire spread, not a new human- or lightning-caused ignition. We apply a 30-day cooldown between independent ignitions per cell.

### 6. Rolling cross-validation with TimeSeriesSplit

**Why**: Standard k-fold CV leaks future fire patterns into past training folds. TimeSeriesSplit guarantees each validation fold is strictly after its training fold, giving honest CV scores.

---

## Pitfalls Avoided

| Pitfall | How We Avoid It |
|---------|-----------------|
| **Future leakage in labels** | Labels use only forward window [t, t+6]; no fires after t+6 influence label at t |
| **Future leakage at split boundary** | `trim_label_boundary()` removes last (window-1) days from training split |
| **Fire spread counted as new ignition** | 30-day cooldown filter in `_filter_first_ignitions()` |
| **Random CV leaking temporal patterns** | Only `TimeSeriesSplit` used — never `KFold` or `StratifiedKFold` |
| **ROC-AUC misleading for imbalanced data** | PR-AUC is the primary metric throughout |
| **Aspect discontinuity at 0°/360°** | Aspect encoded as (sin, cos) pair — no discontinuity |
| **Global package installation** | All packages installed in `.venv` only |
| **Including current day in rolling mean** | `shift(1)` before rolling ensures strictly past window |
| **Leaking H3 neighbors' future fires** | Spatial features look only backward: fires in [date-7, date-1] |
