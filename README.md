# California Wildfire Ignition Prediction

Predict the **probability of wildfire ignition** across California over the next 7 days, at H3 hexagonal cell resolution (~4 km).

- **Unit of prediction**: (cell_id, date) pair
- **Target**: binary label — did any ignition occur in this cell within [t, t+7)?
- **Output**: P(ignition) per H3 cell per day

---

## Model Performance

Trained on 2015–2021, validated on 2022, tested on 2023. ~3.6M samples (4.8% positive rate).

| Model | PR-AUC | ROC-AUC | Recall@500 | Precision@500 |
|-------|--------|---------|------------|---------------|
| Logistic Regression | 0.484 | 0.924 | 0.029 | 0.958 |
| LightGBM | **0.531** | **0.937** | 0.024 | 0.788 |
| ConvLSTM | 0.360 | 0.915 | — | — |

**Baseline** (random classifier): PR-AUC ≈ positive_rate ≈ 0.048

LightGBM is the strongest model, benefiting most from the LANDFIRE vegetation features (`evt`, `fbfm40`). The ConvLSTM trails on PR-AUC due to resolution loss when rasterizing 25k cells to a 32×32 grid.

---

## Architecture

```
Raw Data Sources
┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐
│   MODIS FIRMS    │  │  CAWeather (NOAA │  │  CAPDSI Drought  │
│   CSV (NASA)     │  │  weather stations│  │  Index (monthly) │
└────────┬─────────┘  └────────┬─────────┘  └────────┬─────────┘
         │                     │                      │
         └─────────────────────┴──────────────────────┘
                               │
                    src/data/ingest.py
                    src/labeling/labels.py
                               │
                               ▼
                  H3 Resolution-6 Grid (25,028 cells)
                  Positive cell-days + 20x sampled negatives
                  ~3.6M training rows
                               │
                               ▼
                  32 Features per (cell, date):
                  ├── Weather rolling (7/30-day means, maxes, sums)
                  ├── PDSI drought index
                  ├── Temporal (doy/month sin+cos, year)
                  ├── Fire history (days_since_last_fire,
                  │   cell_fire_count_1yr, neighbor_fire_7d/30d)
                  ├── Spatial (lat, lon, dist_to_coast_km, is_inland)
                  └── Derived weather (temp_range, vpd_proxy,
                      hot_dry_windy, consecutive_dry_days)
                               │
               ┌───────────────┴───────────────┐
               │                               │
       LightGBM (primary)           ConvLSTM (spatiotemporal)
       scripts/train_model.py       scripts/train_convlstm.py
       outputs/models/lgbm_model.pkl  outputs/models/convlstm_model.pt
               │
               ▼
    scripts/visualize_predictions.py
    Folium interactive HTML risk map
    outputs/maps/wildfire_risk_YYYYMMDD.html
```

---

## Setup

### Prerequisites

- Python 3.10+
- macOS / Linux (tested on macOS 14+)

### Installation

```bash
git clone https://github.com/nsidoti13/MLCC_Final
cd MLCC_Final

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## Data Sources

| Source | File | Variables |
|--------|------|-----------|
| **NASA FIRMS** | `data/raw/modis/MODISFireData.csv` | Lat/lon fire detections, confidence, type |
| **NOAA GHCN** | `data/raw/weather/CAWeather.csv` | Tmax, tmin, prcp, wind speed (88 CA stations, 2015–2024) |
| **NOAA NCEI** | `data/raw/drought/CAPDSI.csv` | Palmer Drought Severity Index (monthly, 2015–2024) |
| **LANDFIRE LF2022** | `data/raw/landfire/landfire_features.parquet` | EVT, FBFM40, Canopy Cover (queried via ArcGIS ImageServer API) |

Place all three files in the paths above before running the pipeline.

---

## Running the Pipeline

### 1. Train tabular models (LightGBM + Logistic Regression)

```bash
source .venv/bin/activate
python scripts/train_model.py
```

Outputs: `outputs/models/lgbm_model.pkl`, `outputs/reports/model_metrics.json`, PR/ROC curve PNGs.

### 2. Train ConvLSTM (spatiotemporal)

```bash
python scripts/train_convlstm.py
```

Rasterizes the H3 grid to 32×32 arrays, trains a two-layer ConvLSTM over 7-day sequences for 20 epochs on MPS/CUDA/CPU.

Output: `outputs/models/convlstm_model.pt`

### 3. Generate interactive risk map

```bash
python scripts/visualize_predictions.py
```

Builds features for the target date (default: 2023-08-20), runs LightGBM inference on all 25,028 cells, and renders a Folium map with H3 hexagons colored by probability and actual MODIS fires overlaid.

Output: `outputs/maps/wildfire_risk_20230820.html`

---

## Features (32 total)

| Group | Features |
|-------|----------|
| **Weather rolling** | awnd/prcp/tmax/tmin rolling 7-day and 30-day means; tmax 7-day max; prcp 30-day sum |
| **Weather raw** | awnd, prcp, tmax, tmin (station-interpolated to H3 cell) |
| **Drought** | pdsi (Palmer Drought Severity Index, daily-expanded from monthly) |
| **Temporal** | doy_sin, doy_cos, month_sin, month_cos, year |
| **Fire history** | days_since_last_fire, cell_fire_count_1yr, neighbor_fire_7d, neighbor_fire_30d |
| **Spatial** | cell_lat, cell_lon, dist_to_coast_km, is_inland |
| **Derived weather** | temp_range, vpd_proxy, hot_dry_windy, consecutive_dry_days |
| **Vegetation (LANDFIRE)** | evt (Existing Vegetation Type), fbfm40 (Fire Behavior Fuel Model 40), canopy_cover |

All features are computed strictly from data prior to the forecast date — no future leakage.

---

## Key Design Decisions

### Time-based train/val/test split
Train 2015–2021 / Val 2022 / Test 2023. Random splits would leak future fire patterns into past training folds, inflating all metrics.

### PR-AUC as primary metric
At ~5% positive rate, ROC-AUC can be misleadingly high for poor classifiers. PR-AUC forces the model to actually identify positives; the random baseline is ~0.048.

### H3 hexagonal grid at resolution 6
H3 hexagons are spatially uniform with consistent 6-neighbor relationships. Resolution 6 (~4 km) matches MODIS thermal anomaly resolution. Each cell has area ≈ 36 km².

### Negative sampling (20:1 ratio)
With 25,028 cells × 3,285 days = 82M possible (cell, date) pairs, full materialization is infeasible. We keep all positives and sample 20× negatives (~3.6M rows total).

### Leakage-safe fire history
`bisect_left` finds events strictly before each forecast date. Rolling neighbor counts use cumsum arithmetic — O(n), no per-cell loops.

---

## Directory Structure

```
MLCC_Final/
├── configs/
│   └── config.yaml              # H3 resolution, CA bbox, train/val/test splits
├── data/
│   └── raw/
│       ├── modis/MODISFireData.csv
│       ├── weather/CAWeather.csv
│       └── drought/CAPDSI.csv
├── notebooks/
│   └── 01_eda.ipynb
├── outputs/
│   ├── maps/                    # Folium HTML risk maps
│   ├── models/                  # lgbm_model.pkl, convlstm_model.pt
│   └── reports/                 # Metrics JSON, PR/ROC curve PNGs
├── scripts/
│   ├── train_model.py           # LightGBM + Logistic Regression pipeline
│   ├── train_convlstm.py        # ConvLSTM spatiotemporal pipeline
│   ├── eval_convlstm.py         # Standalone ConvLSTM evaluation on test set
│   └── visualize_predictions.py # Folium interactive risk map
├── src/
│   ├── data/
│   │   ├── download.py          # gridMET, NDFD, ERA5 download utilities
│   │   ├── ingest.py            # load_caweather(), load_capdsi(), weather interpolation
│   │   └── landfire.py          # LANDFIRE EVT/FBFM40/CC via ArcGIS ImageServer API
│   ├── labeling/
│   │   └── labels.py            # MODIS CSV ingestion, H3 deduplication, cell-day labels
│   ├── models/
│   │   └── convlstm_model.py    # ConvLSTMCell, ConvLSTM, WildfireConvLSTM (PyTorch)
│   └── preprocessing/
│       └── align.py             # H3 grid construction, H3 boundary polygon helpers
├── tests/
├── requirements.txt
└── README.md
```

---

## Running Tests

```bash
source .venv/bin/activate
python -m pytest tests/ -v
```
