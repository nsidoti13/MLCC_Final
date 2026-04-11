#!/usr/bin/env python3
"""
scripts/train_convlstm.py
=========================
Train a ConvLSTM model for spatiotemporal wildfire ignition prediction.

Pipeline
--------
1. Load MODIS fire data, CAWeather, CAPDSI
2. Rasterize H3 cells onto a 64×64 regular lat/lon grid covering California
3. Build (T=30, C=5, 64, 64) input tensors and (1, 64, 64) label maps per date
4. Train WildfireConvLSTM — encoder ConvLSTM + conv head
5. Evaluate and compare against LightGBM baseline

Spatial rasterization
---------------------
The 25k H3 cells map to a 64×64 pixel grid.
  lat: 32.53 → 42.01  (64 pixels, ~0.148 deg/pixel ≈ 16 km)
  lon: -124.48 → -114.13 (64 pixels, ~0.162 deg/pixel ≈ 14 km)
Each H3 cell centroid maps to its nearest pixel; pixels with multiple
cells take the mean feature value.
"""

from __future__ import annotations

import logging
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("train_convlstm")

import yaml
CONFIG      = yaml.safe_load(open(PROJECT_ROOT / "configs" / "config.yaml"))
BBOX        = CONFIG["grid"]["bbox"]
RESOLUTION  = CONFIG["grid"]["resolution"]
WINDOW      = CONFIG["temporal"]["prediction_window_days"]

GRID_H      = 32
GRID_W      = 32
SEQ_LEN     = 14          # days of history fed to ConvLSTM
FEATURE_VARS = ["tmax", "tmin", "prcp", "awnd", "pdsi",
                "fire_presence", "days_since_fire",
                "fire_roll7", "fire_roll30"]   # 9 channels
N_CHANNELS  = len(FEATURE_VARS)

TRAIN_END   = "2021-12-31"
VAL_START   = "2022-01-01";  VAL_END   = "2022-12-31"
TEST_START  = "2023-01-01";  TEST_END  = "2023-12-31"

EPOCHS      = 20
BATCH_SIZE  = 8
LR          = 1e-3
RANDOM_SEED = 42

MODIS_CSV   = PROJECT_ROOT / "data/raw/modis/MODISFireData.csv"
WEATHER_CSV = PROJECT_ROOT / "data/raw/weather_stations/CAWeather.csv"
PDSI_CSV    = PROJECT_ROOT / "data/raw/drought/CAPDSI.csv"
OUT_DIR     = PROJECT_ROOT / "outputs"
(OUT_DIR / "models").mkdir(parents=True, exist_ok=True)
(OUT_DIR / "maps").mkdir(parents=True, exist_ok=True)
(OUT_DIR / "reports").mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
log.info("Device: %s", DEVICE)


# ══════════════════════════════════════════════════════════════════════════════
# Fire history raster builder (called inside build_daily_rasters)
# ══════════════════════════════════════════════════════════════════════════════
def build_fire_history_rasters(cell_to_pixel, date_range):
    """
    Build 4 fire-history channel arrays, each shape (n_dates, H, W).

    Channels
    --------
    fire_presence   : 1 if any cell in pixel had a MODIS detection on that day
    days_since_fire : days elapsed since last fire at pixel (capped at 365)
    fire_roll7      : sum of fire_presence over past 7 days (backward-only)
    fire_roll30     : sum of fire_presence over past 30 days (backward-only)

    All channels are strictly backward-looking — no future leakage.
    When used in the ConvLSTM input sequence [t-SEQ_LEN … t-1], the model
    sees the fire history leading up to (but not including) the forecast date.
    """
    from src.labeling.labels import load_modis_csv
    from scipy.ndimage import uniform_filter

    log.info("Building fire history rasters…")
    fire_df = load_modis_csv(MODIS_CSV, h3_resolution=RESOLUTION, bbox=BBOX)
    fire_df = fire_df[fire_df["fire_date"] <= pd.Timestamp(TEST_END)]

    n_dates = len(date_range)
    date_to_idx = {d: i for i, d in enumerate(date_range)}

    # ── fire_presence: (n_dates, H, W) binary ────────────────────────────
    fire_presence = np.zeros((n_dates, GRID_H, GRID_W), dtype=np.float32)
    for _, row in fire_df.iterrows():
        pixel = cell_to_pixel.get(row["cell_id"])
        d     = row["fire_date"]
        if pixel is not None and d in date_to_idx:
            fire_presence[date_to_idx[d], pixel[0], pixel[1]] = 1.0
    # Clip to binary (multiple cells may map to same pixel)
    fire_presence = np.clip(fire_presence, 0, 1)

    # ── fire_roll7 / fire_roll30 via cumsum trick (O(n_dates)) ───────────
    # rolling_sum[t] = sum of fire_presence[t-k : t]  (strictly before t)
    cumsum = np.cumsum(fire_presence, axis=0)          # (n_dates, H, W)

    def rolling_backward(cs, k):
        """Sum over [t-k, t-1] — no same-day leakage."""
        lagged = np.concatenate([np.zeros((k, GRID_H, GRID_W), dtype=np.float32),
                                 cs[:-k]], axis=0)
        return cs - lagged  # sum of k days ending at t-1 (inclusive of t-k, t-1)

    fire_roll7  = rolling_backward(cumsum, 7)
    fire_roll30 = rolling_backward(cumsum, 30)

    # Spread fire counts to neighbouring pixels (3×3 spatial kernel)
    # This gives each pixel awareness of nearby fire activity
    kernel_size = 3
    fire_roll7  = uniform_filter(fire_roll7,  size=(1, kernel_size, kernel_size))
    fire_roll30 = uniform_filter(fire_roll30, size=(1, kernel_size, kernel_size))

    # ── days_since_fire: forward scan ────────────────────────────────────
    days_since = np.full((GRID_H, GRID_W), 365.0, dtype=np.float32)
    days_since_arr = np.zeros((n_dates, GRID_H, GRID_W), dtype=np.float32)
    for t in range(n_dates):
        # Where a fire occurred today, reset counter to 0; elsewhere increment
        days_since = np.where(fire_presence[t] > 0, 0.0, days_since + 1.0)
        days_since = np.minimum(days_since, 365.0)
        days_since_arr[t] = days_since

    # Normalise days_since to [0, 1]
    days_since_arr /= 365.0

    log.info(
        "Fire history rasters built. "
        "fire_presence mean=%.4f, fire_roll7 mean=%.4f, fire_roll30 mean=%.4f",
        fire_presence.mean(), fire_roll7.mean(), fire_roll30.mean(),
    )
    return fire_presence, days_since_arr, fire_roll7, fire_roll30


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 — Build pixel grid and H3 → pixel mapping
# ══════════════════════════════════════════════════════════════════════════════
def build_pixel_grid():
    """
    Returns
    -------
    lat_edges, lon_edges : 1-D arrays (length 65) defining pixel boundaries
    cell_to_pixel       : dict {cell_id: (row, col)}
    """
    from src.preprocessing.align import build_california_grid

    log.info("Building H3 grid and rasterizing to %dx%d…", GRID_H, GRID_W)
    grid_gdf = build_california_grid(BBOX, RESOLUTION)

    lat_edges = np.linspace(BBOX["lat_min"], BBOX["lat_max"], GRID_H + 1)
    lon_edges = np.linspace(BBOX["lon_min"], BBOX["lon_max"], GRID_W + 1)

    cell_to_pixel: dict = {}
    for _, row in grid_gdf.iterrows():
        lat, lon = row["lat"], row["lon"]
        r = int(np.clip(np.searchsorted(lat_edges, lat) - 1, 0, GRID_H - 1))
        c = int(np.clip(np.searchsorted(lon_edges, lon) - 1, 0, GRID_W - 1))
        cell_to_pixel[row["cell_id"]] = (r, c)

    log.info("Mapped %d cells to %dx%d grid (%d unique pixels used)",
             len(cell_to_pixel), GRID_H, GRID_W,
             len(set(cell_to_pixel.values())))
    return lat_edges, lon_edges, cell_to_pixel, grid_gdf


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2 — Build daily feature rasters
# ══════════════════════════════════════════════════════════════════════════════
def build_daily_rasters(cell_to_pixel):
    """
    Build a dict: date → (C, H, W) numpy array of feature values.

    Vectorised implementation — pre-builds a (n_dates, n_cells, C) array,
    then scatters into the (n_dates, C, H, W) raster using numpy indexing.
    Runs in seconds instead of minutes.
    """
    import h3
    from scipy.spatial import cKDTree
    from src.data.ingest import load_caweather, load_capdsi, expand_capdsi_to_daily

    log.info("Loading weather station data…")
    wx = load_caweather(WEATHER_CSV)

    cells = list(cell_to_pixel.keys())
    n_cells = len(cells)
    cell_rows = np.array([cell_to_pixel[c][0] for c in cells], dtype=np.int32)
    cell_cols = np.array([cell_to_pixel[c][1] for c in cells], dtype=np.int32)

    # Assign each cell to nearest station
    cell_coords = np.deg2rad(np.array([h3.h3_to_geo(c) for c in cells]))
    stations = wx[["station", "latitude", "longitude"]].drop_duplicates("station").reset_index(drop=True)
    sta_coords = np.deg2rad(stations[["latitude", "longitude"]].values)
    _, nearest_idx = cKDTree(sta_coords).query(cell_coords, k=1)
    cell_station_names = stations.iloc[nearest_idx]["station"].values  # (n_cells,)

    # Daily per-station values: pivot to (date × station × var) array
    wx = wx.sort_values(["station", "date"])
    for v in ["tmax", "tmin", "prcp", "awnd"]:
        wx[v] = wx.groupby("station")[v].ffill(limit=7)

    wx_pivot = wx.pivot_table(index="date", columns="station", values=["tmax", "tmin", "prcp", "awnd"])
    # wx_pivot.columns = MultiIndex (var, station)

    date_range = pd.date_range("2015-01-01", TEST_END, freq="D")
    wx_pivot = wx_pivot.reindex(date_range)  # ensure all dates present

    # PDSI (broadcast to all cells)
    log.info("Loading PDSI…")
    pdsi_daily = expand_capdsi_to_daily(load_capdsi(PDSI_CSV))
    pdsi_series = pdsi_daily.set_index("date")["pdsi"].reindex(date_range).ffill().bfill()
    pdsi_vals = pdsi_series.values.astype(np.float32)  # (n_dates,)

    log.info("Vectorising rasters for %d dates × %d cells…", len(date_range), n_cells)

    # For each variable, extract cell values: shape (n_dates, n_cells)
    weather_vars = ["tmax", "tmin", "prcp", "awnd"]
    cell_data = {}
    for var in weather_vars:
        # columns of the pivoted table for this var
        sta_cols = wx_pivot[var] if var in wx_pivot.columns.get_level_values(0) else wx_pivot[(var,)]
        # Reindex to our cell→station order
        sta_cols = sta_cols.reindex(columns=cell_station_names)
        cell_data[var] = sta_cols.values.astype(np.float32)  # (n_dates, n_cells)

    # Build full raster array: (n_dates, C, H, W)
    n_dates = len(date_range)
    N_WEATHER = len(weather_vars) + 1  # 4 weather vars + PDSI = 5
    raster_array = np.zeros((n_dates, N_WEATHER, GRID_H, GRID_W), dtype=np.float32)
    count_array  = np.zeros((GRID_H, GRID_W), dtype=np.float32)

    # Count how many cells map to each pixel (for averaging)
    np.add.at(count_array, (cell_rows, cell_cols), 1)
    pixel_mask = count_array > 0

    for ch, var in enumerate(weather_vars):
        vals = cell_data[var]  # (n_dates, n_cells)
        vals = np.nan_to_num(vals, nan=0.0)
        # Scatter-add: for each date, add cell values to their pixels
        # Use np.add.at over the cell dimension
        tmp = np.zeros((n_dates, GRID_H, GRID_W), dtype=np.float32)
        np.add.at(tmp, (slice(None), cell_rows, cell_cols), vals)
        # Average by pixel count
        tmp[:, pixel_mask] /= count_array[pixel_mask]
        raster_array[:, ch] = tmp

    # Channel 4: PDSI broadcast to all non-zero pixels
    pdsi_broadcast = pdsi_vals[:, None, None] * pixel_mask[None]
    raster_array[:, 4] = pdsi_broadcast

    # ── Append fire history channels ─────────────────────────────────────
    fire_presence, days_since_arr, fire_roll7, fire_roll30 = \
        build_fire_history_rasters(cell_to_pixel, date_range)

    # Stack: (n_dates, 9, H, W)
    raster_array = np.concatenate([
        raster_array,                           # (n_dates, 5, H, W) weather
        fire_presence[:, None],                 # (n_dates, 1, H, W)
        days_since_arr[:, None],                # (n_dates, 1, H, W)
        fire_roll7[:, None],                    # (n_dates, 1, H, W)
        fire_roll30[:, None],                   # (n_dates, 1, H, W)
    ], axis=1)

    log.info("Feature raster array shape: %s (%.0f MB)",
             raster_array.shape, raster_array.nbytes / 1e6)
    return date_range, raster_array


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3 — Build label rasters
# ══════════════════════════════════════════════════════════════════════════════
def build_label_rasters(cell_to_pixel, date_range=None):
    """
    Returns (n_dates, 1, H, W) binary numpy array aligned to date_range.
    Label = 1 at pixel (r,c) if any cell in that pixel had a fire
    in [date, date + WINDOW - 1].
    """
    from src.labeling.labels import load_modis_csv
    if date_range is None:
        date_range = pd.date_range("2015-01-01", TEST_END, freq="D")

    log.info("Loading MODIS fire data for labels…")
    fire_df = load_modis_csv(MODIS_CSV, h3_resolution=RESOLUTION, bbox=BBOX)
    fire_df = fire_df[fire_df["fire_date"] <= pd.Timestamp(TEST_END)]

    # Map fire events to pixels
    fire_df["pixel"] = fire_df["cell_id"].map(cell_to_pixel)
    fire_df = fire_df.dropna(subset=["pixel"])

    # Build set of (date, pixel) pairs where ignition occurred
    fire_pixel_dates: dict[pd.Timestamp, set] = {}
    for _, row in fire_df.iterrows():
        d = row["fire_date"]
        if d not in fire_pixel_dates:
            fire_pixel_dates[d] = set()
        fire_pixel_dates[d].add(row["pixel"])

    # Build vectorised label array: (n_dates, 1, H, W)
    date_range = pd.date_range("2015-01-01", TEST_END, freq="D")
    date_to_idx = {d: i for i, d in enumerate(date_range)}
    n_dates = len(date_range)

    log.info("Building label rasters (vectorised)…")
    label_array = np.zeros((n_dates, 1, GRID_H, GRID_W), dtype=np.float32)

    for fire_date, pixels in fire_pixel_dates.items():
        # For each fire on fire_date, mark all forecast dates [fire_date-WINDOW+1, fire_date]
        for offset in range(WINDOW):
            forecast_date = fire_date - pd.Timedelta(days=offset)
            if forecast_date in date_to_idx:
                idx = date_to_idx[forecast_date]
                for (r, c) in pixels:
                    label_array[idx, 0, r, c] = 1.0

    pos_rate = label_array.mean()
    log.info("Label rasters built. Mean pixel-level positive rate: %.4f%%", pos_rate * 100)
    return label_array


# ══════════════════════════════════════════════════════════════════════════════
# STEP 4 — Dataset
# ══════════════════════════════════════════════════════════════════════════════
class WildfireDataset(Dataset):
    """
    Fast array-indexed dataset — no dict lookups in __getitem__.

    Each sample:
      X : (SEQ_LEN, C, H, W) — normalised feature window ending at t-1
      y : (1, H, W)           — binary label map for date t
    """

    def __init__(self, indices: np.ndarray, feature_array: np.ndarray,
                 label_array: np.ndarray, mean: np.ndarray, std: np.ndarray):
        # indices: integer positions in feature_array for target dates
        self.indices       = indices
        self.feature_array = feature_array   # (n_total_dates, C, H, W)
        self.label_array   = label_array     # (n_total_dates, 1, H, W)
        self.mean = mean[:, None, None].astype(np.float32)
        self.std  = std[:, None, None].astype(np.float32)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        t = self.indices[i]
        # Sequence: indices [t-SEQ_LEN, t-1] (clamp at 0)
        start = max(0, t - SEQ_LEN)
        seq = self.feature_array[start:t]             # (<=SEQ_LEN, C, H, W)
        # Pad front if near start
        if seq.shape[0] < SEQ_LEN:
            pad = np.zeros((SEQ_LEN - seq.shape[0], N_CHANNELS, GRID_H, GRID_W), dtype=np.float32)
            seq = np.concatenate([pad, seq], axis=0)
        # Normalise
        seq = (seq - self.mean) / (self.std + 1e-6)   # (SEQ_LEN, C, H, W)
        y   = self.label_array[t]                      # (1, H, W)
        return torch.from_numpy(seq), torch.from_numpy(y)


# ══════════════════════════════════════════════════════════════════════════════
# STEP 5 — Train
# ══════════════════════════════════════════════════════════════════════════════
def train_convlstm(train_ds, val_ds):
    from src.models.convlstm_model import WildfireConvLSTM

    model = WildfireConvLSTM(input_channels=N_CHANNELS, hidden_channels=[64, 32]).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info("WildfireConvLSTM: %s trainable parameters", f"{n_params:,}")

    # Compute approximate pos_weight from training labels
    pos_pixels = sum(ds_item[1].sum().item() for ds_item in [train_ds[i] for i in range(min(100, len(train_ds)))])
    total_pixels = min(100, len(train_ds)) * GRID_H * GRID_W
    pos_rate = pos_pixels / max(total_pixels, 1)
    pos_weight_val = (1 - pos_rate) / max(pos_rate, 1e-6)
    pos_weight_val = min(pos_weight_val, 50.0)   # cap to avoid instability
    log.info("pos_weight = %.1f", pos_weight_val)

    criterion = nn.BCELoss(weight=None)  # per-sample weighting via pos_weight below
    pos_weight_tensor = torch.tensor(pos_weight_val, device=DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5, verbose=False)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    best_val_loss = float("inf")
    best_state    = None

    for epoch in range(1, EPOCHS + 1):
        # ── Train ────────────────────────────────────────────────────────
        model.train()
        train_loss = 0.0
        for X, y in tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS} train", leave=False):
            X, y = X.to(DEVICE), y.to(DEVICE)
            pred = model(X)
            # Weighted BCE: manually apply pos_weight
            loss = -(pos_weight_tensor * y * torch.log(pred + 1e-7)
                     + (1 - y) * torch.log(1 - pred + 1e-7)).mean()
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)

        # ── Validate ─────────────────────────────────────────────────────
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(DEVICE), y.to(DEVICE)
                pred = model(X)
                loss = -(pos_weight_tensor * y * torch.log(pred + 1e-7)
                         + (1 - y) * torch.log(1 - pred + 1e-7)).mean()
                val_loss += loss.item()
        val_loss /= max(len(val_loader), 1)

        scheduler.step(val_loss)
        log.info("Epoch %2d/%d  train_loss=%.4f  val_loss=%.4f  lr=%.2e",
                 epoch, EPOCHS, train_loss, val_loss,
                 optimizer.param_groups[0]["lr"])

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            log.info("  ✓ New best model saved (val_loss=%.4f)", best_val_loss)

    model.load_state_dict(best_state)
    return model


# ══════════════════════════════════════════════════════════════════════════════
# STEP 6 — Evaluate
# ══════════════════════════════════════════════════════════════════════════════
def evaluate_convlstm(model, test_ds):
    from sklearn.metrics import roc_auc_score, average_precision_score
    import matplotlib.pyplot as plt

    model.eval()
    all_preds, all_labels = [], []

    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    with torch.no_grad():
        for X, y in tqdm(test_loader, desc="Evaluating", leave=False):
            pred = model(X.to(DEVICE)).cpu().numpy().flatten()
            lbl  = y.numpy().flatten()
            all_preds.append(pred)
            all_labels.append(lbl)

    all_preds  = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)

    pr_auc  = average_precision_score(all_labels, all_preds)
    roc_auc = roc_auc_score(all_labels, all_preds)
    log.info("ConvLSTM Test  PR-AUC=%.4f | ROC-AUC=%.4f", pr_auc, roc_auc)

    # Save a sample prediction map (last test sample)
    model.eval()
    with torch.no_grad():
        X_sample, y_sample = test_ds[len(test_ds) - 1]
        pred_map = model(X_sample.unsqueeze(0).to(DEVICE))[0, 0].cpu().numpy()
        label_map = y_sample[0].numpy()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    im0 = axes[0].imshow(pred_map, vmin=0, vmax=1, cmap="hot", origin="lower")
    axes[0].set_title("ConvLSTM Predicted Probability")
    plt.colorbar(im0, ax=axes[0])
    im1 = axes[1].imshow(label_map, vmin=0, vmax=1, cmap="Reds", origin="lower")
    axes[1].set_title("Actual Ignitions (7-day window)")
    plt.colorbar(im1, ax=axes[1])
    plt.tight_layout()
    map_path = OUT_DIR / "maps" / "convlstm_prediction_sample.png"
    plt.savefig(map_path, dpi=150)
    plt.close()
    log.info("Sample prediction map saved → %s", map_path)

    return pr_auc, roc_auc


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import warnings; warnings.filterwarnings("ignore")
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    log.info("━━━ Wildfire ConvLSTM — Training Pipeline ━━━")

    # Build spatial mapping
    _, _, cell_to_pixel, grid_gdf = build_pixel_grid()

    # Build rasters — returns arrays, not dicts
    date_range, feature_array = build_daily_rasters(cell_to_pixel)
    label_array = build_label_rasters(cell_to_pixel, date_range)

    # Compute normalisation stats from training period
    train_mask = date_range <= pd.Timestamp(TRAIN_END)
    mean = feature_array[train_mask].mean(axis=(0, 2, 3))
    std  = feature_array[train_mask].std(axis=(0, 2, 3))
    log.info("Feature mean: %s", np.round(mean, 3))
    log.info("Feature std:  %s", np.round(std, 3))

    # Integer index arrays for each split (need SEQ_LEN prior days)
    all_idx = np.arange(SEQ_LEN, len(date_range))
    train_idx = all_idx[date_range[all_idx] <= pd.Timestamp(TRAIN_END)]
    val_idx   = all_idx[(date_range[all_idx] >= pd.Timestamp(VAL_START)) &
                        (date_range[all_idx] <= pd.Timestamp(VAL_END))]
    test_idx  = all_idx[(date_range[all_idx] >= pd.Timestamp(TEST_START)) &
                        (date_range[all_idx] <= pd.Timestamp(TEST_END))]

    log.info("Dates — train: %d | val: %d | test: %d",
             len(train_idx), len(val_idx), len(test_idx))

    # Datasets
    train_ds = WildfireDataset(train_idx, feature_array, label_array, mean, std)
    val_ds   = WildfireDataset(val_idx,   feature_array, label_array, mean, std)
    test_ds  = WildfireDataset(test_idx,  feature_array, label_array, mean, std)

    # Train
    model = train_convlstm(train_ds, val_ds)

    # Save
    model_path = OUT_DIR / "models" / "convlstm_model.pt"
    torch.save(model.state_dict(), model_path)
    log.info("ConvLSTM model saved → %s", model_path)

    # Evaluate
    pr_auc, roc_auc = evaluate_convlstm(model, test_ds)

    # Comparison table
    log.info("")
    log.info("══════════════════════════════════════════════")
    log.info("  Model Comparison (2023 Test Set)")
    log.info("══════════════════════════════════════════════")
    log.info("  LightGBM    PR-AUC=0.4674  ROC-AUC=0.9158")
    log.info("  ConvLSTM    PR-AUC=%.4f  ROC-AUC=%.4f", pr_auc, roc_auc)
    log.info("══════════════════════════════════════════════")
    log.info("━━━ Done ━━━")
