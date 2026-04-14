#!/usr/bin/env python3
"""
scripts/visualize_predictions.py
=================================
Run the trained LightGBM model on a target date and render an interactive
Folium map showing 7-day wildfire ignition probability per H3 cell.

Output: outputs/maps/wildfire_risk_<date>.html
"""

from __future__ import annotations

import bisect
import logging
import math
import pickle
import sys
from pathlib import Path

import h3
import numpy as np
import pandas as pd
import folium
from folium.plugins import FloatImage
import branca.colormap as cm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("visualize")

import yaml
CONFIG     = yaml.safe_load(open(PROJECT_ROOT / "configs" / "config.yaml"))
BBOX       = CONFIG["grid"]["bbox"]
RESOLUTION = CONFIG["grid"]["resolution"]
LAG_SHORT  = CONFIG["temporal"]["lag_days_short"]
LAG_LONG   = CONFIG["temporal"]["lag_days_long"]

MODIS_CSV   = PROJECT_ROOT / "data/raw/modis/MODISFireData.csv"
WEATHER_CSV = PROJECT_ROOT / "data/raw/weather_stations/CAWeather.csv"
PDSI_CSV    = PROJECT_ROOT / "data/raw/drought/CAPDSI.csv"
MODEL_PATH  = PROJECT_ROOT / "outputs/models/lgbm_model.pkl"
OUT_DIR     = PROJECT_ROOT / "outputs/maps"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Target date ───────────────────────────────────────────────────────────────
TARGET_DATE = pd.Timestamp("2023-08-20")   # peak fire day in 2023 test set


# ══════════════════════════════════════════════════════════════════════════════
# Feature builders (mirrors train_model.py logic)
# ══════════════════════════════════════════════════════════════════════════════

def build_features_for_date(target_date: pd.Timestamp, grid_df) -> pd.DataFrame:
    """Build a feature row for every H3 cell on target_date."""
    from scipy.spatial import cKDTree
    from src.data.ingest import load_caweather, load_capdsi, expand_capdsi_to_daily
    from src.labeling.labels import load_modis_csv

    log.info("Building features for %s across %d cells…", target_date.date(), len(grid_df))

    # ── Weather: rolling features from station data ───────────────────────
    wx = load_caweather(WEATHER_CSV)
    wx = wx.sort_values(["station", "date"])
    for v in ["awnd", "prcp", "tmax", "tmin"]:
        wx[v] = wx.groupby("station")[v].ffill(limit=7)

    # Assign each grid cell to nearest station
    stations = wx[["station","latitude","longitude"]].drop_duplicates("station").reset_index(drop=True)
    cell_ids  = grid_df["cell_id"].values
    cell_coords = np.deg2rad(np.array([h3.h3_to_geo(c) for c in cell_ids]))
    sta_coords  = np.deg2rad(stations[["latitude","longitude"]].values)
    _, nn_idx   = cKDTree(sta_coords).query(cell_coords, k=1)
    cell_to_station = dict(zip(cell_ids, stations.iloc[nn_idx]["station"].values))

    # Compute rolling features per station up to target_date
    cutoff_long = target_date - pd.Timedelta(days=LAG_LONG)
    wx_window   = wx[(wx["date"] >= cutoff_long) & (wx["date"] <= target_date)]

    sta_features: dict = {}
    for sta, grp in wx_window.groupby("station"):
        grp = grp.set_index("date").sort_index()
        # Use only rows up to and including target_date
        grp = grp.loc[:target_date]
        if grp.empty:
            continue
        row: dict = {"station": sta}
        for v in ["awnd", "prcp", "tmax", "tmin"]:
            s = grp[v]
            row[f"{v}_roll{LAG_SHORT}_mean"] = s.tail(LAG_SHORT).mean()
            row[f"{v}_roll{LAG_LONG}_mean"]  = s.tail(LAG_LONG).mean()
            if v == "prcp":
                row[f"prcp_roll{LAG_LONG}_sum"] = s.tail(LAG_LONG).sum()
            if v == "tmax":
                row[f"tmax_roll{LAG_SHORT}_max"] = s.tail(LAG_SHORT).max()
            # Same-day value
            row[v] = s.iloc[-1] if len(s) > 0 else np.nan
        sta_features[sta] = row

    # Map cells to station features
    feat_rows = []
    for cid in cell_ids:
        sta = cell_to_station.get(cid)
        base = sta_features.get(sta, {})
        base["cell_id"] = cid
        feat_rows.append(base)

    df = pd.DataFrame(feat_rows)

    # ── PDSI ─────────────────────────────────────────────────────────────
    pdsi_daily = expand_capdsi_to_daily(load_capdsi(PDSI_CSV))
    pdsi_val   = pdsi_daily.loc[pdsi_daily["date"] <= target_date, "pdsi"].iloc[-1]
    df["pdsi"] = pdsi_val

    # ── Temporal features ─────────────────────────────────────────────────
    doy = target_date.dayofyear
    df["doy_sin"]   = math.sin(2 * math.pi * doy / 365.25)
    df["doy_cos"]   = math.cos(2 * math.pi * doy / 365.25)
    df["month_sin"] = math.sin(2 * math.pi * target_date.month / 12)
    df["month_cos"] = math.cos(2 * math.pi * target_date.month / 12)
    df["year"]      = target_date.year

    # ── Fire history features ─────────────────────────────────────────────
    log.info("Computing fire history features…")
    fire_df = load_modis_csv(MODIS_CSV, h3_resolution=RESOLUTION, bbox=BBOX)
    fire_df = fire_df[fire_df["fire_date"] < target_date]  # strictly past

    fire_dates_by_cell: dict = {}
    for _, r in fire_df.iterrows():
        fire_dates_by_cell.setdefault(r["cell_id"], []).append(r["fire_date"])
    for cid in fire_dates_by_cell:
        fire_dates_by_cell[cid].sort()

    neighbor_fire_dates: dict = {}
    for cid, dates in fire_dates_by_cell.items():
        for nb in h3.k_ring(cid, 1):
            neighbor_fire_dates.setdefault(nb, []).extend(dates)
    for nb in neighbor_fire_dates:
        neighbor_fire_dates[nb].sort()

    n = len(df)
    days_since = np.full(n, 9999.0)
    fire_1yr   = np.zeros(n, dtype=np.int32)
    nbr_7d     = np.zeros(n, dtype=np.int32)
    nbr_30d    = np.zeros(n, dtype=np.int32)

    cut_1yr = target_date - pd.Timedelta(days=365)
    cut_7d  = target_date - pd.Timedelta(days=7)
    cut_30d = target_date - pd.Timedelta(days=30)

    for i, cid in enumerate(df["cell_id"].values):
        if cid in fire_dates_by_cell:
            dl = fire_dates_by_cell[cid]
            idx = bisect.bisect_left(dl, target_date)
            if idx > 0:
                days_since[i] = (target_date - dl[idx-1]).days
                fire_1yr[i]   = idx - bisect.bisect_left(dl, cut_1yr)
        if cid in neighbor_fire_dates:
            nl  = neighbor_fire_dates[cid]
            end = bisect.bisect_left(nl, target_date)
            nbr_7d[i]  = end - bisect.bisect_left(nl, cut_7d)
            nbr_30d[i] = end - bisect.bisect_left(nl, cut_30d)

    df["days_since_last_fire"] = days_since
    df["cell_fire_count_1yr"]  = fire_1yr
    df["neighbor_fire_7d"]     = nbr_7d
    df["neighbor_fire_30d"]    = nbr_30d

    # ── Spatial features ─────────────────────────────────────────────────
    COAST = [(32.5,-117.2),(33.7,-118.3),(35.0,-120.6),(36.6,-121.9),
             (37.8,-122.5),(38.3,-123.0),(39.5,-123.8),(40.8,-124.2),(41.5,-124.1)]

    def haversine(lat1, lon1, lat2, lon2):
        R = 6371.0
        dl = math.radians(lat2-lat1); dlo = math.radians(lon2-lon1)
        a = math.sin(dl/2)**2 + math.cos(math.radians(lat1))*math.cos(math.radians(lat2))*math.sin(dlo/2)**2
        return R*2*math.asin(math.sqrt(a))

    cell_lat = np.array([grid_df.loc[grid_df["cell_id"]==c, "lat"].values[0]
                          if c in grid_df["cell_id"].values else np.nan for c in df["cell_id"]])
    cell_lon = np.array([grid_df.loc[grid_df["cell_id"]==c, "lon"].values[0]
                          if c in grid_df["cell_id"].values else np.nan for c in df["cell_id"]])

    lat_map = dict(zip(grid_df["cell_id"], grid_df["lat"]))
    lon_map = dict(zip(grid_df["cell_id"], grid_df["lon"]))
    cell_lat = np.array([lat_map.get(c, np.nan) for c in df["cell_id"]])
    cell_lon = np.array([lon_map.get(c, np.nan) for c in df["cell_id"]])
    dist_coast = np.array([
        min(haversine(la, lo, *cp) for cp in COAST) if not (np.isnan(la) or np.isnan(lo)) else np.nan
        for la, lo in zip(cell_lat, cell_lon)
    ])

    df["cell_lat"]         = cell_lat
    df["cell_lon"]         = cell_lon
    df["dist_to_coast_km"] = dist_coast
    df["is_inland"]        = (dist_coast > 80).astype(int)

    # ── Derived weather ───────────────────────────────────────────────────
    df["temp_range"]  = df["tmax"] - df["tmin"]
    df["vpd_proxy"]   = np.where(df["tmin"] < df["tmax"]*0.3, df["tmax"]*0.6, df["tmax"]-df["tmin"])
    hot_col  = f"tmax_roll{LAG_SHORT}_mean"
    prcp_col = f"prcp_roll{LAG_LONG}_sum"
    wind_col = f"awnd_roll{LAG_SHORT}_mean"
    df["hot_dry_windy"]       = ((df[hot_col]  > df[hot_col].quantile(0.75)).astype(int) *
                                  (df[prcp_col] < 5.0).astype(int) *
                                  (df[wind_col] > df[wind_col].quantile(0.60)).astype(int))
    df["consecutive_dry_days"] = (df[prcp_col] < 5.0).astype(int) * 30

    # ── Vegetation features (LANDFIRE) ────────────────────────────────────
    from src.data.landfire import build_vegetation_features as _build_veg
    veg_df = _build_veg(grid_df, tif_dir=None)
    df = df.merge(veg_df, on="cell_id", how="left")
    df["evt"]          = df["evt"].fillna(0).astype(int)
    df["fbfm40"]       = df["fbfm40"].fillna(0).astype(int)
    df["canopy_cover"] = df["canopy_cover"].fillna(0.0)

    log.info("Features built: %d cells × %d columns", len(df), len(df.columns))
    return df


# ══════════════════════════════════════════════════════════════════════════════
# Run inference
# ══════════════════════════════════════════════════════════════════════════════

def run_inference(feature_df) -> pd.DataFrame:
    log.info("Loading model and running inference…")
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)

    FEATURE_COLS = [
        'awnd_roll7_mean', 'awnd_roll30_mean', 'prcp_roll7_mean', 'prcp_roll30_mean',
        'prcp_roll30_sum', 'tmax_roll7_mean', 'tmax_roll30_mean', 'tmax_roll7_max',
        'tmin_roll7_mean', 'tmin_roll30_mean', 'awnd', 'prcp', 'tmax', 'tmin',
        'pdsi', 'doy_sin', 'doy_cos', 'month_sin', 'month_cos', 'year',
        'days_since_last_fire', 'cell_fire_count_1yr', 'neighbor_fire_7d',
        'neighbor_fire_30d', 'cell_lat', 'cell_lon', 'dist_to_coast_km', 'is_inland',
        'temp_range', 'vpd_proxy', 'hot_dry_windy', 'consecutive_dry_days',
        'evt', 'fbfm40', 'canopy_cover',
    ]
    X = feature_df[FEATURE_COLS].fillna(0)
    proba = model.predict_proba(X)[:, 1]

    result = feature_df[["cell_id"]].copy()
    result["probability"] = proba
    log.info("Inference done. Probability range: [%.4f, %.4f], mean=%.4f",
             proba.min(), proba.max(), proba.mean())
    return result


# ══════════════════════════════════════════════════════════════════════════════
# Build Folium map
# ══════════════════════════════════════════════════════════════════════════════

def build_map(pred_df, target_date: pd.Timestamp) -> folium.Map:
    log.info("Building Folium map…")

    # Centre on California
    ca_center = [37.5, -119.5]
    m = folium.Map(
        location=ca_center,
        zoom_start=6,
        tiles="CartoDB dark_matter",
        control_scale=True,
    )

    # Colormap: white → yellow → orange → red
    colormap = cm.LinearColormap(
        colors=["#ffffcc", "#fed976", "#fd8d3c", "#e31a1c", "#800026"],
        vmin=0.0,
        vmax=1.0,
        caption="7-Day Wildfire Ignition Probability",
    )
    colormap.add_to(m)

    # Load actual fires on target date for ground truth overlay
    fire_df = pd.read_csv(MODIS_CSV, parse_dates=["acq_date"])
    actual_fires = fire_df[
        (fire_df["acq_date"] == target_date) &
        (fire_df["latitude"].between(BBOX["lat_min"], BBOX["lat_max"])) &
        (fire_df["longitude"].between(BBOX["lon_min"], BBOX["lon_max"])) &
        (fire_df["type"] == 0) &
        (fire_df["confidence"] >= 30)
    ]
    log.info("Actual fires on %s: %d detections", target_date.date(), len(actual_fires))

    # ── H3 hexagons ───────────────────────────────────────────────────────
    # Only draw cells with probability > threshold to keep map fast
    DRAW_THRESHOLD = 0.05
    high_risk = pred_df[pred_df["probability"] >= DRAW_THRESHOLD]
    low_risk  = pred_df[pred_df["probability"] < DRAW_THRESHOLD]

    log.info("Drawing %d high-risk cells (p≥%.2f) and %d low-risk cells…",
             len(high_risk), DRAW_THRESHOLD, len(low_risk))

    # Draw low-risk cells as a thin outline layer (less detail)
    for _, row in low_risk.iterrows():
        boundary = h3.h3_to_geo_boundary(row["cell_id"], geo_json=True)
        coords   = [[lat, lon] for lon, lat in boundary]  # folium expects [lat,lon]
        p = row["probability"]
        folium.Polygon(
            locations=coords,
            color=colormap(p),
            weight=0,
            fill=True,
            fill_color=colormap(p),
            fill_opacity=max(0.05, p * 0.6),
        ).add_to(m)

    # Draw high-risk cells with tooltips
    for _, row in high_risk.iterrows():
        boundary = h3.h3_to_geo_boundary(row["cell_id"], geo_json=True)
        coords   = [[lat, lon] for lon, lat in boundary]
        p = row["probability"]
        folium.Polygon(
            locations=coords,
            color=colormap(p),
            weight=0.3,
            fill=True,
            fill_color=colormap(p),
            fill_opacity=min(0.9, 0.2 + p * 0.75),
            tooltip=folium.Tooltip(
                f"<b>P(fire)</b>: {p:.1%}<br>"
                f"<b>Cell</b>: {row['cell_id'][:10]}…",
                sticky=False,
            ),
        ).add_to(m)

    # ── Actual fire detections overlay ────────────────────────────────────
    fire_group = folium.FeatureGroup(name="Actual MODIS Fires", show=True)
    for _, fire in actual_fires.iterrows():
        folium.CircleMarker(
            location=[fire["latitude"], fire["longitude"]],
            radius=4,
            color="#00ffff",
            fill=True,
            fill_color="#00ffff",
            fill_opacity=0.8,
            weight=1,
            tooltip=folium.Tooltip(
                f"<b>MODIS Fire Detection</b><br>"
                f"Confidence: {fire['confidence']:.0f}%<br>"
                f"FRP: {fire['frp']:.1f} MW",
                sticky=False,
            ),
        ).add_to(fire_group)
    fire_group.add_to(m)

    # ── Title & legend ────────────────────────────────────────────────────
    title_html = f"""
    <div style="position:fixed; top:12px; left:60px; z-index:1000;
                background:rgba(0,0,0,0.7); padding:10px 16px; border-radius:8px;
                color:white; font-family:Arial; font-size:14px;">
        <b>🔥 California Wildfire Risk</b><br>
        <span style="font-size:12px; color:#aaa;">
            7-day ignition probability · {target_date.strftime('%B %d, %Y')}<br>
            LightGBM model · MODIS fire detections shown in cyan
        </span>
    </div>
    """
    m.get_root().html.add_child(folium.Element(title_html))

    folium.LayerControl().add_to(m)

    return m


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import warnings; warnings.filterwarnings("ignore")

    from src.preprocessing.align import build_california_grid

    log.info("━━━ Wildfire Risk Map — %s ━━━", TARGET_DATE.date())

    grid_df     = build_california_grid(BBOX, RESOLUTION)
    feature_df  = build_features_for_date(TARGET_DATE, grid_df)
    pred_df     = run_inference(feature_df)

    # Print top 10 highest-risk cells
    top10 = pred_df.nlargest(10, "probability")
    log.info("Top 10 highest-risk cells:")
    for _, r in top10.iterrows():
        lat, lon = h3.h3_to_geo(r["cell_id"])
        log.info("  %s  p=%.3f  (%.3f°N, %.3f°W)", r["cell_id"], r["probability"], lat, abs(lon))

    m = build_map(pred_df, TARGET_DATE)

    out_path = OUT_DIR / f"wildfire_risk_{TARGET_DATE.strftime('%Y%m%d')}.html"
    m.save(str(out_path))
    log.info("Map saved → %s", out_path)
    log.info("Open in browser: open '%s'", out_path)
