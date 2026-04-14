#!/usr/bin/env python3
"""
scripts/eval_convlstm.py
========================
Load saved ConvLSTM weights and evaluate on the 2023 test set.
Prints PR-AUC / ROC-AUC and comparison against LightGBM.
"""

from __future__ import annotations
import logging, sys, warnings
warnings.filterwarnings("ignore")
from pathlib import Path

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("eval_convlstm")

# ── reuse constants and pipeline functions from train_convlstm ────────────────
from scripts.train_convlstm import (
    build_pixel_grid, build_daily_rasters, build_label_rasters,
    WildfireDataset, evaluate_convlstm,
    TRAIN_END, VAL_START, VAL_END, TEST_START, TEST_END,
    SEQ_LEN, N_CHANNELS, GRID_H, GRID_W, DEVICE, RANDOM_SEED, OUT_DIR,
)
from src.models.convlstm_model import WildfireConvLSTM

torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

log.info("━━━ ConvLSTM Evaluation ━━━")

_, _, cell_to_pixel, _ = build_pixel_grid()
date_range, feature_array = build_daily_rasters(cell_to_pixel)
label_array = build_label_rasters(cell_to_pixel, date_range)

train_mask = date_range <= pd.Timestamp(TRAIN_END)
mean = feature_array[train_mask].mean(axis=(0, 2, 3))
std  = feature_array[train_mask].std(axis=(0, 2, 3))

all_idx  = np.arange(SEQ_LEN, len(date_range))
test_idx = all_idx[(date_range[all_idx] >= pd.Timestamp(TEST_START)) &
                   (date_range[all_idx] <= pd.Timestamp(TEST_END))]
log.info("Test dates: %d", len(test_idx))

test_ds = WildfireDataset(test_idx, feature_array, label_array, mean, std)

model = WildfireConvLSTM(input_channels=N_CHANNELS, hidden_channels=[64, 32]).to(DEVICE)
model_path = OUT_DIR / "models" / "convlstm_model.pt"
model.load_state_dict(torch.load(model_path, map_location=DEVICE))
log.info("Loaded weights from %s", model_path)

pr_auc, roc_auc = evaluate_convlstm(model, test_ds)

log.info("")
log.info("══════════════════════════════════════════════")
log.info("  Model Comparison (2023 Test Set)")
log.info("══════════════════════════════════════════════")
log.info("  LightGBM    PR-AUC=0.4674  ROC-AUC=0.9158")
log.info("  LogReg      PR-AUC=0.4812  ROC-AUC=0.9226")
log.info("  ConvLSTM    PR-AUC=%.4f  ROC-AUC=%.4f", pr_auc, roc_auc)
log.info("══════════════════════════════════════════════")
