#!/usr/bin/env python3
"""
scripts/plot_roc_comparison.py
==============================
Load saved test predictions and plot ROC curves for LightGBM vs
Logistic Regression on the same axes.

Requires outputs/reports/test_predictions.parquet (written by train_model.py).
"""

from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PREDS_PATH   = PROJECT_ROOT / "outputs/reports/test_predictions.parquet"
OUT_PATH     = PROJECT_ROOT / "outputs/reports/roc_comparison.png"

df = pd.read_parquet(PREDS_PATH)
y  = df["y_test"].values

models = {
    "LightGBM": df["lgb_proba"].values,
    "Logistic Regression": df["lr_proba"].values,
}

colors = {"LightGBM": "#e34a33", "Logistic Regression": "#3182bd"}

fig, ax = plt.subplots(figsize=(8, 6))

for name, scores in models.items():
    fpr, tpr, _ = roc_curve(y, scores)
    auc = roc_auc_score(y, scores)
    ax.plot(fpr, tpr, lw=2, color=colors[name], label=f"{name} (AUC = {auc:.4f})")

ax.plot([0, 1], [0, 1], color="gray", linestyle="--", lw=1, label="Random")
ax.set_xlabel("False Positive Rate", fontsize=13)
ax.set_ylabel("True Positive Rate", fontsize=13)
ax.set_title("ROC Curve Comparison — Wildfire Ignition", fontsize=14)
ax.legend(fontsize=11)
ax.set_xlim([0.0, 1.0])
ax.set_ylim([0.0, 1.05])
ax.grid(True, alpha=0.3)
fig.tight_layout()

fig.savefig(OUT_PATH, dpi=150)
print(f"Saved → {OUT_PATH}")
