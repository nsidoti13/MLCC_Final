"""
src/evaluation/metrics.py
=========================
Evaluation functions for the wildfire ignition prediction pipeline.

Primary metric: Precision-Recall AUC (PR-AUC / Average Precision)
  - Preferred over ROC-AUC for severely imbalanced datasets because it is
    sensitive to true positives relative to the positive class, not the
    (much larger) negative class.
  - A random classifier achieves PR-AUC ≈ positive rate (~0.01), so any
    useful model must substantially exceed this baseline.

Secondary metric: ROC-AUC
  - Reported for comparison with external benchmarks.

Operational metric: Recall @ k
  - Answers the operational question: "If we issue alerts for the top-k
    highest-risk cells, what fraction of actual ignitions do we catch?"
  - k=500 corresponds to roughly 1.5% of California burnable cells.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for servers
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Main evaluation function
# ---------------------------------------------------------------------------


def evaluate(
    y_true: np.ndarray,
    y_score: np.ndarray,
    top_k: int = 500,
    output_dir: Optional[str] = None,
    model_name: str = "model",
    save_plots: bool = True,
) -> Dict[str, float]:
    """
    Compute all evaluation metrics and optionally save plots.

    Parameters
    ----------
    y_true:
        Binary ground-truth labels (0/1).  Shape (n_samples,).
    y_score:
        Predicted probabilities for the positive class.  Shape (n_samples,).
    top_k:
        Number of highest-scoring cells to consider for recall@k.
    output_dir:
        Directory for saving plots and metrics JSON.  Defaults to
        ``outputs/reports/``.
    model_name:
        Model identifier used in filenames and plot titles.
    save_plots:
        Whether to write PNG files for precision-recall and ROC curves.

    Returns
    -------
    dict
        Keys: pr_auc, roc_auc, recall_at_k, precision_at_k, f1_at_k,
              positive_rate, n_samples, k.
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)

    if y_true.ndim != 1 or y_score.ndim != 1:
        raise ValueError("evaluate: y_true and y_score must be 1-D arrays.")
    if len(y_true) != len(y_score):
        raise ValueError("evaluate: y_true and y_score must have the same length.")

    n_pos = int(y_true.sum())
    if n_pos == 0:
        raise ValueError("evaluate: no positive samples in y_true.")

    positive_rate = n_pos / len(y_true)

    # ------------------------------------------------------------------
    # PR-AUC (Average Precision)
    # ------------------------------------------------------------------
    pr_auc = average_precision_score(y_true, y_score)

    # ------------------------------------------------------------------
    # ROC-AUC
    # ------------------------------------------------------------------
    roc_auc = roc_auc_score(y_true, y_score)

    # ------------------------------------------------------------------
    # Recall @ k  (operational metric)
    # ------------------------------------------------------------------
    recall_k, precision_k, f1_k = recall_at_k(y_true, y_score, k=top_k)

    metrics = {
        "pr_auc": float(pr_auc),
        "roc_auc": float(roc_auc),
        "recall_at_k": float(recall_k),
        "precision_at_k": float(precision_k),
        "f1_at_k": float(f1_k),
        "positive_rate": float(positive_rate),
        "n_samples": int(len(y_true)),
        "k": int(top_k),
    }

    logger.info(
        "[%s] PR-AUC=%.4f | ROC-AUC=%.4f | Recall@%d=%.4f | Prec@%d=%.4f",
        model_name,
        pr_auc,
        roc_auc,
        top_k,
        recall_k,
        top_k,
        precision_k,
    )

    if output_dir is None:
        output_dir = "outputs/reports"
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save metrics JSON
    metrics_file = output_path / f"{model_name}_metrics.json"
    with open(metrics_file, "w") as fh:
        json.dump(metrics, fh, indent=2)
    logger.info("Metrics saved to %s", metrics_file)

    if save_plots:
        _plot_precision_recall_curve(y_true, y_score, pr_auc, model_name, output_path)
        _plot_roc_curve(y_true, y_score, roc_auc, model_name, output_path)
        _plot_score_distribution(y_true, y_score, model_name, output_path)

    return metrics


# ---------------------------------------------------------------------------
# Recall @ k
# ---------------------------------------------------------------------------


def recall_at_k(
    y_true: np.ndarray,
    y_score: np.ndarray,
    k: int = 500,
) -> Tuple[float, float, float]:
    """
    Compute recall, precision, and F1 score in the top-k highest-risk cells.

    The operational interpretation: "If we issue wildfire risk alerts for the
    k cells with the highest predicted probability, what fraction of true
    ignitions do we capture?"

    Parameters
    ----------
    y_true:
        Binary ground-truth labels.
    y_score:
        Predicted probabilities.
    k:
        Number of top-scoring samples to evaluate.

    Returns
    -------
    Tuple[float, float, float]
        (recall@k, precision@k, F1@k)
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)

    k = min(k, len(y_true))
    top_k_idx = np.argsort(y_score)[::-1][:k]

    y_top = y_true[top_k_idx]
    tp = int(y_top.sum())
    n_pos = int(y_true.sum())

    recall = tp / n_pos if n_pos > 0 else 0.0
    precision = tp / k if k > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    return recall, precision, f1


# ---------------------------------------------------------------------------
# Precision-Recall curve at multiple k values
# ---------------------------------------------------------------------------


def recall_precision_at_multiple_k(
    y_true: np.ndarray,
    y_score: np.ndarray,
    k_values: Optional[list] = None,
) -> pd.DataFrame:
    """
    Compute recall and precision at multiple k values.

    Useful for choosing the operational alert threshold.

    Parameters
    ----------
    y_true:
        Binary ground-truth labels.
    y_score:
        Predicted probabilities.
    k_values:
        List of k values to evaluate.  Defaults to [100, 250, 500, 1000, 2500, 5000].

    Returns
    -------
    pd.DataFrame
        Columns: k, recall, precision, f1.
    """
    if k_values is None:
        k_values = [100, 250, 500, 1000, 2500, 5000]

    rows = []
    for k in k_values:
        rec, prec, f1 = recall_at_k(y_true, y_score, k)
        rows.append({"k": k, "recall": rec, "precision": prec, "f1": f1})

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------


def _plot_precision_recall_curve(
    y_true: np.ndarray,
    y_score: np.ndarray,
    pr_auc: float,
    model_name: str,
    output_path: Path,
) -> None:
    """Plot and save a precision-recall curve."""
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    baseline = y_true.mean()

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(recall, precision, lw=2, label=f"{model_name} (PR-AUC = {pr_auc:.4f})")
    ax.axhline(baseline, color="gray", linestyle="--", lw=1, label=f"Baseline (pos rate = {baseline:.4f})")
    ax.set_xlabel("Recall", fontsize=13)
    ax.set_ylabel("Precision", fontsize=13)
    ax.set_title("Precision-Recall Curve — Wildfire Ignition", fontsize=14)
    ax.legend(fontsize=11)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    out_file = output_path / f"{model_name}_pr_curve.png"
    fig.savefig(out_file, dpi=150)
    plt.close(fig)
    logger.info("PR curve saved to %s", out_file)


def _plot_roc_curve(
    y_true: np.ndarray,
    y_score: np.ndarray,
    roc_auc: float,
    model_name: str,
    output_path: Path,
) -> None:
    """Plot and save a ROC curve."""
    fpr, tpr, _ = roc_curve(y_true, y_score)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, lw=2, label=f"{model_name} (ROC-AUC = {roc_auc:.4f})")
    ax.plot([0, 1], [0, 1], color="gray", linestyle="--", lw=1, label="Random")
    ax.set_xlabel("False Positive Rate", fontsize=13)
    ax.set_ylabel("True Positive Rate", fontsize=13)
    ax.set_title("ROC Curve — Wildfire Ignition", fontsize=14)
    ax.legend(fontsize=11)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    out_file = output_path / f"{model_name}_roc_curve.png"
    fig.savefig(out_file, dpi=150)
    plt.close(fig)
    logger.info("ROC curve saved to %s", out_file)


def _plot_score_distribution(
    y_true: np.ndarray,
    y_score: np.ndarray,
    model_name: str,
    output_path: Path,
) -> None:
    """Plot predicted score distribution split by class."""
    fig, ax = plt.subplots(figsize=(8, 5))

    scores_pos = y_score[y_true == 1]
    scores_neg = y_score[y_true == 0]

    # Sample negatives to avoid over-plotting (cap at 10x positives)
    max_neg = min(len(scores_neg), max(10 * len(scores_pos), 10000))
    rng = np.random.default_rng(seed=0)
    idx = rng.choice(len(scores_neg), size=max_neg, replace=False)
    scores_neg_sample = scores_neg[idx]

    bins = np.linspace(0, 1, 50)
    ax.hist(scores_neg_sample, bins=bins, alpha=0.6, label="No ignition (sample)", density=True)
    ax.hist(scores_pos, bins=bins, alpha=0.8, label="Ignition", density=True, color="tomato")
    ax.set_xlabel("Predicted Probability", fontsize=13)
    ax.set_ylabel("Density", fontsize=13)
    ax.set_title(f"Score Distribution — {model_name}", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    out_file = output_path / f"{model_name}_score_dist.png"
    fig.savefig(out_file, dpi=150)
    plt.close(fig)
    logger.info("Score distribution saved to %s", out_file)
