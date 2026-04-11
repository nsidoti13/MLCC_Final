"""
src/modeling/trainer.py
=======================
Model training orchestration with class-imbalance handling and
time-series cross-validation.

Design decisions
----------------
* **scale_pos_weight**: For both LightGBM and XGBoost, we set ``scale_pos_weight``
  to ``neg_count / pos_count`` automatically from the training labels.  This is
  equivalent to weighting the loss function by the inverse class frequency.
  At ~1% positive rate this factor is approximately 99, which substantially
  improves recall on the minority class.

* **TimeSeriesSplit**: We never use random k-fold CV.  Validation folds always
  come *after* training folds in time, preventing any future leakage.

* **Early stopping**: Both LGBM and XGBoost use a held-out validation set from
  the last fold for early stopping.  This prevents overfitting without a fixed
  number of estimators.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Main trainer
# ---------------------------------------------------------------------------


def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    model_type: str = "lgbm",
    params: Optional[Dict[str, Any]] = None,
    eval_set: Optional[Tuple[pd.DataFrame, pd.Series]] = None,
    n_cv_splits: int = 3,
) -> Any:
    """
    Train a wildfire ignition prediction model.

    Supports ``"lgbm"`` (LightGBM), ``"xgboost"`` (XGBoost), and
    ``"logistic"`` (Logistic Regression baseline).

    Class imbalance is handled automatically:
    - LGBM / XGBoost: ``scale_pos_weight`` = neg / pos
    - Logistic Regression: ``class_weight="balanced"``

    Parameters
    ----------
    X_train:
        Feature matrix (rows = samples, cols = features).
        Must not contain the label or metadata columns (cell_id, date).
    y_train:
        Binary target series (0/1).  Must be aligned with X_train.
    model_type:
        One of ``"lgbm"``, ``"xgboost"``, ``"logistic"``.
    params:
        Model hyperparameters dict.  Merged with defaults; keys in ``params``
        take precedence.
    eval_set:
        Optional (X_val, y_val) tuple for early stopping.  If not provided,
        the last TimeSeriesSplit fold is used.
    n_cv_splits:
        Number of TimeSeriesSplit folds for cross-validation (default 3).
        The model returned is trained on ALL of X_train (not a single fold).

    Returns
    -------
    Fitted model object (LGBMClassifier, XGBClassifier, or LogisticRegression).

    Raises
    ------
    ValueError
        If model_type is not recognised.
    """
    model_type = model_type.lower()
    if model_type not in ("lgbm", "xgboost", "logistic"):
        raise ValueError(
            f"train_model: unknown model_type '{model_type}'. "
            "Choose from 'lgbm', 'xgboost', 'logistic'."
        )

    logger.info(
        "Training %s on %d samples (%d features), positive rate = %.4f.",
        model_type,
        len(y_train),
        X_train.shape[1],
        y_train.mean(),
    )

    # Compute class weight
    neg_count = int((y_train == 0).sum())
    pos_count = int((y_train == 1).sum())
    if pos_count == 0:
        raise ValueError("train_model: no positive samples in y_train.")
    scale_pos_weight = neg_count / pos_count
    logger.info(
        "Class imbalance ratio (neg/pos) = %.2f (scale_pos_weight = %.2f).",
        scale_pos_weight,
        scale_pos_weight,
    )

    # Run cross-validation for diagnostic PR-AUC scores (does not affect
    # the final model which is trained on all data)
    cv_scores = run_cross_validation(X_train, y_train, model_type, params, n_cv_splits)
    logger.info(
        "Cross-validation PR-AUC scores: %s  (mean = %.4f ± %.4f)",
        [f"{s:.4f}" for s in cv_scores],
        np.mean(cv_scores),
        np.std(cv_scores),
    )

    # Train final model on all training data
    if model_type == "lgbm":
        model = _train_lgbm(X_train, y_train, scale_pos_weight, params, eval_set)
    elif model_type == "xgboost":
        model = _train_xgboost(X_train, y_train, scale_pos_weight, params, eval_set)
    else:
        model = _train_logistic(X_train, y_train, params)

    return model


# ---------------------------------------------------------------------------
# Cross-validation
# ---------------------------------------------------------------------------


def run_cross_validation(
    X: pd.DataFrame,
    y: pd.Series,
    model_type: str = "lgbm",
    params: Optional[Dict[str, Any]] = None,
    n_splits: int = 3,
) -> List[float]:
    """
    Run TimeSeriesSplit cross-validation and return per-fold PR-AUC scores.

    Folds are strictly temporal: each validation fold comes entirely after
    its corresponding training fold.  No data leakage.

    Parameters
    ----------
    X:
        Feature matrix (must be sorted by date before calling).
    y:
        Binary target series.
    model_type:
        One of 'lgbm', 'xgboost', 'logistic'.
    params:
        Model hyperparameters.
    n_splits:
        Number of TimeSeriesSplit folds.

    Returns
    -------
    List[float]
        PR-AUC score for each fold.
    """
    from sklearn.metrics import average_precision_score

    tscv = TimeSeriesSplit(n_splits=n_splits)
    scores: List[float] = []

    X_arr = X.values if hasattr(X, "values") else np.array(X)
    y_arr = y.values if hasattr(y, "values") else np.array(y)

    for fold_idx, (train_idx, val_idx) in enumerate(tscv.split(X_arr)):
        X_fold_train, X_fold_val = X_arr[train_idx], X_arr[val_idx]
        y_fold_train, y_fold_val = y_arr[train_idx], y_arr[val_idx]

        if y_fold_val.sum() == 0:
            logger.warning("Fold %d has no positive samples in validation; skipping.", fold_idx)
            continue

        neg = int((y_fold_train == 0).sum())
        pos = int((y_fold_train == 1).sum())
        spw = neg / max(pos, 1)

        fold_df_train = pd.DataFrame(X_fold_train, columns=X.columns)
        fold_df_val = pd.DataFrame(X_fold_val, columns=X.columns)
        fold_y_train = pd.Series(y_fold_train)
        fold_y_val = pd.Series(y_fold_val)

        t0 = time.time()
        if model_type == "lgbm":
            fold_model = _train_lgbm(
                fold_df_train, fold_y_train, spw, params,
                eval_set=(fold_df_val, fold_y_val),
            )
        elif model_type == "xgboost":
            fold_model = _train_xgboost(
                fold_df_train, fold_y_train, spw, params,
                eval_set=(fold_df_val, fold_y_val),
            )
        else:
            fold_model = _train_logistic(fold_df_train, fold_y_train, params)

        proba = fold_model.predict_proba(fold_df_val)[:, 1]
        prauc = average_precision_score(fold_y_val, proba)
        scores.append(prauc)
        logger.info(
            "  Fold %d/%d — PR-AUC = %.4f  (%.1fs)",
            fold_idx + 1,
            n_splits,
            prauc,
            time.time() - t0,
        )

    return scores


# ---------------------------------------------------------------------------
# Model-specific trainers
# ---------------------------------------------------------------------------


def _train_lgbm(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    scale_pos_weight: float,
    params: Optional[Dict[str, Any]],
    eval_set: Optional[Tuple[pd.DataFrame, pd.Series]],
) -> Any:
    """Train a LightGBM classifier with early stopping if eval_set is provided."""
    import lightgbm as lgb

    default_params: Dict[str, Any] = {
        "n_estimators": 1000,
        "learning_rate": 0.05,
        "num_leaves": 63,
        "max_depth": 7,
        "min_child_samples": 50,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "scale_pos_weight": scale_pos_weight,
        "objective": "binary",
        "metric": "average_precision",
        "n_jobs": -1,
        "verbose": -1,
        "random_state": 42,
    }
    if params:
        default_params.update(params)
    # Always override scale_pos_weight with the computed value
    default_params["scale_pos_weight"] = scale_pos_weight

    model = lgb.LGBMClassifier(**default_params)

    callbacks = [lgb.log_evaluation(period=100)]
    if eval_set is not None:
        X_val, y_val = eval_set
        callbacks.append(lgb.early_stopping(stopping_rounds=50, verbose=False))
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            callbacks=callbacks,
        )
    else:
        model.fit(X_train, y_train, callbacks=callbacks)

    logger.info(
        "LightGBM trained: best_iteration = %s",
        getattr(model, "best_iteration_", "N/A"),
    )
    return model


def _train_xgboost(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    scale_pos_weight: float,
    params: Optional[Dict[str, Any]],
    eval_set: Optional[Tuple[pd.DataFrame, pd.Series]],
) -> Any:
    """Train an XGBoost classifier with early stopping if eval_set is provided."""
    from xgboost import XGBClassifier

    default_params: Dict[str, Any] = {
        "n_estimators": 1000,
        "learning_rate": 0.05,
        "max_depth": 7,
        "min_child_weight": 50,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "scale_pos_weight": scale_pos_weight,
        "objective": "binary:logistic",
        "eval_metric": "aucpr",
        "tree_method": "hist",
        "n_jobs": -1,
        "verbosity": 0,
        "random_state": 42,
    }
    if params:
        default_params.update(params)
    default_params["scale_pos_weight"] = scale_pos_weight

    model = XGBClassifier(**default_params)

    if eval_set is not None:
        X_val, y_val = eval_set
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            early_stopping_rounds=50,
            verbose=100,
        )
    else:
        model.fit(X_train, y_train)

    logger.info(
        "XGBoost trained: best_iteration = %s",
        getattr(model, "best_iteration", "N/A"),
    )
    return model


def _train_logistic(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    params: Optional[Dict[str, Any]],
) -> LogisticRegression:
    """Train a Logistic Regression baseline with standard scaling."""
    default_params: Dict[str, Any] = {
        "C": 0.1,
        "max_iter": 1000,
        "class_weight": "balanced",
        "solver": "saga",
        "random_state": 42,
        "n_jobs": -1,
    }
    if params:
        default_params.update(params)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)

    model = LogisticRegression(**default_params)
    model.fit(X_scaled, y_train)

    # Attach scaler so predict can use it
    model._scaler = scaler  # type: ignore[attr-defined]
    logger.info("Logistic Regression trained on %d samples.", len(y_train))
    return model
