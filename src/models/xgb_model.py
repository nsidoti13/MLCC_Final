"""
src/models/xgb_model.py
=======================
Thin wrapper around XGBoost for the wildfire ignition prediction pipeline.

Mirrors the LGBMModel API so that the two model types can be swapped
transparently in the pipeline, evaluation, and inference code.
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class XGBModel:
    """
    XGBoost binary classifier wrapper for wildfire ignition prediction.

    Handles class-imbalance weighting via ``scale_pos_weight``, feature name
    tracking, and serialisation to disk.

    Parameters
    ----------
    params:
        XGBoost hyperparameter dict.  Merged with sensible defaults.

    Example
    -------
    >>> model = XGBModel(params={"n_estimators": 500, "max_depth": 6})
    >>> model.fit(X_train, y_train, eval_set=(X_val, y_val))
    >>> proba = model.predict_proba(X_test)
    >>> model.save("outputs/models/xgb_model.pkl")
    """

    DEFAULT_PARAMS: Dict[str, Any] = {
        "n_estimators": 1000,
        "learning_rate": 0.05,
        "max_depth": 7,
        "min_child_weight": 50,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "objective": "binary:logistic",
        "eval_metric": "aucpr",
        "tree_method": "hist",
        "n_jobs": -1,
        "verbosity": 0,
        "random_state": 42,
    }

    def __init__(self, params: Optional[Dict[str, Any]] = None) -> None:
        self.params: Dict[str, Any] = {**self.DEFAULT_PARAMS, **(params or {})}
        self._model: Any = None
        self.feature_names_: Optional[List[str]] = None
        self.scale_pos_weight_: Optional[float] = None

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        eval_set: Optional[Tuple[pd.DataFrame, pd.Series]] = None,
        early_stopping_rounds: int = 50,
    ) -> "XGBModel":
        """
        Fit the XGBoost model.

        Parameters
        ----------
        X:
            Training feature matrix.
        y:
            Binary target series.
        eval_set:
            Optional (X_val, y_val) for early stopping.
        early_stopping_rounds:
            Patience for early stopping (ignored if eval_set is None).

        Returns
        -------
        XGBModel
            self (for method chaining).
        """
        from xgboost import XGBClassifier

        self.feature_names_ = list(X.columns)

        neg = int((y == 0).sum())
        pos = int((y == 1).sum())
        if pos == 0:
            raise ValueError("XGBModel.fit: no positive samples in y.")
        self.scale_pos_weight_ = neg / pos

        if "scale_pos_weight" not in self.params:
            self.params["scale_pos_weight"] = self.scale_pos_weight_

        self._model = XGBClassifier(**self.params)

        if eval_set is not None:
            X_val, y_val = eval_set
            self._model.fit(
                X,
                y,
                eval_set=[(X_val, y_val)],
                early_stopping_rounds=early_stopping_rounds,
                verbose=100,
            )
        else:
            self._model.fit(X, y)

        logger.info(
            "XGBModel fitted: n_features=%d, best_iteration=%s, pos_weight=%.2f",
            len(self.feature_names_),
            getattr(self._model, "best_iteration", "N/A"),
            self.scale_pos_weight_,
        )
        return self

    # ------------------------------------------------------------------
    # Predict
    # ------------------------------------------------------------------

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Return binary class predictions.

        Parameters
        ----------
        X:
            Feature matrix.

        Returns
        -------
        np.ndarray
            1-D integer array of predicted classes.
        """
        self._check_fitted()
        X = self._align_columns(X)
        return self._model.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Return predicted probability of ignition (positive class).

        Parameters
        ----------
        X:
            Feature matrix.

        Returns
        -------
        np.ndarray
            1-D float array of P(ignition) values in [0, 1].
        """
        self._check_fitted()
        X = self._align_columns(X)
        return self._model.predict_proba(X)[:, 1]

    # ------------------------------------------------------------------
    # Feature importance
    # ------------------------------------------------------------------

    def feature_importances(self, importance_type: str = "gain") -> pd.Series:
        """
        Return feature importances sorted in descending order.

        Parameters
        ----------
        importance_type:
            ``"gain"`` (default), ``"weight"``, or ``"cover"``.

        Returns
        -------
        pd.Series
            Index = feature names, values = importance scores.
        """
        self._check_fitted()
        imp_dict = self._model.get_booster().get_score(importance_type=importance_type)
        series = pd.Series(imp_dict).reindex(self.feature_names_).fillna(0)
        return series.sort_values(ascending=False)

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def save(self, path: Union[str, Path]) -> None:
        """
        Serialise the fitted model to disk.

        Parameters
        ----------
        path:
            Output file path (e.g. ``"outputs/models/xgb_model.pkl"``).
        """
        self._check_fitted()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as fh:
            pickle.dump(self, fh, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info("XGBModel saved to %s", path)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "XGBModel":
        """
        Load a serialised XGBModel from disk.

        Parameters
        ----------
        path:
            Path to a .pkl file written by :meth:`save`.

        Returns
        -------
        XGBModel
            Loaded model instance.

        Raises
        ------
        FileNotFoundError
            If the path does not exist.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"XGBModel.load: file not found: {path}")
        with open(path, "rb") as fh:
            model = pickle.load(fh)
        logger.info("XGBModel loaded from %s", path)
        return model

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _check_fitted(self) -> None:
        if self._model is None:
            raise RuntimeError(
                "XGBModel has not been fitted yet.  Call .fit() first."
            )

    def _align_columns(self, X: pd.DataFrame) -> pd.DataFrame:
        """Reorder / fill columns to match training feature set."""
        if self.feature_names_ is None:
            return X
        missing = set(self.feature_names_) - set(X.columns)
        if missing:
            logger.warning(
                "XGBModel: %d features missing at inference; filling with 0: %s",
                len(missing),
                sorted(missing),
            )
            X = X.copy()
            for col in missing:
                X[col] = 0.0
        return X[self.feature_names_]

    def __repr__(self) -> str:
        status = "fitted" if self._model is not None else "unfitted"
        n_feat = len(self.feature_names_) if self.feature_names_ else 0
        return (
            f"XGBModel(status={status}, n_features={n_feat}, "
            f"scale_pos_weight={self.scale_pos_weight_})"
        )
