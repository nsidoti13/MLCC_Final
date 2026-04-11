"""
src/models/lgbm_model.py
========================
Thin wrapper around LightGBM for the wildfire ignition prediction pipeline.

Provides a consistent fit / predict / predict_proba / save / load API that
is shared across model types, making it easy to swap models in the pipeline
without changing downstream evaluation or inference code.
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class LGBMModel:
    """
    LightGBM binary classifier wrapper for wildfire ignition prediction.

    Handles class-imbalance weighting, feature name tracking, and
    serialisation to disk.

    Parameters
    ----------
    params:
        LightGBM hyperparameter dict.  Merged with sensible defaults.
        ``scale_pos_weight`` is set automatically from training data when
        ``fit()`` is called unless explicitly provided here.

    Example
    -------
    >>> model = LGBMModel(params={"n_estimators": 500, "num_leaves": 31})
    >>> model.fit(X_train, y_train, eval_set=(X_val, y_val))
    >>> proba = model.predict_proba(X_test)
    >>> model.save("outputs/models/lgbm_model.pkl")
    """

    DEFAULT_PARAMS: Dict[str, Any] = {
        "n_estimators": 1000,
        "learning_rate": 0.05,
        "num_leaves": 63,
        "max_depth": 7,
        "min_child_samples": 50,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "objective": "binary",
        "metric": "average_precision",
        "n_jobs": -1,
        "verbose": -1,
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
    ) -> "LGBMModel":
        """
        Fit the LightGBM model.

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
        LGBMModel
            self (for method chaining).
        """
        import lightgbm as lgb

        self.feature_names_ = list(X.columns)

        # Compute scale_pos_weight from training data
        neg = int((y == 0).sum())
        pos = int((y == 1).sum())
        if pos == 0:
            raise ValueError("LGBMModel.fit: no positive samples in y.")
        self.scale_pos_weight_ = neg / pos

        # Use provided scale_pos_weight if explicitly set in params
        if "scale_pos_weight" not in self.params:
            self.params["scale_pos_weight"] = self.scale_pos_weight_

        self._model = lgb.LGBMClassifier(**self.params)

        callbacks = [lgb.log_evaluation(period=100)]
        if eval_set is not None:
            X_val, y_val = eval_set
            callbacks.append(
                lgb.early_stopping(stopping_rounds=early_stopping_rounds, verbose=False)
            )
            self._model.fit(
                X,
                y,
                eval_set=[(X_val, y_val)],
                callbacks=callbacks,
            )
        else:
            self._model.fit(X, y, callbacks=callbacks)

        logger.info(
            "LGBMModel fitted: n_features=%d, best_iteration=%s, pos_weight=%.2f",
            len(self.feature_names_),
            getattr(self._model, "best_iteration_", "N/A"),
            self.scale_pos_weight_,
        )
        return self

    # ------------------------------------------------------------------
    # Predict
    # ------------------------------------------------------------------

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Return binary class predictions (0 or 1).

        Parameters
        ----------
        X:
            Feature matrix with same columns as training data.

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
        Return predicted probabilities for the positive class.

        Parameters
        ----------
        X:
            Feature matrix with same columns as training data.

        Returns
        -------
        np.ndarray
            1-D float array of P(ignition) in [0, 1].
        """
        self._check_fitted()
        X = self._align_columns(X)
        return self._model.predict_proba(X)[:, 1]

    # ------------------------------------------------------------------
    # Feature importance
    # ------------------------------------------------------------------

    def feature_importances(self, importance_type: str = "gain") -> pd.Series:
        """
        Return feature importances as a pandas Series sorted descending.

        Parameters
        ----------
        importance_type:
            ``"gain"`` (default) or ``"split"``.

        Returns
        -------
        pd.Series
            Index = feature names, values = importance scores.
        """
        self._check_fitted()
        imp = self._model.booster_.feature_importance(importance_type=importance_type)
        return (
            pd.Series(imp, index=self.feature_names_)
            .sort_values(ascending=False)
        )

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def save(self, path: Union[str, Path]) -> None:
        """
        Serialise the fitted model to disk using pickle.

        Parameters
        ----------
        path:
            Output file path (e.g. ``"outputs/models/lgbm_model.pkl"``).
        """
        self._check_fitted()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as fh:
            pickle.dump(self, fh, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info("LGBMModel saved to %s", path)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "LGBMModel":
        """
        Load a serialised LGBMModel from disk.

        Parameters
        ----------
        path:
            Path to a .pkl file written by :meth:`save`.

        Returns
        -------
        LGBMModel
            Loaded model instance.

        Raises
        ------
        FileNotFoundError
            If the path does not exist.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"LGBMModel.load: file not found: {path}")
        with open(path, "rb") as fh:
            model = pickle.load(fh)
        logger.info("LGBMModel loaded from %s", path)
        return model

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _check_fitted(self) -> None:
        if self._model is None:
            raise RuntimeError(
                "LGBMModel has not been fitted yet.  Call .fit() first."
            )

    def _align_columns(self, X: pd.DataFrame) -> pd.DataFrame:
        """Reorder / fill columns to match training feature set."""
        if self.feature_names_ is None:
            return X
        missing = set(self.feature_names_) - set(X.columns)
        if missing:
            logger.warning(
                "LGBMModel: %d features missing at inference; filling with 0: %s",
                len(missing),
                sorted(missing),
            )
            for col in missing:
                X = X.copy()
                X[col] = 0.0
        return X[self.feature_names_]

    def __repr__(self) -> str:
        status = "fitted" if self._model is not None else "unfitted"
        n_feat = len(self.feature_names_) if self.feature_names_ else 0
        return (
            f"LGBMModel(status={status}, n_features={n_feat}, "
            f"scale_pos_weight={self.scale_pos_weight_})"
        )
