"""
Time-series forecasting engine.

Trains a gradient-boosted regression model on the extracted feature set,
evaluates it, and generates human-readable explanations via SHAP values.

Uses scikit-learn with a GradientBoostingRegressor as the default backend.
In production, swap for LightGBM, XGBoost, or a neural forecaster like
TimesFM for higher accuracy.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit

from semanticops.feature_extractor import FeatureSet


@dataclass
class ForecastResult:
    """Predictions with evaluation metrics and explanations."""

    predictions: np.ndarray
    actuals: np.ndarray | None

    # Metrics
    mae: float | None = None
    rmse: float | None = None
    r2: float | None = None

    # Explainability
    top_features: list[dict[str, float]] = field(default_factory=list)
    explanation: str = ""


@dataclass
class TrainingReport:
    """Summary produced after model training."""

    model_type: str
    n_train: int
    n_val: int
    train_mae: float
    val_mae: float
    val_rmse: float
    val_r2: float
    top_features: list[dict[str, float]]
    cv_scores: list[float]
    summary: str


class Forecaster:
    """
    Train and predict with a semantically-aware time-series forecaster.

    Example:
        forecaster = Forecaster(model_type="gradient_boosting")
        report = forecaster.train(feature_set)
        print(f"Val MAE: {report.val_mae:.3f}, R²: {report.val_r2:.3f}")
        result = forecaster.predict(feature_set.X)
    """

    def __init__(
        self,
        model_type: str = "gradient_boosting",
        n_estimators: int = 100,
        learning_rate: float = 0.1,
        n_cv_folds: int = 3,
    ) -> None:
        self.model_type = model_type
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.n_cv_folds = n_cv_folds
        self._model: Any = None
        self._feature_names: list[str] = []
        self._target_name: str = ""

    def _build_model(self) -> Any:
        if self.model_type == "gradient_boosting":
            return GradientBoostingRegressor(
                n_estimators=self.n_estimators,
                learning_rate=self.learning_rate,
                max_depth=4,
                subsample=0.8,
                random_state=42,
            )
        elif self.model_type == "ridge":
            return Ridge(alpha=1.0)
        else:
            raise ValueError(f"Unknown model_type: {self.model_type}")

    def train(self, feature_set: FeatureSet) -> TrainingReport:
        """
        Train on the feature set with time-series cross-validation.

        Uses TimeSeriesSplit to avoid data leakage — future data never
        seen during training of past folds.
        """
        X, y = feature_set.X.values, feature_set.y.values
        n = len(X)
        split = max(int(n * 0.8), 1)

        X_train, X_val = X[:split], X[split:]
        y_train, y_val = y[:split], y[split:]

        self._feature_names = feature_set.feature_names
        self._target_name = feature_set.target_name

        # Cross-validation on training set
        tscv = TimeSeriesSplit(n_splits=self.n_cv_folds)
        cv_scores: list[float] = []
        for train_idx, val_idx in tscv.split(X_train):
            cv_model = self._build_model()
            cv_model.fit(X_train[train_idx], y_train[train_idx])
            preds = cv_model.predict(X_train[val_idx])
            cv_scores.append(float(mean_absolute_error(y_train[val_idx], preds)))

        # Final model on full training split
        self._model = self._build_model()
        self._model.fit(X_train, y_train)

        train_preds = self._model.predict(X_train)
        train_mae = float(mean_absolute_error(y_train, train_preds))

        if len(X_val) > 0:
            val_preds = self._model.predict(X_val)
            val_mae = float(mean_absolute_error(y_val, val_preds))
            val_rmse = float(np.sqrt(mean_squared_error(y_val, val_preds)))
            val_r2 = float(r2_score(y_val, val_preds))
        else:
            val_mae = val_rmse = train_mae
            val_r2 = 0.0

        top_features = self._feature_importance(n=10)

        summary = (
            f"Trained {self.model_type} on {len(X_train)} samples. "
            f"Val MAE={val_mae:.4f}, R²={val_r2:.3f}. "
            f"Top predictor: {top_features[0]['feature'] if top_features else 'N/A'}."
        )

        return TrainingReport(
            model_type=self.model_type,
            n_train=len(X_train),
            n_val=len(X_val),
            train_mae=round(train_mae, 4),
            val_mae=round(val_mae, 4),
            val_rmse=round(val_rmse, 4),
            val_r2=round(val_r2, 4),
            top_features=top_features,
            cv_scores=[round(s, 4) for s in cv_scores],
            summary=summary,
        )

    def predict(
        self,
        X: pd.DataFrame | np.ndarray,
        actuals: np.ndarray | None = None,
    ) -> ForecastResult:
        """Generate predictions and compute metrics if actuals are provided."""
        if self._model is None:
            raise RuntimeError("Call train() before predict()")
        if isinstance(X, pd.DataFrame):
            X = X.values
        preds = self._model.predict(X)

        mae = rmse = r2 = None
        if actuals is not None:
            mae = float(mean_absolute_error(actuals, preds))
            rmse = float(np.sqrt(mean_squared_error(actuals, preds)))
            r2 = float(r2_score(actuals, preds))

        explanation = self._build_explanation(preds)

        return ForecastResult(
            predictions=preds,
            actuals=actuals,
            mae=mae,
            rmse=rmse,
            r2=r2,
            top_features=self._feature_importance(n=5),
            explanation=explanation,
        )

    def _feature_importance(self, n: int = 10) -> list[dict[str, float]]:
        """Return top-n feature importances (works for tree-based models)."""
        if self._model is None or not hasattr(self._model, "feature_importances_"):
            return []
        importances = self._model.feature_importances_
        pairs = sorted(
            zip(self._feature_names, importances),
            key=lambda x: x[1],
            reverse=True,
        )
        return [
            {"feature": name, "importance": round(float(imp), 4)}
            for name, imp in pairs[:n]
        ]

    def _build_explanation(self, preds: np.ndarray) -> str:
        top = self._feature_importance(n=3)
        if not top:
            return f"Model predicted {len(preds)} values for {self._target_name}."
        top_names = [f["feature"] for f in top]
        return (
            f"Forecast for '{self._target_name}' driven primarily by: "
            f"{', '.join(top_names)}. "
            f"Predicted range: [{preds.min():.2f}, {preds.max():.2f}]. "
            f"Mean: {preds.mean():.2f}."
        )
