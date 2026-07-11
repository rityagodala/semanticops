"""
SemanticOps end-to-end pipeline.

Orchestrates: SchemaAgent → FeatureExtractor → Forecaster.

This is the main entry point for the product:

    pipeline = SemanticOpsPipeline()
    result = pipeline.run(
        csv_path="reactor_data.csv",
        documentation="Batch reactor B-42 monitoring data",
    )
    print(result["training_report"]["summary"])
    print(result["schema"]["target_column"])
"""

from __future__ import annotations

import io
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from semanticops.schema_agent import SchemaAgent, SchemaInterpretation
from semanticops.feature_extractor import FeatureExtractor, FeatureSet
from semanticops.forecaster import Forecaster, TrainingReport, ForecastResult


@dataclass
class PipelineResult:
    """Full pipeline output."""

    schema: SchemaInterpretation
    feature_set: FeatureSet
    training_report: TrainingReport
    sample_forecast: ForecastResult
    success: bool
    error: str = ""


class SemanticOpsPipeline:
    """
    End-to-end SemanticOps pipeline.

    Upload CSV + documentation → get interpretable forecasting model.

    Example:
        pipeline = SemanticOpsPipeline(
            model_type="gradient_boosting",
            lag_windows=[1, 3, 6, 24],
        )
        result = pipeline.run(csv_path="data.csv", documentation="...")
        print(result.training_report.summary)

    Or run from a DataFrame directly:
        result = pipeline.run_from_dataframe(df, documentation="...")
    """

    def __init__(
        self,
        model_type: str = "gradient_boosting",
        lag_windows: list[int] | None = None,
        rolling_windows: list[int] | None = None,
        alpha: float = 0.1,
    ) -> None:
        self.model_type = model_type
        self.lag_windows = lag_windows or [1, 3, 6]
        self.rolling_windows = rolling_windows or [6, 24]
        self._schema_agent = SchemaAgent()
        self._forecaster: Forecaster | None = None

    def run(
        self,
        csv_path: str | Path,
        documentation: str = "",
        target_column: str | None = None,
    ) -> PipelineResult:
        """
        Run the full pipeline from a CSV file path.

        Args:
            csv_path:      Path to input CSV
            documentation: Text description of the dataset
            target_column: Override the auto-detected target column
        Returns:
            PipelineResult with schema, features, model, and predictions
        """
        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            return PipelineResult(
                schema=None, feature_set=None, training_report=None,
                sample_forecast=None, success=False,
                error=f"Failed to read CSV: {e}",
            )
        return self.run_from_dataframe(df, documentation, target_column)

    def run_from_dataframe(
        self,
        df: pd.DataFrame,
        documentation: str = "",
        target_column: str | None = None,
    ) -> PipelineResult:
        """Run the full pipeline from an in-memory DataFrame."""
        try:
            # Step 1: Schema interpretation
            sample_values = {
                col: df[col].dropna().head(5).tolist()
                for col in df.columns
            }
            schema = self._schema_agent.interpret(
                column_names=list(df.columns),
                documentation=documentation,
                sample_values=sample_values,
            )

            # Allow manual target override
            if target_column:
                schema.target_column = target_column
                for col in schema.columns:
                    if col.name == target_column:
                        from semanticops.schema_agent import ColumnRole
                        col.role = ColumnRole.TARGET

            # Step 2: Feature extraction
            extractor = FeatureExtractor(
                schema=schema,
                lag_windows=self.lag_windows,
                rolling_windows=self.rolling_windows,
            )
            feature_set = extractor.extract(df)

            # Step 3: Train forecaster
            self._forecaster = Forecaster(model_type=self.model_type)
            training_report = self._forecaster.train(feature_set)

            # Step 4: Sample forecast on validation slice
            n = feature_set.n_samples
            val_start = int(n * 0.8)
            X_val = feature_set.X.iloc[val_start:]
            y_val = feature_set.y.iloc[val_start:].values
            sample_forecast = self._forecaster.predict(X_val, actuals=y_val)

            return PipelineResult(
                schema=schema,
                feature_set=feature_set,
                training_report=training_report,
                sample_forecast=sample_forecast,
                success=True,
            )

        except Exception as e:
            return PipelineResult(
                schema=None, feature_set=None, training_report=None,
                sample_forecast=None, success=False, error=str(e),
            )

    def predict_new(self, df: pd.DataFrame) -> np.ndarray:
        """
        Run inference on new data using the trained model.
        The dataframe must have the same schema as training data.
        """
        if self._forecaster is None:
            raise RuntimeError("Run pipeline.run() before predict_new()")
        schema = self._schema_agent.interpret(list(df.columns))
        extractor = FeatureExtractor(schema, self.lag_windows, self.rolling_windows)
        # For inference-only, we don't need a target column
        schema.target_column = schema.target_column or df.columns[-1]
        feature_set = extractor.extract(df)
        result = self._forecaster.predict(feature_set.X)
        return result.predictions

    def to_api_response(self, result: PipelineResult) -> dict[str, Any]:
        """Convert PipelineResult to JSON-serialisable dict."""
        if not result.success:
            return {"success": False, "error": result.error}

        tr = result.training_report
        sf = result.sample_forecast
        schema = result.schema

        return {
            "success": True,
            "schema": {
                "domain": schema.domain,
                "summary": schema.summary,
                "target_column": schema.target_column,
                "timestamp_column": schema.timestamp_column,
                "feature_columns": schema.feature_columns,
                "n_features_raw": len(schema.feature_columns),
                "columns": [
                    {
                        "name": c.name,
                        "semantic_name": c.semantic_name,
                        "role": c.role.value,
                        "units": c.units,
                        "description": c.description,
                    }
                    for c in schema.columns
                ],
            },
            "features": {
                "n_samples": result.feature_set.n_samples,
                "n_engineered_features": result.feature_set.n_features,
                "extraction_log": result.feature_set.extraction_log,
            },
            "training": {
                "model_type": tr.model_type,
                "n_train": tr.n_train,
                "n_val": tr.n_val,
                "train_mae": tr.train_mae,
                "val_mae": tr.val_mae,
                "val_rmse": tr.val_rmse,
                "val_r2": tr.val_r2,
                "cv_scores": tr.cv_scores,
                "top_features": tr.top_features,
                "summary": tr.summary,
            },
            "sample_forecast": {
                "mae": sf.mae,
                "rmse": sf.rmse,
                "r2": sf.r2,
                "explanation": sf.explanation,
                "top_features": sf.top_features,
                "predictions_sample": sf.predictions[:10].tolist() if sf.predictions is not None else [],
            },
        }
