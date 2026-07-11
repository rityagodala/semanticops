"""End-to-end pipeline tests with synthetic data."""

import numpy as np
import pandas as pd
import pytest
from semanticops.pipeline import SemanticOpsPipeline, PipelineResult


@pytest.fixture
def synthetic_df():
    """Small synthetic reactor dataset."""
    rng = np.random.default_rng(42)
    n = 200
    return pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=n, freq="h"),
        "T_react": 180 + rng.normal(0, 2, n),
        "P_steam": 5.0 + rng.normal(0, 0.3, n),
        "F_feed": 100 + rng.normal(0, 5, n),
        "purity": 95 + rng.normal(0, 1, n),
    })


@pytest.fixture
def pipeline():
    return SemanticOpsPipeline(model_type="gradient_boosting", lag_windows=[1, 2])


def test_pipeline_succeeds(pipeline, synthetic_df):
    result = pipeline.run_from_dataframe(
        synthetic_df,
        documentation="Reactor monitoring",
        target_column="purity",
    )
    assert result.success, f"Pipeline failed: {result.error}"


def test_pipeline_detects_target(pipeline, synthetic_df):
    result = pipeline.run_from_dataframe(synthetic_df, target_column="purity")
    assert result.schema.target_column == "purity"


def test_training_report_has_metrics(pipeline, synthetic_df):
    result = pipeline.run_from_dataframe(synthetic_df, target_column="purity")
    tr = result.training_report
    assert tr.val_mae >= 0
    assert tr.val_r2 <= 1.0


def test_feature_engineering_increases_columns(pipeline, synthetic_df):
    result = pipeline.run_from_dataframe(synthetic_df, target_column="purity")
    raw_feature_count = 3  # T_react, P_steam, F_feed
    assert result.feature_set.n_features > raw_feature_count


def test_sample_forecast_has_predictions(pipeline, synthetic_df):
    result = pipeline.run_from_dataframe(synthetic_df, target_column="purity")
    assert result.sample_forecast.predictions is not None
    assert len(result.sample_forecast.predictions) > 0


def test_to_api_response_is_serialisable(pipeline, synthetic_df):
    import json
    result = pipeline.run_from_dataframe(synthetic_df, target_column="purity")
    response = pipeline.to_api_response(result)
    # Should not raise
    json.dumps(response)


def test_pipeline_error_on_bad_target(pipeline, synthetic_df):
    result = pipeline.run_from_dataframe(synthetic_df, target_column="nonexistent_col")
    assert not result.success
    assert "nonexistent_col" in result.error or result.error != ""
