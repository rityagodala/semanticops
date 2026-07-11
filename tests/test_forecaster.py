"""Tests for the Forecaster class."""

import numpy as np
import pandas as pd
import pytest
from semanticops.feature_extractor import FeatureSet
from semanticops.forecaster import Forecaster, TrainingReport, ForecastResult


@pytest.fixture
def simple_feature_set():
    rng = np.random.default_rng(0)
    n = 100
    X = pd.DataFrame({
        "feat_a": rng.normal(0, 1, n),
        "feat_b": rng.normal(0, 1, n),
    })
    y = pd.Series(X["feat_a"] * 2 + X["feat_b"] + rng.normal(0, 0.1, n), name="target")
    return FeatureSet(
        X=X, y=y, feature_names=["feat_a", "feat_b"],
        target_name="target", n_samples=n, n_features=2,
        extraction_log=[],
    )


def test_train_returns_report(simple_feature_set):
    f = Forecaster()
    report = f.train(simple_feature_set)
    assert isinstance(report, TrainingReport)


def test_val_mae_is_non_negative(simple_feature_set):
    f = Forecaster()
    report = f.train(simple_feature_set)
    assert report.val_mae >= 0


def test_predict_returns_array(simple_feature_set):
    f = Forecaster()
    f.train(simple_feature_set)
    result = f.predict(simple_feature_set.X)
    assert isinstance(result, ForecastResult)
    assert len(result.predictions) == len(simple_feature_set.X)


def test_predict_before_train_raises(simple_feature_set):
    f = Forecaster()
    with pytest.raises(RuntimeError, match="train"):
        f.predict(simple_feature_set.X)


def test_feature_importance_non_empty(simple_feature_set):
    f = Forecaster(model_type="gradient_boosting")
    f.train(simple_feature_set)
    result = f.predict(simple_feature_set.X)
    assert len(result.top_features) > 0


def test_ridge_model_type(simple_feature_set):
    f = Forecaster(model_type="ridge")
    report = f.train(simple_feature_set)
    assert report.model_type == "ridge"


def test_cv_scores_length(simple_feature_set):
    f = Forecaster(n_cv_folds=2)
    report = f.train(simple_feature_set)
    assert len(report.cv_scores) == 2
