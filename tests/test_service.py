"""Tests for the Streamlit-agnostic prediction service."""

from __future__ import annotations

import pandas as pd

from app.service import PredictionResult, run_prediction


def test_run_prediction_extracts_ground_truth(model, example_df):
    result = run_prediction(model, example_df)
    assert isinstance(result, PredictionResult)
    assert result.has_ground_truth
    assert result.ground_truth is not None
    assert set(result.ground_truth.unique()) <= {0, 1}


def test_run_prediction_without_target(model):
    df = pd.DataFrame({"age": [40], "sex": ["M"], "diameter": [2.5], "adc": [0.5]})
    result = run_prediction(model, df)
    assert not result.has_ground_truth
    assert result.ground_truth is None
    assert len(result.predictions) == 1
