"""Streamlit-agnostic prediction service.

Keeping the data/model logic here (free of any ``streamlit`` import) lets the
page renderers stay thin and makes this layer unit-testable on its own.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from adenopredict import map_target, predict_dataframe
from adenopredict.constants import TARGET_COLUMN


@dataclass
class PredictionResult:
    """Predictions plus the optional ground truth extracted from the input."""

    predictions: pd.DataFrame
    ground_truth: pd.Series | None

    @property
    def has_ground_truth(self) -> bool:
        return self.ground_truth is not None


def run_prediction(model, df: pd.DataFrame) -> PredictionResult:
    """Run the model and, when present, decode the ground-truth target.

    Args:
        model: A fitted estimator or scikit-learn ``Pipeline``.
        df: Raw input ``DataFrame``.

    Returns:
        A :class:`PredictionResult` bundling predictions and ground truth.
    """
    predictions = predict_dataframe(model, df)

    ground_truth: pd.Series | None = None
    if TARGET_COLUMN in df.columns:
        ground_truth = map_target(df[TARGET_COLUMN]).astype(int)

    return PredictionResult(predictions=predictions, ground_truth=ground_truth)
