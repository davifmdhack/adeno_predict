"""Model loading and inference for AdenoPredict.

This module keeps a small, stable public surface -- :func:`load_model` and
:func:`predict_dataframe` -- while delegating the details to focused
collaborators: :mod:`adenopredict.preprocessing` (feature preparation) and
:mod:`adenopredict.estimators` (substitutable probability strategies).
"""

from __future__ import annotations

import joblib
import pandas as pd

from .constants import DEFAULT_THRESHOLD, INV_LABEL_MAP
from .estimators import make_estimator
from .preprocessing import prepare_features


def load_model(model_path: str):
    """Load a serialized scikit-learn estimator or ``Pipeline``.

    Args:
        model_path: Path to the ``.pkl`` file.

    Returns:
        The loaded estimator (ideally a scikit-learn ``Pipeline``).
    """
    return joblib.load(model_path)


def predict_dataframe(
    model,
    df: pd.DataFrame,
    threshold: float = DEFAULT_THRESHOLD,
) -> pd.DataFrame:
    """Predict non-soft consistency probability for each row of ``df``.

    Args:
        model: A fitted estimator or scikit-learn ``Pipeline``.
        df: Input ``DataFrame`` with the required feature columns.
        threshold: Probability cut-off for the positive (``non-soft``) class.

    Returns:
        DataFrame indexed like ``df`` with columns:

        * ``proba_non_soft``: probability of the positive class in ``[0, 1]``.
        * ``predicted_label``: integer label (``0`` soft, ``1`` non-soft).
        * ``predicted_consistency``: human-readable label.
    """
    X, _ = prepare_features(df)
    proba = make_estimator(model).probabilities(X)
    labels = (proba >= threshold).astype(int)

    result = pd.DataFrame(
        {"proba_non_soft": proba, "predicted_label": labels},
        index=df.index,
    )
    result["predicted_consistency"] = result["predicted_label"].map(INV_LABEL_MAP)
    return result
