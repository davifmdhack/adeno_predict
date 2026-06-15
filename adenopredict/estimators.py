"""Substitutable probability estimators (Liskov-compliant strategy hierarchy).

A fitted scikit-learn model may expose its confidence through different APIs:
``predict_proba`` (calibrated probabilities), ``decision_function`` (unbounded
margins) or only ``predict`` (hard labels). The previous code branched on
``hasattr`` inline inside ``predict_dataframe``.

Here that logic is expressed as a small hierarchy where every subclass honours
the *same* contract, so any one can stand in for another without surprising the
caller -- the Liskov Substitution Principle:

* Precondition: ``X`` is the prepared feature matrix (same for all subtypes).
* Postcondition: ``probabilities(X)`` returns a 1-D ``numpy`` array of length
  ``len(X)`` with every value in ``[0, 1]``; no side effects, no
  ``NotImplementedError``.

:func:`make_estimator` selects the most informative available strategy, in the
same priority order the original code used (proba > decision > label), so model
outputs are unchanged for estimators that expose ``predict_proba``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

#: Index of the positive class (``non-soft``) in scikit-learn's class ordering.
POSITIVE_CLASS_INDEX = 1


class ProbabilityEstimator(ABC):
    """Strategy that turns a fitted model into positive-class probabilities."""

    def __init__(self, model: object) -> None:
        self._model = model

    @abstractmethod
    def probabilities(self, X: pd.DataFrame) -> np.ndarray:
        """Return ``P(positive class)`` in ``[0, 1]`` with shape ``(len(X),)``."""

    @staticmethod
    def supports(model: object) -> bool:
        """Whether this strategy can be applied to ``model``."""
        return False


class ProbaEstimator(ProbabilityEstimator):
    """Use a model's calibrated ``predict_proba`` output directly."""

    @staticmethod
    def supports(model: object) -> bool:
        return hasattr(model, "predict_proba")

    def probabilities(self, X: pd.DataFrame) -> np.ndarray:
        proba = self._model.predict_proba(X)[:, POSITIVE_CLASS_INDEX]
        return np.asarray(proba, dtype=float)


class DecisionFunctionEstimator(ProbabilityEstimator):
    """Min-max normalize ``decision_function`` margins into ``[0, 1]``."""

    @staticmethod
    def supports(model: object) -> bool:
        return hasattr(model, "decision_function")

    def probabilities(self, X: pd.DataFrame) -> np.ndarray:
        scores = np.asarray(self._model.decision_function(X), dtype=float)
        min_score, max_score = scores.min(), scores.max()
        if max_score > min_score:
            return (scores - min_score) / (max_score - min_score)
        # Degenerate (constant) scores: fall back to sign of the margin.
        return (scores > 0).astype(float)


class LabelEstimator(ProbabilityEstimator):
    """Last-resort strategy: treat hard ``predict`` labels as probabilities."""

    @staticmethod
    def supports(model: object) -> bool:
        return hasattr(model, "predict")

    def probabilities(self, X: pd.DataFrame) -> np.ndarray:
        return np.asarray(self._model.predict(X), dtype=float)


#: Strategies ordered from most to least informative.
_STRATEGY_PRIORITY = (ProbaEstimator, DecisionFunctionEstimator, LabelEstimator)


def make_estimator(model: object) -> ProbabilityEstimator:
    """Return the most informative :class:`ProbabilityEstimator` for ``model``.

    Args:
        model: A fitted estimator or scikit-learn ``Pipeline``.

    Returns:
        A ready-to-use probability estimator wrapping ``model``.

    Raises:
        TypeError: If ``model`` exposes none of ``predict_proba``,
            ``decision_function`` or ``predict``.
    """
    for strategy in _STRATEGY_PRIORITY:
        if strategy.supports(model):
            return strategy(model)
    raise TypeError(
        "Unsupported model: expected one of predict_proba, decision_function or predict."
    )
