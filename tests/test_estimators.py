"""Tests for the substitutable probability-estimator strategies (LSP)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from adenopredict.estimators import (
    DecisionFunctionEstimator,
    LabelEstimator,
    ProbabilityEstimator,
    ProbaEstimator,
    make_estimator,
)

X = pd.DataFrame({"age": [40, 60], "sex_M": [1, 0], "diameter": [2.0, 3.0], "adc": [1.0, 2.0]})


class FakeProbaModel:
    def predict_proba(self, X):
        return np.column_stack([np.full(len(X), 0.3), np.full(len(X), 0.7)])

    def decision_function(self, X):
        return np.linspace(-1, 1, len(X))

    def predict(self, X):
        return np.ones(len(X))


class FakeDecisionModel:
    def decision_function(self, X):
        return np.array([-2.0, 2.0])

    def predict(self, X):
        return np.array([0, 1])


class FakeLabelModel:
    def predict(self, X):
        return np.array([0, 1])


@pytest.mark.parametrize(
    "strategy, model",
    [
        (ProbaEstimator, FakeProbaModel()),
        (DecisionFunctionEstimator, FakeDecisionModel()),
        (LabelEstimator, FakeLabelModel()),
    ],
)
def test_strategies_honour_contract(strategy, model):
    # LSP: every subtype returns a 1-D array of len(X) bounded in [0, 1].
    estimator = strategy(model)
    assert isinstance(estimator, ProbabilityEstimator)
    proba = estimator.probabilities(X)
    assert proba.shape == (len(X),)
    assert np.all((proba >= 0.0) & (proba <= 1.0))


def test_proba_estimator_uses_positive_class():
    proba = ProbaEstimator(FakeProbaModel()).probabilities(X)
    assert np.allclose(proba, 0.7)


def test_decision_function_normalizes_to_unit_interval():
    proba = DecisionFunctionEstimator(FakeDecisionModel()).probabilities(X)
    assert proba.min() == 0.0
    assert proba.max() == 1.0


def test_make_estimator_prefers_proba_then_decision_then_label():
    assert isinstance(make_estimator(FakeProbaModel()), ProbaEstimator)
    assert isinstance(make_estimator(FakeDecisionModel()), DecisionFunctionEstimator)
    assert isinstance(make_estimator(FakeLabelModel()), LabelEstimator)


def test_make_estimator_rejects_unsupported_model():
    with pytest.raises(TypeError):
        make_estimator(object())
