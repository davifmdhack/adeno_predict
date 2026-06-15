"""Tests for deterministic feature preparation, including the sex-encoding fix."""

from __future__ import annotations

import pandas as pd
import pytest

from adenopredict.constants import FEATURE_ORDER
from adenopredict.preprocessing import (
    encode_sex,
    map_target,
    prepare_features,
    validate_input,
)


def test_encode_sex_male_and_female():
    encoded = encode_sex(pd.Series(["M", "F", "m", "f"]))
    assert encoded.tolist() == [1, 0, 1, 0]
    assert encoded.name == "sex_M"


def test_encode_sex_single_male_row_is_one():
    # Regression test: a lone male row must encode to sex_M = 1, not 0.
    assert encode_sex(pd.Series(["M"])).tolist() == [1]


def test_prepare_features_single_male_row_distinct_from_female():
    # The core bug fix: single-row male and female differ in sex_M.
    male = prepare_features(
        pd.DataFrame({"age": [40], "sex": ["M"], "diameter": [2.5], "adc": [0.5]})
    )[0]
    female = prepare_features(
        pd.DataFrame({"age": [40], "sex": ["F"], "diameter": [2.5], "adc": [0.5]})
    )[0]
    assert male["sex_M"].iloc[0] == 1
    assert female["sex_M"].iloc[0] == 0


def test_prepare_features_returns_feature_order(example_df):
    X, y = prepare_features(example_df)
    assert list(X.columns) == FEATURE_ORDER
    assert y is not None
    assert len(X) == len(example_df)


def test_prepare_features_without_target():
    X, y = prepare_features(
        pd.DataFrame({"age": [40], "sex": ["M"], "diameter": [2.5], "adc": [0.5]})
    )
    assert y is None
    assert list(X.columns) == FEATURE_ORDER


def test_map_target_maps_labels():
    assert map_target(pd.Series(["soft", "non-soft"])).tolist() == [0, 1]


def test_validate_input_raises_on_missing_columns():
    with pytest.raises(ValueError, match="Missing columns"):
        validate_input(pd.DataFrame({"age": [40]}))
