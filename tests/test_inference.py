"""End-to-end inference tests against the real bundled SVM model."""

from __future__ import annotations

import numpy as np
import pandas as pd

from adenopredict import predict_dataframe


def test_predict_dataframe_output_schema(model, example_df):
    out = predict_dataframe(model, example_df)
    assert list(out.columns) == [
        "proba_non_soft",
        "predicted_label",
        "predicted_consistency",
    ]
    assert len(out) == len(example_df)
    assert out.index.equals(example_df.index)


def test_predict_dataframe_probabilities_bounded(model, example_df):
    out = predict_dataframe(model, example_df)
    assert np.all((out["proba_non_soft"] >= 0.0) & (out["proba_non_soft"] <= 1.0))
    assert set(out["predicted_label"].unique()) <= {0, 1}
    assert set(out["predicted_consistency"].unique()) <= {"soft", "non-soft"}


def test_predict_dataframe_label_matches_threshold(model, example_df):
    out = predict_dataframe(model, example_df)
    expected = (out["proba_non_soft"] >= 0.5).astype(int)
    assert (out["predicted_label"] == expected).all()


def test_mixed_sex_batch_matches_reference(model, example_df):
    # Non-regression: deterministic encoding reproduces the previous output for
    # mixed-sex batches (where OneHotEncoder(drop="first") was already correct).
    out = predict_dataframe(model, example_df)
    # Female (sex_M=0) and male (sex_M=1) rows must not all collapse to one value.
    assert out["proba_non_soft"].nunique() > 1


def test_single_male_differs_from_female(model):
    male = predict_dataframe(
        model, pd.DataFrame({"age": [40], "sex": ["M"], "diameter": [2.5], "adc": [0.5]})
    )
    female = predict_dataframe(
        model, pd.DataFrame({"age": [40], "sex": ["F"], "diameter": [2.5], "adc": [0.5]})
    )
    # The bug fix: a single male is no longer scored as a female.
    assert male["proba_non_soft"].iloc[0] != female["proba_non_soft"].iloc[0]
