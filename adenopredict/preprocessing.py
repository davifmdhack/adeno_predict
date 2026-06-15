"""Deterministic feature preparation for AdenoPredict.

This module is the single, testable place where raw input rows are validated
and turned into the feature matrix expected by the trained model. It replaces
the previous in-line preprocessing that refit a ``OneHotEncoder(drop="first")``
on every call.

Why the change matters
----------------------
``OneHotEncoder(drop="first")`` derives its output from the categories *present
in the data it is fit on*. When a batch contained a single sex -- most notably
the single-row "Individual Patient" prediction -- the encoder produced no
``sex_M`` column and the code silently fell back to ``sex_M = 0``. A male
patient was therefore scored as if female. Encoding ``sex`` deterministically
(``M`` -> 1, anything else -> 0) removes that data dependence and fixes the bug
while matching the training-time schema exactly for mixed-sex batches.
"""

from __future__ import annotations

import pandas as pd

from .constants import (
    FEATURE_ORDER,
    LABEL_MAP,
    MALE_LABEL,
    REQUIRED_COLUMNS,
    TARGET_COLUMN,
)


def validate_input(df: pd.DataFrame) -> None:
    """Ensure the input DataFrame contains every required column.

    Args:
        df: Raw input features.

    Raises:
        ValueError: If one or more of :data:`REQUIRED_COLUMNS` is missing.
    """
    missing = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}. Expected columns: {REQUIRED_COLUMNS}")


def encode_sex(sex: pd.Series) -> pd.Series:
    """Encode the ``sex`` column as the binary ``sex_M`` feature.

    The encoding is deterministic and row-independent: ``"M"`` maps to ``1`` and
    every other value maps to ``0``. This mirrors the training-time
    ``OneHotEncoder(drop="first")`` output for mixed-sex data while remaining
    correct for single-row or single-sex inputs.

    Args:
        sex: Series of raw sex labels (e.g. ``"M"`` / ``"F"``).

    Returns:
        Integer series named ``sex_M`` aligned to the input index.
    """
    encoded = (sex.astype("string").str.upper() == MALE_LABEL).astype(int)
    return encoded.rename("sex_M")


def map_target(target: pd.Series) -> pd.Series:
    """Map human-readable consistency labels to integer classes.

    Values already expressed as integers (or unknown labels) are preserved, so
    the function is safe to apply to partially-encoded data.

    Args:
        target: Series of consistency labels (``"soft"`` / ``"non-soft"``).

    Returns:
        Series with labels replaced by :data:`LABEL_MAP` integers.
    """
    return target.map(LABEL_MAP).fillna(target)


def prepare_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series | None]:
    """Validate and transform raw input into the model's feature matrix.

    Args:
        df: Raw input ``DataFrame`` with :data:`REQUIRED_COLUMNS` and an
            optional :data:`TARGET_COLUMN`.

    Returns:
        Tuple ``(X, y)`` where ``X`` holds the features in
        :data:`FEATURE_ORDER` and ``y`` is the optional integer-encoded target
        (``None`` when the target column is absent).
    """
    validate_input(df)

    features = df.copy()
    features["sex_M"] = encode_sex(features["sex"]).to_numpy()

    target: pd.Series | None = None
    if TARGET_COLUMN in features.columns:
        target = map_target(features[TARGET_COLUMN])

    return features[FEATURE_ORDER], target
