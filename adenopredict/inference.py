from __future__ import annotations

from typing import List, Tuple

import pandas as pd
import joblib
from sklearn.preprocessing import OneHotEncoder


REQUIRED_COLUMNS: List[str] = ["age", "sex", "diameter", "adc"]
TARGET_COLUMN: str = "consistency"


def _validate_input_dataframe(df: pd.DataFrame) -> None:
    """Validate that the input DataFrame contains required columns.

    Args:
        df: Input features `DataFrame`.

    Raises:
        ValueError: If one or more required columns are missing.
    """
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing columns: {missing}. Expected columns: {REQUIRED_COLUMNS}"
        )


def _prepare_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series | None]:
    """Prepare features to match the training-time schema.

    - Map optional target labels (soft/non-soft) to integers if present.
    - One-hot encode `sex` with `drop='first'` to yield `sex_M`.
    - Reorder/select feature columns in the expected order.

    Args:
        df: Raw input `DataFrame` with columns `age, sex, diameter, adc` and
            optional `consistency`.

    Returns:
        A tuple `(X, y)` where `X` is the prepared feature `DataFrame` and `y` is
        the optional target series (or `None` if not provided).
    """
    df_local = df.copy()

    # Mapear consistência se existir
    y = None
    if TARGET_COLUMN in df_local.columns:
        replace_map_outcome = {"soft": 0, "non-soft": 1}
        df_local[TARGET_COLUMN] = df_local[TARGET_COLUMN].map(replace_map_outcome).fillna(df_local[TARGET_COLUMN])
        y = df_local[TARGET_COLUMN]

    # One-hot para sexo (drop first para gerar sex_M compatível)
    encoder = OneHotEncoder(drop="first")
    encoded = encoder.fit_transform(df_local[["sex"]]).toarray()
    encoded_sex = pd.DataFrame(
        encoded,
        columns=encoder.get_feature_names_out(["sex"]),
        index=df_local.index,
    )

    df_encoded = pd.concat([df_local.drop(["sex"], axis=1), encoded_sex], axis=1)

    # Ordem das features usada no código original
    x_columns = ["age", "sex_M", "diameter", "adc"]
    for col in x_columns:
        if col not in df_encoded.columns:
            # Garantir coluna se por acaso nomes vindo do encoder forem diferentes
            df_encoded[col] = 0

    X = df_encoded[x_columns]
    return X, y


def load_model(model_path: str):
    """Load a serialized scikit-learn Pipeline/estimator.

    Args:
        model_path: Path to the `.pkl` file.

    Returns:
        The loaded estimator (ideally a scikit-learn Pipeline).
    """
    return joblib.load(model_path)


def predict_dataframe(model, df: pd.DataFrame) -> pd.DataFrame:
    """Predict non-soft consistency probability for each row in a DataFrame.

    Args:
        model: A fitted estimator or scikit-learn Pipeline.
        df: Input `DataFrame` with required columns.

    Returns:
        DataFrame with columns:
        - `proba_non_soft`: probability of the positive class.
        - `predicted_label`: integer label (0 soft, 1 non-soft).
        - `predicted_consistency`: human-readable label.
    """
    _validate_input_dataframe(df)
    X, y = _prepare_features(df)

    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)[:, 1]
    else:
        if hasattr(model, "decision_function"):
            scores = model.decision_function(X)
            min_s, max_s = scores.min(), scores.max()
            proba = (scores - min_s) / (max_s - min_s) if max_s > min_s else (scores > 0).astype(float)
        else:
            preds = model.predict(X)
            proba = preds.astype(float)

    preds = (proba >= 0.5).astype(int)

    result = pd.DataFrame(
        {
            "proba_non_soft": proba,
            "predicted_label": preds,
        },
        index=df.index,
    )

    result["predicted_consistency"] = result["predicted_label"].map({0: "soft", 1: "non-soft"})
    return result
