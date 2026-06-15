"""Shared pytest fixtures and path helpers."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from adenopredict import load_model

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EXAMPLE_CSV = PROJECT_ROOT / "examples" / "df_example.csv"
MODEL_PATH = PROJECT_ROOT / "examples" / "best_model_svm.pkl"


@pytest.fixture(scope="session")
def example_df() -> pd.DataFrame:
    return pd.read_csv(EXAMPLE_CSV)


@pytest.fixture(scope="session")
def model():
    return load_model(str(MODEL_PATH))
