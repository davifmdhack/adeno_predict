"""CSV input/output helpers for the Streamlit app."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .config import EXAMPLE_CSV_PATH, RESULTS_PATH


def load_example_dataframe(csv_path: str = EXAMPLE_CSV_PATH) -> pd.DataFrame:
    """Load the bundled example dataset."""
    return pd.read_csv(csv_path)


def save_results(df: pd.DataFrame, path: str = RESULTS_PATH) -> str:
    """Write ``df`` to ``path`` as CSV, creating parent directories if needed.

    Returns:
        The path the results were written to.
    """
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(destination, index=False)
    return str(destination)
