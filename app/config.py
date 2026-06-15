"""App-level configuration: file paths plus the shared schema constants.

The schema constants (``REQUIRED_COLUMNS``, ``TARGET_COLUMN``) are re-exported
from :mod:`adenopredict.constants` so the UI and the inference library always
agree on a single definition.
"""

from adenopredict.constants import REQUIRED_COLUMNS, TARGET_COLUMN

DEFAULT_MODEL_PATH = "examples/best_model_svm.pkl"
EXAMPLE_CSV_PATH = "examples/df_example.csv"
RESULTS_PATH = "results/df_prediction-results.csv"

__all__ = [
    "DEFAULT_MODEL_PATH",
    "EXAMPLE_CSV_PATH",
    "RESULTS_PATH",
    "REQUIRED_COLUMNS",
    "TARGET_COLUMN",
]
