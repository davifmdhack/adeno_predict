"""Single source of truth for the AdenoPredict schema and label conventions.

Centralizing these constants avoids the duplication that previously existed
across ``inference.py``, ``app/config.py``, ``app/pages.py`` and the example
script, where the same column lists and label maps were redefined by hand.
"""

from __future__ import annotations

#: Raw input columns required from the user before any preprocessing.
REQUIRED_COLUMNS: list[str] = ["age", "sex", "diameter", "adc"]

#: Feature order expected by the trained scikit-learn pipeline.
#: ``sex`` is encoded into ``sex_M`` (see :mod:`adenopredict.preprocessing`).
FEATURE_ORDER: list[str] = ["age", "sex_M", "diameter", "adc"]

#: Optional ground-truth column used to compute evaluation metrics.
TARGET_COLUMN: str = "consistency"

#: Mapping from human-readable consistency labels to integer classes.
LABEL_MAP: dict[str, int] = {"soft": 0, "non-soft": 1}

#: Inverse of :data:`LABEL_MAP`, used to render predictions back to text.
INV_LABEL_MAP: dict[int, str] = {value: key for key, value in LABEL_MAP.items()}

#: Value of the ``sex`` column that maps to ``sex_M = 1`` (positive encoding).
MALE_LABEL: str = "M"

#: Decision threshold applied to the positive-class probability.
DEFAULT_THRESHOLD: float = 0.5
