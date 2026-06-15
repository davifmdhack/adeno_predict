# Changelog

All notable changes to this project are documented in this file. The format is
based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this
project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2026-06-15

Full structural refactor applying the Liskov Substitution Principle and general
script best-practices, while preserving the public API
(`load_model`, `predict_dataframe`) and the deployed Streamlit application.

### Fixed
- **Individual / single-sex predictions were wrong.** Feature preparation refit a
  `OneHotEncoder(drop="first")` on every call, so a batch containing a single sex
  (including the single-row *Individual Patient* form) produced no `sex_M` column
  and silently fell back to `sex_M = 0`, scoring **male patients as female**.
  Encoding is now deterministic (`M -> 1`, otherwise `0`). For example, a male
  patient with `age=40, diameter=2.5, adc=0.5` was previously reported as
  *non-soft* (~95%); it is now correctly reported as *soft* (~29%). Mixed-sex
  batch outputs are unchanged (verified by a non-regression test).

### Added
- `adenopredict.constants`: single source of truth for the schema, label maps and
  decision threshold (removes constants duplicated across modules).
- `adenopredict.preprocessing`: deterministic, unit-tested feature preparation
  (`validate_input`, `encode_sex`, `map_target`, `prepare_features`).
- `adenopredict.estimators`: a Liskov-compliant `ProbabilityEstimator` strategy
  hierarchy (`ProbaEstimator`, `DecisionFunctionEstimator`, `LabelEstimator`) with
  a `make_estimator` factory, replacing inline `hasattr` branching. Every subtype
  honours the same contract — return positive-class probabilities in `[0, 1]` of
  shape `(n,)` — so they are fully substitutable.
- `app/service.py`: a Streamlit-agnostic prediction service (`run_prediction`,
  `PredictionResult`) separating data/model logic from rendering.
- Test suite (`tests/`, pytest) covering the sex-encoding fix, estimator
  strategies, input validation and end-to-end inference, including a
  non-regression check against the bundled SVM model.
- Tooling: `ruff` (lint + format) and `pytest` configuration in `pyproject.toml`,
  plus a GitHub Actions CI workflow (Python 3.11 / 3.12).

### Changed
- `adenopredict.inference.predict_dataframe` is now a thin orchestrator over the
  preprocessing and estimator modules and accepts an optional `threshold`.
- Streamlit page renderers (`app/pages.py`) were decomposed into small helpers,
  with specific exception handling and no silent failures or redundant local
  imports.
- `examples/model_apply.py` is now a clean `argparse` CLI that reuses the library
  instead of re-implementing preprocessing, and no longer auto-installs packages.
- `app/config.py` and `app/data.py` reuse the shared constants and `pathlib`.

## [0.1.0]

- Initial release: SVM-based inference library and Streamlit application.

[0.2.0]: https://github.com/davifmdhack/adeno_predict/releases/tag/v0.2.0
