# Copilot Instructions

## Project Overview

`openadmet-models` is a machine learning library for ADMET (Absorption, Distribution, Metabolism, Excretion, Toxicity) molecular property prediction. It provides traditional ML, deep learning, and active learning workflows through a unified, registry-based API.

## Install & Setup

```bash
mamba env create -f devtools/conda-envs/openadmet-models.yaml
python -m pip install -e --no-deps .
```

## Commands

```bash
# Run all unit tests
pytest -v -n auto --cov=openadmet.models openadmet/models/tests/unit

# Run a single test file
pytest -v openadmet/models/tests/unit/models/test_xgboost.py

# Run a single test
pytest -v openadmet/models/tests/unit/models/test_base.py::test_save_load_pickleable
```

Lint/format is enforced via pre-commit hooks (ruff, black, isort, flake8). There is no standalone lint command — run `pre-commit run --all-files` to check manually.

## Architecture

The library is organized around four registries, each backed by a `ClassRegistry` from `class-registry`:

- **`models`** — ML model implementations (`openadmet/models/architecture/`)
- **`featurizers`** — Molecular feature extractors (`openadmet/models/features/`)
- **`trainers`** — Training loops (`openadmet/models/trainer/`)
- **`evaluators`** — Metrics and cross-validation (`openadmet/models/eval/`)
- **`splitters`** — Data splitting strategies (`openadmet/models/split/`)

All registries are populated in `openadmet/models/registries.py` via wildcard imports. Import order in that file matters — concrete classes must be imported before the registry object.

Every component follows the same pattern: a Pydantic `BaseModel` ABC with `build()`, `save()`, `load()`, and `serialize()` abstract methods. Models fall into two subclasses of `ModelBase`:
- `PickleableModelBase` — sklearn-style models (XGBoost, CatBoost, RF, SVM, LightGBM)
- `LightningModelBase` — deep learning models using PyTorch Lightning (ChemProp, MTENN, NEPARE)

The CLI entry point is `openadmet` (`openadmet/models/cli/cli.py`), with subcommands `predict`, `compare`, and `anvil`.

## Key Conventions

**Registering new components** — Decorate the class with `@models.register("key")` (or the relevant registry). Use wildcard `__all__` exports so `registries.py` picks them up via `from module import *`.

**Model config** — All model hyperparameters are Pydantic fields on the class. Extra kwargs are allowed via `model_config = ConfigDict(extra="allow")` so that underlying library kwargs pass through to the estimator.

**Training loops** — Use PyTorch Lightning (`lightning.pytorch`) for all deep learning training. Do not write vanilla PyTorch training loops.

**Docstrings** — NumPy-style for all classes, methods, and functions. Test files are exempt from docstring requirements.

**Code style**
- Max line length: 120 characters
- Ruff + Black formatting; isort with Black-compatible profile
- Sentence case in comments and print statements; acronyms (MPNN, MVE, ADMET, FFN) stay capitalized
- Do not number steps in comments; do not end comments with a period
