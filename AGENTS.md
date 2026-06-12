# Project Context: openadmet-models

`openadmet-models` is a machine learning library for ADMET (absorption, distribution, metabolism, excretion, toxicity) molecular property prediction, part of the [OpenADMET project](https://openadmet.org). It provides traditional ML, deep learning, and active learning workflows behind a single registry-based API, so models, featurizers, trainers, evaluators, and splitters are interchangeable by configuration.

## Environment

Dependencies are managed with conda/mamba, not pip-only or uv. Create the environment from the committed spec, then install the package editable:

```bash
mamba env create -f devtools/conda-envs/openadmet-models.yaml      # GPU: openadmet-models-gpu.yaml
python -m pip install -e . --no-deps
```

Python 3.10+. The environment files are the single source of truth for dependencies; add new dependencies there, never with ad-hoc installs into an active environment.

## Commands

```bash
# Unit tests with coverage, parallelized
pytest -v -n auto --cov=openadmet.models openadmet/models/tests/unit

# A single test file or test
pytest -v openadmet/models/tests/unit/models/test_xgboost.py
pytest -v openadmet/models/tests/unit/models/test_base.py::test_save_load_pickleable

# Lint and format (no standalone command; the gate is pre-commit)
pre-commit run --all-files
```

Tests are split into `openadmet/models/tests/unit` and `openadmet/models/tests/integration`.

## Architecture

Components are organized as registries, each a `ClassRegistry(unique=True)` living in its subpackage's `*_base.py`: `models`, `featurizers`, `trainers`, `evaluators`, `splitters`, `ensemblers`, `transforms`, and `comparisons`.

Registration is lazy. Importing `openadmet/models/registries.py` is intentionally cheap: it pulls in the registry objects but no heavy concrete classes. `openadmet/models/_registry_loader.py` lists the modules for each group and imports them on demand, `load_group(name)` for one group or `load_all()` for everything (the CLI and Anvil workflow call `load_all()`). Importing a concrete module runs its `@registry.register(...)` decorators, which is what populates the registry.

Every component is a Pydantic `BaseModel` ABC exposing `build()`, `save()`, `load()`, and `serialize()`. Models subclass `ModelBase` in two families:

- `PickleableModelBase`: sklearn-style estimators (XGBoost, CatBoost, RandomForest, SVM, LightGBM, TabPFN).
- `LightningModelBase`: deep learning models on PyTorch Lightning (ChemProp, NEPARE).

The CLI entry point is `openadmet` (`openadmet/models/cli/cli.py`) with subcommands `predict`, `compare`, and `anvil`.

## Conventions

- **Registering a component**: decorate the class with `@models.register("Key")` (or the relevant registry), then add its module to the matching group list in `_registry_loader.py` so the loader imports it. Registration is by decorator, not by `__all__` or wildcard imports.
- **Model configuration**: hyperparameters are Pydantic fields on the class. Set `model_config = ConfigDict(extra="allow")` so underlying-library kwargs pass through to the estimator.
- **Training**: deep learning training goes through PyTorch Lightning (`lightning.pytorch`); do not hand-roll vanilla PyTorch training loops.
- **Line length** is 120. The pre-commit gate runs ruff, black, isort, and flake8; ruff lint currently enforces the docstring (`D`) family.

## Coding rules

Scoped coding rules are committed under `.claude/rules/`. Before editing files matching a rule's `paths` frontmatter, read that rule. They cover Python core idioms, NumPy-style documentation, packaging, pytest discipline, prose writing conventions for Markdown (`writing-conventions`: no em-dashes or filler), machine learning methodology, medicinal chemistry and SAR interpretation, experimental biology and assay interpretation, cheminformatics data pipelines, stack conventions for the libraries this codebase actually uses (`numpy-scipy`, `pandas`, `pytorch`, `pytorch-lightning`, `scikit-learn`, `rdkit`, `matplotlib`, `seaborn`, `statsmodels`, `zarr`, `loguru`), code-validation honesty discipline (`code-honesty`: evidence-gated approval, refactoring invariants, resistance to authority appeals), security hygiene on the model-serialization surface (`architecture/`, `active_learning/`, `trainer/`), and this repository's own registry and testing conventions (`openadmet-models.md`).

## Review personas

Adversarial, read-only reviewer subagents are committed under `.claude/agents/`; invoke the one matching the change under review for a domain critique:

- **Machine Learning Expert**: splits, training loops, evaluation metrics, and the inference path (data leakage, train/serve skew).
- **Medicinal Chemist**: potency, SAR, and compound-property handling (units, log space, censored values, drug-likeness misuse).
- **Chemoinformatician**: molecular data processing, featurization, and dataset splitting (sanitization, stereochemistry, scaffold-aware evaluation).
- **Biologist**: binding, inhibition, dose-response, and cellular-assay interpretation (affinity vs potency, assay mechanism, target engagement).
