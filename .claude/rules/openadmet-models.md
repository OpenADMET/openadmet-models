---
description: 'OpenADMET model-library conventions: the registry and lazy-loader pattern, the Pydantic ABC component contract, Lightning-only training, and repository-specific test discipline. Apply when adding or modifying models, featurizers, trainers, evaluators, splitters, or their tests.'
paths: ['openadmet/**/*.py']
---

You are an expert contributor to openadmet-models, a registry-based ML library for ADMET property prediction. This rule covers conventions specific to this repository; for general Python, testing, ML, and chemistry guidance see `python-core`, `python-testing`, `machine-learning`, and `medicinal-chemistry`.

## Principles

1. Components are interchangeable by configuration; a new model, featurizer, or splitter must be reachable through its registry without touching call sites.
2. The base-class contract is the API; honor it rather than special-casing a concrete class downstream.
3. Importing the package stays cheap; heavy dependencies load only when a component group is actually used.

## Registries and the lazy loader

- Register a component by decorating its class with `@models.register("Key")` (or the matching registry: `featurizers`, `trainers`, `evaluators`, `splitters`, `ensemblers`, `transforms`, `comparisons`), each a `ClassRegistry(unique=True)` in its subpackage's `*_base.py`.
- After adding a concrete module, add its import path to the matching group list in `openadmet/models/_registry_loader.py`; the loader imports it so the decorator runs. Registration is by decorator plus loader entry, never by `__all__` or wildcard imports.
- Do not trigger registration with eager top-level imports in `registries.py`; keep that module cheap and let `load_group()` / `load_all()` do the work.

## Component contract

- Every component subclasses a Pydantic `BaseModel` ABC and implements `build()`, `save()`, `load()`, and `serialize()`; a component that cannot honor all four does not belong in a registry.
- Models extend `PickleableModelBase` (sklearn-style estimators) or `LightningModelBase` (deep learning); pick the family that matches the estimator rather than reimplementing persistence.
- Hyperparameters are Pydantic fields. Set `model_config = ConfigDict(extra="allow")` so underlying-library kwargs pass through to the estimator.
- Deep learning training goes through PyTorch Lightning (`lightning.pytorch`); do not hand-roll vanilla PyTorch training loops.

## Tests

- Standardize on `pytest-mock`'s `mocker` fixture; never write custom dummy classes or bespoke mock fixtures. Mock heavy I/O and external dependencies, never the component under test (see `python-testing` for the full test-double order).
- Exercise the real `save()` / `load()` / `serialize()` round trip for new models (the `test_save_load_pickleable` pattern), asserting state is restored rather than only that a file appears.
- When testing splitters or clustering, assert the resulting train/validation/test index sets are mutually exclusive, and use synthetic data with diverse SMILES scaffolds so the split logic is actually exercised (see `machine-learning` on leakage).
