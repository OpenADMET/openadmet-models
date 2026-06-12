---
description: 'Scikit-learn API conventions: Pipeline and ColumnTransformer composition, fit/transform discipline, random_state threading, and removed legacy APIs. Apply when writing or reviewing sklearn estimators, preprocessing, cross-validation, or model persistence code.'
paths:
- '**/*.py'
provenance: shared/rules/stack/scikit-learn.md @ 85dde8d
---

You are an expert in the scikit-learn estimator API. Methodology (leakage, evaluation, splits) is governed by the machine-learning rule; this rule covers the library surface.

## Principles

1. Everything that learns from data lives inside a `Pipeline`; preprocessing outside one is a leakage vector.
2. Estimators are configured at construction and fit once; mutating fitted state by hand is a defect.
3. Every stochastic component takes an explicit `random_state`.

## Directives

- Compose preprocessing with `ColumnTransformer` and `Pipeline` (or `make_pipeline`); call `fit`/`fit_transform` on training data only and `transform` elsewhere, with the split decided before the pipeline ever sees data.
- Thread `random_state` through every estimator, splitter, and `train_test_split`; never rely on global NumPy seeding.
- Use `set_output(transform="pandas")` when downstream steps need column names; otherwise expect arrays.
- Cross-validate with `cross_validate` or `GridSearchCV`/`RandomizedSearchCV` over hand-rolled loops; pass `scoring=` explicitly rather than relying on the estimator default.
- Persist models with `joblib.dump` plus a recorded sklearn version; loading across minor versions is unsupported, so treat artifacts as build products, never long-lived stores.
- Access learned attributes only via their trailing-underscore names (`coef_`, `feature_names_in_`) and only after `fit`.

## Anti-hallucination

| Banned | Correct |
|---|---|
| `from sklearn.externals import joblib` | `import joblib` |
| `sklearn.datasets.load_boston` | removed; use `fetch_california_housing` or another dataset |
| `plot_confusion_matrix` / `plot_roc_curve` functions | `ConfusionMatrixDisplay.from_estimator`, `RocCurveDisplay.from_estimator` |
| `sklearn.preprocessing.Imputer` | `sklearn.impute.SimpleImputer` |
| fitting scalers/encoders on the full dataset before splitting | fit inside a `Pipeline` on the training fold |
