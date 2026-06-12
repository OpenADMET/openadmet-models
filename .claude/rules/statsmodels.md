---
description: 'Statsmodels conventions: formula versus array API choice, explicit constants, robust covariance, results-object discipline, and honest reporting of fit diagnostics. Apply when writing or reviewing regression, GLM, or time-series models with statsmodels.'
paths:
- '**/*.py'
provenance: shared/rules/stack/statsmodels.md @ 85dde8d
---

You are an expert in statistical modeling with statsmodels.

## Principles

1. A model's specification is visible in the code: terms, constant, and covariance choice are explicit, never implied.
2. The results object is the deliverable; numbers quoted anywhere must be readable off it.

## Directives

- Prefer the formula API (`smf.ols("y ~ x1 + x2", data=df)`) for named-column data; it documents the specification and adds the intercept itself.
- With the array API (`sm.OLS(y, X)`), add the constant explicitly via `sm.add_constant(X)`; a missing intercept is the classic silent misfit.
- Choose the covariance estimator deliberately (`fit(cov_type="HC3")` for heteroskedasticity-robust, cluster options where observations group); the default assumes homoskedasticity.
- Extract numbers programmatically (`results.params`, `results.conf_int()`, `results.pvalues`) for anything reported downstream; `summary()` is for reading, never for parsing.
- Keep model objects and results objects distinct: `model = smf.ols(...)`, `results = model.fit()`; refitting with variations goes through new calls, never mutation.
- Report fit alongside diagnostics honestly: condition numbers, residual checks, and the n actually used after missing-data handling (`missing="drop"` shrinks the sample silently).

## Anti-hallucination

| Banned | Correct |
|---|---|
| `sm.OLS(y, X).fit()` without `add_constant` | `sm.OLS(y, sm.add_constant(X)).fit()` (or the formula API) |
| parsing values out of `summary()` text | `results.params` / `results.pvalues` / `results.conf_int()` |
| `from statsmodels.formula.api import *` | `import statsmodels.formula.api as smf` |
| sklearn idioms (`model.predict` before `fit`, `coef_`) | statsmodels API (`results.predict`, `results.params`) |
