---
description: 'NumPy 2.x and SciPy conventions: vectorization, dtype and copy semantics, the Generator random API, float comparison, and removed 1.x aliases. Apply when writing or reviewing array code, numerical routines, random sampling, integration, interpolation, or sparse-matrix handling.'
paths:
- '**/*.py'
provenance: shared/rules/stack/numpy-scipy.md @ 85dde8d
---

You are an expert in numerical computing with NumPy 2.x and SciPy.

## Principles

1. Vectorize: a Python loop over array elements is a bug until proven necessary.
2. Dtypes are part of the contract; never let silent promotion or copies surprise a caller.
3. Floating-point equality is always tolerance-based.

## Directives

- Random numbers come from `np.random.default_rng(seed)`; pass the `Generator` explicitly through call chains instead of relying on global state.
- Compare floats with `np.isclose` / `np.testing.assert_allclose` with explicit tolerances; never `==` on computed floats.
- State dtypes explicitly at array creation when precision matters; NumPy 2 follows NEP 50, so Python-scalar operations no longer upcast by value.
- `np.array(x, copy=False)` raises when a copy is required (1.x silently copied); use `np.asarray` when a view is acceptable.
- Preallocate output arrays in hot paths; avoid growing arrays by repeated `np.append`/`np.concatenate` in loops.
- New sparse code uses the array interface (`csr_array`, `coo_array`); the `*_matrix` classes are legacy.
- Frozen distributions (`scipy.stats.norm(loc, scale)`) over repeating shape parameters at every call.

## Anti-hallucination

| Banned | Correct |
|---|---|
| `np.float_`, `np.int_`, `np.NaN`, `np.Inf`, `np.unicode_` | `np.float64`, `np.int64`, `np.nan`, `np.inf`, `np.str_` |
| `np.product`, `np.cumproduct`, `np.alltrue`, `np.sometrue`, `np.round_` | `np.prod`, `np.cumprod`, `np.all`, `np.any`, `np.round` |
| `np.random.seed` + `np.random.rand` in new code | `rng = np.random.default_rng(seed)`; `rng.random` |
| imports from `np.core.*` | public `np.*` namespace |
| `scipy.integrate.simps`, `scipy.integrate.trapz` | `simpson`, `trapezoid` |
| `scipy.interpolate.interp1d` in new code | `make_interp_spline`, `CubicSpline`, or `np.interp` |
| `scipy.misc` functions | moved homes (`scipy.datasets`, image I/O libraries) |
