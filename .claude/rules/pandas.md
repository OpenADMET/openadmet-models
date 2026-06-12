---
description: 'Pandas 2.x conventions: Copy-on-Write-safe assignment, method chaining, dtype discipline at I/O boundaries, groupby hygiene, and removed 1.x APIs. Apply when writing or reviewing DataFrame transformations, joins, groupbys, or CSV/Parquet ingestion with pandas.'
paths:
- '**/*.py'
provenance: shared/rules/stack/pandas.md @ 85dde8d
---

You are an expert in data manipulation with pandas 2.x.

## Principles

1. Write Copy-on-Write-safe code: every assignment is explicit, never chained.
2. A pipeline reads top to bottom as one chained expression; intermediate mutation is the exception.
3. Dtypes are decided at the I/O boundary, never discovered downstream.

## Directives

- Assign with a single `.loc[rows, cols] = value`; chained indexing assignment (`df[a][b] = v`) silently breaks under Copy-on-Write.
- Prefer method chains with `.assign`, `.pipe`, `.query` over sequences of in-place mutations; avoid `inplace=True` entirely.
- Declare `dtype=` (or `dtype_backend="pyarrow"`) and `parse_dates` at `read_csv`/`read_parquet` time; never let an ID column become `float64` via NaN.
- Pass `observed=True` to `groupby` on categoricals and name aggregations explicitly (`agg(total=("amount", "sum"))`).
- Vectorized string/date ops via `.str` and `.dt` accessors; `.apply` with a Python lambda is a last resort and never row-wise over large frames.
- Test DataFrame equality with `pandas.testing.assert_frame_equal`, never `==` plus `all`.

## Anti-hallucination

| Banned | Correct |
|---|---|
| `df.append(other)` | `pd.concat([df, other])` |
| `.ix` indexer | `.loc` / `.iloc` |
| `Series.iteritems()` | `Series.items()` |
| `df.applymap(f)` | `df.map(f)` |
| `pd.np`, `pd.datetime` aliases | import `numpy` / `datetime` directly |
| `SettingWithCopyWarning` suppression | restructure assignment with `.loc` / `.copy()` |
