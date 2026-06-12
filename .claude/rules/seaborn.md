---
description: 'Seaborn 0.13+ conventions: tidy long-form data, axes-level versus figure-level function choice, and removed legacy distribution APIs. Apply when writing or reviewing statistical visualization code with seaborn.'
paths:
- '**/*.py'
provenance: shared/rules/stack/seaborn.md @ 85dde8d
---

You are an expert in statistical visualization with seaborn; the matplotlib rule governs the Axes-level substrate.

## Principles

1. Seaborn consumes tidy long-form data; reshape the frame before reaching for plotting workarounds.
2. Axes-level functions compose into matplotlib figures; figure-level functions own their figure. Choose deliberately, never by habit.

## Directives

- Pass `data=` with column names (`x="dose"`, `hue="arm"`), never positional arrays extracted from the frame.
- Use axes-level functions (`histplot`, `boxplot`, `scatterplot`) with an explicit `ax=` when the plot lives inside a larger figure; use figure-level functions (`displot`, `catplot`, `relplot`) only when their faceting is wanted, and style them through their returned `FacetGrid`.
- Melt wide data with `pd.melt` (or polars `unpivot`) before plotting rather than looping plot calls per column.
- Set themes once with `sns.set_theme(...)` at script level; pass `palette` by name and pin `hue_order`/`order` where category order carries meaning.

## Anti-hallucination

| Banned | Correct |
|---|---|
| `sns.distplot` | `sns.histplot` / `sns.kdeplot` / `sns.displot` |
| `ax=` passed to figure-level functions (`displot`, `catplot`) | axes-level equivalent, or accept the managed figure |
| `sns.plt` alias | `import matplotlib.pyplot as plt` |
| `factorplot` | `catplot` |
