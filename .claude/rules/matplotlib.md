---
description: 'Matplotlib conventions: the object-oriented Axes API over pyplot state, constrained layout, colormap access, figure lifecycle, and publication output settings. Apply when writing or reviewing plotting code, figure generation scripts, or matplotlib styling.'
paths:
- '**/*.py'
provenance: shared/rules/stack/matplotlib.md @ 85dde8d
---

You are an expert in matplotlib's object-oriented API.

## Principles

1. Figures and Axes are objects passed around explicitly; implicit pyplot state belongs only in throwaway interactive work.
2. A figure-producing function takes or returns an `Axes`, so callers compose it.
3. Generated figures are build products: produced by scripts, never hand-tuned afterward.

## Directives

- Start with `fig, ax = plt.subplots(layout="constrained")`; call methods on `ax` (`ax.plot`, `ax.set_xlabel`), never the `plt.*` state-machine equivalents in library code.
- Batch label/title/limit settings with `ax.set(xlabel=..., ylabel=..., title=...)`.
- Save with `fig.savefig(path, dpi=300, bbox_inches="tight")` and close with `plt.close(fig)` in loops; leaked figures exhaust memory in batch jobs.
- Resolve colormaps via `matplotlib.colormaps["viridis"]`; pick perceptually uniform maps for scalar data, never `jet`.
- Set style at the script level (`plt.style.use`, `rcParams` context managers), never by repeating per-plot cosmetic kwargs.
- Date axes use real datetime values plus locators/formatters, never pre-formatted strings.

## Anti-hallucination

| Banned | Correct |
|---|---|
| `plt.cm.get_cmap("x")` / `matplotlib.cm.get_cmap` | `matplotlib.colormaps["x"]` |
| `fig.tight_layout()` in new code | `plt.subplots(layout="constrained")` |
| `plt.plot` / `plt.title` inside library functions | explicit `ax` methods |
| `ax.set_xticklabels` without `set_xticks` | `ax.set_xticks(ticks, labels=...)` |
| `pylab` imports | `matplotlib.pyplot` + `numpy` |
