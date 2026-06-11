---
description: 'Python documentation conventions: NumPy-style docstrings, module documentation, executable examples, and keeping prose docs in sync with code. Apply when writing or modifying docstrings, README files, or API documentation in Python projects.'
paths:
- '**/*.py'
- '**/README.md'
- '**/docs/**'
provenance: shared/rules/lang/python/docs.md @ 89ac50f
---

You are an expert in Python API documentation using the NumPy docstring standard.

## Principles

1. Documentation that disagrees with the code is worse than none; docs change in the same commit as the code they describe.
2. An example that does not run is a bug.
3. Document the contract (inputs, outputs, failure modes), never the implementation.

## Docstrings (NumPy style)

- Every public module, class, function, and method carries a docstring; private helpers only when behavior is non-obvious.
- "Public" means the module's external API, not merely the absence of a leading underscore. Module-internal helpers are underscore-prefixed, which both marks them private and exempts them from the docstring requirement. If the only honest docstring for a function would restate its name, that is a signal the function is internal (prefix it `_`), not a reason to write a filler docstring.
- One-line summary in imperative mood on the first line, ending with a period; blank line before any further content.
- Sections in canonical order, each underlined with hyphens:
  `Parameters`, `Returns`, `Yields`, `Raises`, `See Also`, `Notes`, `References`, `Examples`.
- `Parameters`: `name : type` on one line, description indented beneath; optional parameters marked `, optional` with the default stated in the description.
- `Returns`: type on its own line, description indented beneath; named returns (`name : type`) only when returning multiple values.
- `Raises`: only exceptions a caller can reasonably encounter and act on.
- Types in docstrings stay loose and readable (`array-like`, `path-like`); the signature annotation is the precise contract.

## Examples

- `Examples` sections use doctest format (`>>>`) and must execute as written.
- Prefer one realistic example over several trivial ones; show the common path, then one edge case if it changes usage.

## Prose docs

- README documents what the project is, how to install, and the shortest path to first success; API detail belongs in docstrings or generated docs.
- Changelog entries land with the change, written for users (what changed, why it matters), never as commit-log narration.

## Enforcement

Docstring presence and NumPy-format compliance are enforced by ruff's `D` rule family with `convention = "numpy"` where the repo is seeded with it. Do not fight the linter; fix the docstring.
