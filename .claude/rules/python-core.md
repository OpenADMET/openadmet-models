---
description: 'Core Python conventions: typing, naming, control flow, error handling, logging, and data shapes. Apply when creating or modifying any Python source file.'
paths:
- '**/*.py'
provenance: shared/rules/lang/python/core.md @ 89ac50f
diverged: true
---

You are an expert in modern Python (3.10+, the project's supported floor) and idiomatic, production-quality code.

## Principles

1. Code is read far more often than written; optimize for the reader.
2. Types are documentation that cannot rot; annotate fully.
3. Fail loudly at the boundary, never silently in the middle.
4. The standard library wins unless a dependency earns its weight.

## Typing

- Full annotations on every function signature, including return types.
- Modern syntax: `list[str]`, `dict[str, int]`, `X | None`; never the deprecated `typing` aliases.
- `TypedDict`, `dataclass`, or Pydantic models for structured data crossing a boundary; bare dicts only for genuinely dynamic shapes.
- `Protocol` over abstract base classes when only an interface is needed; the component model here deliberately uses Pydantic `BaseModel` ABCs (`ModelBase` and friends) because it needs shared implementation and validation, not a bare interface, so extend those ABCs rather than introducing parallel Protocols.

## Structure and naming

- Guard clauses and early returns; happy path last; no `else` after a returning `if`.
- Descriptive names with auxiliary verbs for booleans: `is_active`, `has_permission`.
- Modules and packages in `lowercase_with_underscores`; one concern per module.
- Functions short enough to read as a unit; chunk comments mark phases within a function (per global code-style doctrine), and extraction is for reuse or genuinely separable concerns, never comment avoidance.

## Errors and logging

- Catch the narrowest exception that you can handle; never bare `except:` or `except Exception:` without re-raise.
- Custom exception types for domain errors; `raise ... from err` to preserve cause chains.
- `logging` over `print` in library and application code; loggers named `__name__`.
- Validate at system boundaries (user input, external APIs); trust internal callers.

## Anti-hallucination

| Banned | Correct |
|---|---|
| `typing.List`, `typing.Dict`, `typing.Optional[X]` | `list`, `dict`, `X \| None` |
| `os.path.join`, `os.path.exists` | `pathlib.Path` operations |
| `%`-formatting and `.format()` in new code | f-strings (except logging, which defers: `logger.info("x=%s", x)`) |
| Mutable default arguments (`def f(x=[])`) | `None` sentinel plus guard |
| `datetime.utcnow()` | `datetime.now(tz=timezone.utc)` |
| `asyncio.get_event_loop()` in new code | `asyncio.run()` / `asyncio.get_running_loop()` |

## Enforcement

This repo's pre-commit gate runs ruff (lint + format), black, isort, flake8, and pyupgrade, with a line length of 120; ruff lint currently enforces only the docstring (`D`) family. The typing, pathlib, and deprecated-alias directives above are therefore project conventions rather than gate-enforced rules here: follow them, but do not assume the linter will catch a violation. pyupgrade keeps syntax modern. Do not fight the formatter; fix the code.
