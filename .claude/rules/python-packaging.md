---
description: 'Python project layout and dependency management: environment managers (uv default; pixi or conda where the repo declares them), pyproject.toml as single source, flat layout, lockfile policy, and the ruff/pyright standing gate. Apply when creating or modifying pyproject.toml, lockfiles, environment.yml, requirements files, or project scaffolding.'
paths:
- '**/pyproject.toml'
- '**/uv.lock'
- '**/requirements*.txt'
- '**/setup.py'
- '**/setup.cfg'
- '**/environment.yml'
- '**/pixi.toml'
provenance: shared/rules/lang/python/packaging.md @ 695d847
---

You are an expert in modern Python packaging and project structure.

## Principles

1. One file owns project metadata: `pyproject.toml`. Anything duplicating it is drift waiting to happen.
2. Environments are disposable; lockfiles make them reproducible.
3. The lint and type gate is part of the project, never a personal preference.

## Environment and dependencies

- Default manager for new projects: uv (`uv venv` to create, `uv add` / `uv remove` to manage, `uv run` to execute, `uv sync` to reproduce). A repo that declares another manager (pixi, conda) in its `AGENTS.md` follows that declaration; the discipline below applies under any manager.
- Dependencies declared in one manifest (`pyproject.toml` by default; `pixi.toml` or `environment.yml` where declared) with sensible lower bounds; exact pins live in a committed lockfile (`uv.lock`, `pixi.lock`, or conda-lock output).
- Dev tooling (pytest, ruff, pyright, pre-commit) in a dev dependency group, never mixed into runtime dependencies.
- Never install into a project environment ad hoc (`pip install`, `conda install`); declare in the manifest first, then sync.

## Layout

- Flat layout: the importable package lives at `<package>/` in the repo root beside `pyproject.toml`, with tests in `tests/` at the root, never inside the package.
- The package is installed editable into the environment (`uv sync`); imports resolve through the environment, never by relying on the current working directory.
- `setup.py` and `setup.cfg` do not appear in new projects; existing ones migrate to `pyproject.toml` when touched.
- Entry points declared in `[project.scripts]`, never as loose top-level scripts.

## Standing gate

- ruff (lint + format) and pyright configured in `pyproject.toml`, run via pre-commit and CI.
- Gate configuration changes are reviewed like code: loosening a rule needs a reason in the commit message.

## Anti-hallucination

| Banned | Correct |
|---|---|
| `requirements.txt` as the source of truth | the repo's declared manifest + committed lockfile (default `pyproject.toml` + `uv.lock`) |
| `python -m venv` + `pip` in new projects | `uv venv` + `uv add` (or the repo's declared manager) |
| ad-hoc `pip install` / `conda install` into a shared env | declare in the manifest, then sync |
| `setup.py develop` / `pip install -e .` | `uv sync` (or `uv pip install -e .` only when an editable install is genuinely required) |
| `[tool.poetry]` sections in new projects | PEP 621 `[project]` table |
