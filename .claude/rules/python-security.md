---
description: 'Python security hygiene: boundary validation, secrets handling, subprocess safety, deserialization, and dependency caution. Apply when handling user input, credentials, subprocess calls, network data, or file paths derived from external sources.'
paths:
- '**/architecture/**'
- '**/active_learning/**'
- '**/trainer/**'
provenance: shared/rules/lang/python/security.md @ cd14581
diverged: true
---

You are an expert in writing security-conscious Python.

## Principles

1. Validate at the boundary, trust inside it: every external input is hostile until parsed into a typed structure.
2. Secrets never touch the repository, the logs, or an error message.
3. The safe API is the default API; reaching for the unsafe variant requires a stated reason.

## Boundaries and input

- Parse, don't validate: convert external input into typed objects (Pydantic, dataclasses) at the entry point; downstream code receives structure, never raw payloads.
- Paths built from external input are resolved and checked against an allowed root before use (`Path.resolve()`, then `is_relative_to`).
- SQL through parameterized queries or an ORM, never string interpolation.

## Secrets

- Credentials from the environment or a secret manager, never from committed files or defaults.
- Secrets excluded from logs, exceptions, and `repr` output; redact before raising.
- `.env` files are gitignored; a committed `.env.example` documents the shape without values.

## Subprocess and deserialization

- `subprocess.run([...])` with an argument list; `shell=True` only with a fixed string and a comment stating why.
- `yaml.safe_load`, never `yaml.load`; `pickle` never on untrusted data; `json` for interchange.
- `tempfile` module for temporary files, never predictable paths in `/tmp`.

## Anti-hallucination

| Banned | Correct |
|---|---|
| `shell=True` with interpolated input | argument-list `subprocess.run` |
| `yaml.load(data)` | `yaml.safe_load(data)` |
| `pickle.loads` on external data | `json.loads` or a validated schema |
| `random` for tokens or keys | `secrets` module |
| `hashlib.md5`/`sha1` for security purposes | `sha256`+ or `hashlib.scrypt`/`bcrypt` for passwords |
| `eval`/`exec` on any external string | explicit parsing |

## Enforcement

This repo's pre-commit gate runs ruff (lint + format), black, isort, flake8, and pyupgrade at line length 120; ruff lint currently enforces only the docstring (`D`) family, so ruff's `S` (bandit) family is not active here. The directives above are therefore project conventions rather than gate-enforced rules: follow them, but do not assume the linter will catch a violation.

The deserialization guidance is load-bearing in the files this rule scopes. Model persistence rides on untrusted-by-default loaders: `joblib`/`pickle` in `architecture/model_base.py` and `active_learning/committee.py`, and `torch.load` in `architecture/lightning_model_base.py`, `architecture/chemprop.py`, and `trainer/lightning.py`. Treat any artifact loaded from outside the project's own trusted outputs as hostile input: prefer `torch.load(..., weights_only=True)` (note `trainer/lightning.py` currently loads a checkpoint with `weights_only=False`), and never unpickle a model file received from an untrusted source.
