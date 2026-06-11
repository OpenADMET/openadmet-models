---
description: 'Pytest conventions for Python test suites: structure, fixtures, parametrization, naming, and coverage policy. Apply when creating or modifying tests, conftest.py, or pytest configuration.'
paths:
- '**/test_*.py'
- '**/*_test.py'
- '**/tests/**'
- '**/conftest.py'
provenance: shared/rules/lang/python/testing.md @ 89ac50f
---

You are an expert in Python test architecture with pytest.

## Principles

1. A test exists to catch a regression; a test that cannot fail on a real bug gets deleted.
2. Tests document behavior: the name states the behavior, the body demonstrates it.
3. Independence is absolute: any subset of tests passes in any order.
4. Test code is production code: the same style and review standards apply.

## Structure

- One behavior per test, named for the behavior: `test_rejects_expired_token`, never `test_token_2`.
- Arrange, act, assert, in that order, separated by blank lines; no logic between act and assert.
- Group tests in classes only when they share fixtures, never for namespacing alone.
- Test files parallel the source layout: `pkg/auth.py` is tested by `tests/test_auth.py`.

## Fixtures and parametrization

- Fixtures over setup methods or module-level state; narrowest scope that works (function scope unless proven otherwise).
- `tmp_path` and `monkeypatch` over manual temp dirs and attribute patching.
- `pytest.mark.parametrize` over loops in test bodies; `ids=` for non-obvious cases.
- `conftest.py` holds only fixtures used by more than one file.

## Test doubles

- Prefer, in strict order: (1) the real component, configured cheaply (tiny inputs, `tmp_path`, in-memory backends, baseline implementations like a `DummyRegressor`); (2) a fake the repo already provides; (3) a mock, only at true process boundaries (network, clock, subprocess, paid APIs), always with `autospec=True`.
- Before introducing any mock or stub, search the repo (`conftest.py`, test utilities, fixtures, existing tests) for a lightweight real implementation or existing fake; reuse it. Mocking code the repo owns is a finding, not a convenience.
- When a mock is justified, patch where the name is looked up, never where it is defined.

## Coverage policy

- Cover the contract, the edge cases, and every fixed bug; a regression test lands with the fix and is shown to fail before it.
- Do not test framework guarantees, third-party libraries, or trivial accessors.
- Property-based tests (hypothesis) for pure functions with rich input spaces; example-based otherwise.
- The deletion test for every test: name a realistic code change that would make it fail. If you cannot, the test is tautological; delete it.
- Never assert that a mock returned what you configured it to return, that a function was called with the arguments you just passed, or that a result equals an expectation computed by mirroring the implementation's own logic. Expected values are hand-computed constants or known-good references.

## Anti-hallucination

| Banned | Correct |
|---|---|
| `unittest.TestCase` subclasses in new code | plain pytest functions |
| `self.assertEqual` and friends | bare `assert` (pytest introspection) |
| `@pytest.yield_fixture` | `@pytest.fixture` (yield is native) |
| `tmpdir` fixture (py.path) | `tmp_path` (pathlib.Path) |
| `pytest.warns(None)` | `pytest.warns(SpecificWarning)` |
| mutating `os.environ` directly | `monkeypatch.setenv` / `delenv` |
| `MagicMock()` without `spec`/`autospec` | `create_autospec` / `autospec=True` |
| asserting `mock.return_value` round-trips | exercise real behavior via a fake or cheap real component |
| expected values derived by re-running the implementation's logic | hand-computed constants or known-good fixtures |

## Enforcement

Fixture style, assertion style, and parametrize formatting are enforced by ruff's `PT` rule family where the repo is seeded with it. Do not fight the linter; fix the test.
