"""Fixtures that load the standalone bench scripts by file path."""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

BENCH_DIR = Path(__file__).resolve().parents[1]


def _load(label: str, filename: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(label, BENCH_DIR / filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[label] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="session")
def run() -> ModuleType:
    """Return the bench/run.py module."""
    return _load("bench_run", "run.py")


@pytest.fixture(scope="session")
def analyze() -> ModuleType:
    """Return the bench/analyze.py module."""
    return _load("bench_analyze", "analyze.py")
