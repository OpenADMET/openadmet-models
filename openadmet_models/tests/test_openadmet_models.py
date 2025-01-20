"""
Unit and regression test for the openadmet_models package.
"""

# Import package, test suite, and other packages as needed
import sys

import pytest

import openadmet_models


def test_openadmet_models_imported():
    """Sample test, will always pass so long as import statement worked."""
    assert "openadmet_models" in sys.modules
