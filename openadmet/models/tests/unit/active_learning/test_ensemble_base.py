import pytest

from openadmet.models.active_learning.committee import CommitteeRegressor
from openadmet.models.active_learning.ensemble_base import get_ensemble_class


def test_get_ensemble_class_success():
    """Verify that get_ensemble_class returns the correct class for a valid ensemble type."""
    assert get_ensemble_class("CommitteeRegressor") is CommitteeRegressor


def test_get_ensemble_class_raises_for_invalid_type():
    """Ensure get_ensemble_class raises ValueError when requested for a non-existent ensemble type."""
    with pytest.raises(ValueError, match="Ensemble type not-real not found"):
        get_ensemble_class("not-real")
