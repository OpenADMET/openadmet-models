import numpy as np
from numpy.testing import assert_allclose
from scipy.stats import norm

from openadmet.models.active_learning.acquisition import (
    _ACQUISITION_FUNCTIONS,
    expected_improvement,
    exploitation,
    max_uncertainty_reduction,
    probability_improvement,
    upper_confidence_bound,
)


def test_basic_acquisition_functions_passthrough():
    """
    Validate that basic acquisition functions return expected values based on mean and standard deviation.

    This verifies that:
    - Max uncertainty reduction returns standard deviation (uncertainty).
    - Exploitation returns the mean prediction.
    - UCB correctly combines mean and uncertainty with the beta parameter.
    """
    mean = np.array([[1.0], [2.0]])
    std = np.array([[0.1], [0.2]])
    assert_allclose(max_uncertainty_reduction(mean, std), std)
    assert_allclose(exploitation(mean, std), mean)
    assert_allclose(upper_confidence_bound(mean, std, beta=3.0), mean + 3.0 * std)


def test_probability_improvement_matches_formula():
    """
    Verify Probability of Improvement (PI) calculation against the explicit mathematical formula.

    This ensures that the implementation correctly computes the cumulative distribution function (CDF)
    of the improvement over the best observed value, accounting for the exploration parameter xi.
    """
    mean = np.array([[1.0], [2.0]])
    std = np.array([[0.5], [1e-12]])
    best_y = 1.2
    xi = 0.1
    expected = norm.cdf((mean - best_y - xi) / std.clip(min=1e-9))
    assert_allclose(probability_improvement(mean, std, best_y=best_y, xi=xi), expected)


def test_expected_improvement_matches_formula():
    """
    Verify Expected Improvement (EI) calculation against the explicit mathematical formula.

    This ensures that EI correctly balances exploration and exploitation using both the CDF and PDF
    of the normal distribution, which is critical for efficient active learning query strategies.
    """
    mean = np.array([[1.0], [1.5]])
    std = np.array([[0.2], [1e-12]])
    best_y = 0.8
    xi = 0.01
    std_clip = std.clip(min=1e-9)
    improvement = mean - best_y - xi
    z_score = improvement / std_clip
    expected = improvement * norm.cdf(z_score) + std_clip * norm.pdf(z_score)
    assert_allclose(expected_improvement(mean, std, best_y=best_y, xi=xi), expected)


def test_acquisition_aliases_map_to_same_function():
    """Ensure that shorthand aliases for acquisition functions map to the correct implementation functions."""
    assert (
        _ACQUISITION_FUNCTIONS["ur"]
        is _ACQUISITION_FUNCTIONS["max-uncertainty-reduction"]
    )
    assert _ACQUISITION_FUNCTIONS["exp"] is _ACQUISITION_FUNCTIONS["exploitation"]
    assert (
        _ACQUISITION_FUNCTIONS["ucb"]
        is _ACQUISITION_FUNCTIONS["upper-confidence-bound"]
    )
    assert (
        _ACQUISITION_FUNCTIONS["ei"] is _ACQUISITION_FUNCTIONS["expected-improvement"]
    )
    assert (
        _ACQUISITION_FUNCTIONS["pi"]
        is _ACQUISITION_FUNCTIONS["probability-improvement"]
    )
