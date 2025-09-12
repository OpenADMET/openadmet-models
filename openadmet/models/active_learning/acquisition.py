from scipy.stats import norm


def max_uncertainty_reduction(mean, std, **kwargs):
    r"""Maximum uncertainty reduction acquisition function. Refines an already well-performing model.

    .. math::

        x_{\text{next}} = \arg\max_x \sigma(x)

    Where:
        - \\( \sigma(x) \\): Predictive standard deviation at \\( x \\)

    Parameters
    ----------
    mean : np.array
        Predicted mean values, unused.
    std : np.array
        Predicted standard deviation values.
    kwargs : keyword arguments
        Additional keyword arguments.

    Returns
    -------
    np.array
        Uncertainty values for each instance in `X`.

    References
    ----------
    .. [1] Cohn, D., Ghahramani, Z., & Jordan, M. I. (1996). Active Learning with Statistical Models.
    Journal of Artificial Intelligence Research, 4, 129–145.

    """

    return std


def exploitation(mean, std, **kwargs):
    r"""Returns the instances within `X` with highest predicted values.

    Parameters
    ----------
    mean : np.array
        Predicted mean values.
    std : np.array
        Predicted standard deviation values, unused.
    kwargs : keyword arguments
        Additional keyword arguments.

    Returns
    -------
    np.array
        Predicted values for each instance in `X`.

    """

    return mean


def probability_improvement(mean, std, best_y=0, xi=0.01, **kwargs):
    r"""
    Probability Improvement (PI) acquisition function. Balances exploration and exploitation.

    .. math::

        PI(x) = \Phi(\frac{\mu(x) - f^* - \xi}{\sigma(x)})

    Where:
        - \\( \mu(x) \\): Predictive mean at \\( x \\)
        - \\( \sigma(x) \\): Predictive standard deviation at \\( x \\)
        - \\( f^* \\): Best observed value so far
        - \\( \xi \\): Small positive number to encourage exploration
        - \\( \Phi(Z) \\): CDF of standard normal distribution

    Parameters
    ----------
    mean : np.array
        Predicted mean values.
    std : np.array
        Predicted standard deviation values.
    best_y : float
        Best observed value so far.
    xi : float
        Exploration-exploitation tradeoff parameter.
    kwargs : keyword arguments
        Additional keyword arguments.

    Returns
    -------
    np.array
        Probability improvement values for each instance in `X`.

    References
    ----------
    .. [1] Kushner, H. J. (1964). A new method of locating the maximum point of an arbitrary multipeak curve in the
    presence of noise. Journal of Basic Engineering, 86(1), 97–106.

    """

    std = std.clip(min=1e-9)  # Avoid division by zero

    PI = norm.cdf((mean - best_y - xi) / std)

    return PI


def expected_improvement(mean, std, best_y=0, xi=0.01, **kwargs):
    r"""
    Expected Improvement (EI) acquisition function. Balances exploration and exploitation.

    .. math::

        EI(x) = (\mu(x) - f^* - \xi) \cdot \Phi(Z) + \sigma(x) \cdot \phi(Z)

        Z = \frac{\mu(x) - f^* - \xi}{\sigma(x)}

    Where:
        - \\( \mu(x) \\): Predictive mean at \\( x \\)
        - \\( \sigma(x) \\): Predictive standard deviation at \\( x \\)
        - \\( f^* \\): Best observed value so far
        - \\( \xi \\): Small positive number to encourage exploration
        - \\( \Phi(Z) \\): CDF of standard normal distribution
        - \\( \phi(Z) \\): PDF of standard normal distribution

    Parameters
    ----------
    mean : np.array
        Predicted mean values.
    std : np.array
        Predicted standard deviation values.
    best_y : float
        Best observed value so far.
    xi : float
        Exploration-exploitation tradeoff parameter.
    kwargs : keyword arguments
        Additional keyword arguments.

    Returns
    -------
    np.array
        Expected improvement values for each instance in `X`.

    References
    ----------
    .. [1] Jones, D. R., Schonlau, M., & Welch, W. J. (1998). Efficient global optimization of expensive black-box
    functions. Journal of Global Optimization, 13(4), 455–492.

    """

    std = std.clip(min=1e-9)  # Avoid division by zero

    improvement = mean - best_y - xi
    Z = improvement / std
    EI = improvement * norm.cdf(Z) + std * norm.pdf(Z)

    return EI


def upper_confidence_bound(mean, std, beta=2.0, **kwargs):
    r"""
    Upper Confidence Bound (UCB) acquisition function. Ensures exploration while still considering high predictions.

    .. math::

        UCB(x) = \mu(x) + \beta \cdot \sigma(x)

    Where:
        - \\( \mu(x) \\): Predictive mean at \\( x \\)
        - \\( \sigma(x) \\): Predictive standard deviation at \\( x \\)
        - \\( \beta \\): Trade-off parameter (higher \\( \beta \\) favors exploration)

    Parameters
    ----------
    mean : np.array
        Predicted mean values.
    std : np.array
        Predicted standard deviation values, unused.
    beta : float
        Tradeoff parameter (higher = more exploration).
    kwargs : keyword arguments
        Additional keyword arguments.

    Returns
    -------
    np.array
        Upper confidence bound values for each instance in `X`.

    References
    ----------
    .. [1] Srinivas, N., Krause, A., Kakade, S. M., & Seeger, M. (2010). Gaussian Process Optimization in the Bandit
    Setting: No Regret and Experimental Design. ICML.

    """

    ucb = mean + beta * std

    return ucb


_ACQUISITION_FUNCTIONS = {
    "max-uncertainty-reduction": max_uncertainty_reduction,
    "exploitation": exploitation,
    "upper-confidence-bound": upper_confidence_bound,
    "expected-improvement": expected_improvement,
    "probability-improvement": probability_improvement,
    "ur": max_uncertainty_reduction,
    "exp": exploitation,
    "ucb": upper_confidence_bound,
    "ei": expected_improvement,
    "pi": probability_improvement,
}
