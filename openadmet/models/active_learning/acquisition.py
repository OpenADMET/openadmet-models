from scipy.stats import norm


def max_uncertainty_reduction_query(regressor, X, **kwargs):
    r"""Maximum uncertainty reduction acquisition function. Refines an already well-performing model.

    .. math::

        x_{\text{next}} = \arg\max_x \sigma(x)

    Where:
        - \\( \sigma(x) \\): Predictive standard deviation at \\( x \\)

    Parameters
    ----------
    regressor : estimator object
        Regressor with `predict(X, return_std=True)`.
    X : np.array
        Pool of examples.
    kwargs : keyword arguments
        Additional keyword arguments to pass to the regressor's `predict` method.

    Returns
    -------
    np.array
        Uncertainty values for each instance in `X`.

    References
    ----------
    .. [1] Cohn, D., Ghahramani, Z., & Jordan, M. I. (1996). Active Learning with Statistical Models.
    Journal of Artificial Intelligence Research, 4, 129–145.

    """
    # Predict on available points
    _, std = regressor.predict(X, return_std=True, **kwargs)

    return std


def exploitation_query(regressor, X, **kwargs):
    r"""Returns the instances within `X` with highest predicted values.

    Parameters
    ----------
    regressor : estimator object
        Regressor with `predict(X, return_std=True)`.
    X : np.array
        Pool of examples.
    kwargs : keyword arguments
        Additional keyword arguments to pass to the regressor's `predict` method.

    Returns
    -------
    np.array
        Predicted values for each instance in `X`.

    """
    # Predict on available points
    preds = regressor.predict(X, return_std=False, **kwargs)

    return preds


def probability_improvement_query(regressor, X, best_y, xi=0.01, **kwargs):
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
    regressor : estimator object
        Regressor with `predict(X, return_std=True)`.
    X : np.array
        Pool of examples.
    best_y : float
        Best observed value so far.
    xi : float
        Exploration-exploitation tradeoff parameter.
    kwargs : keyword arguments
        Additional keyword arguments to pass to the regressor's `predict` method.

    Returns
    -------
    np.array
        Probability improvement values for each instance in `X`.

    References
    ----------
    .. [1] Kushner, H. J. (1964). A new method of locating the maximum point of an arbitrary multipeak curve in the
    presence of noise. Journal of Basic Engineering, 86(1), 97–106.

    """
    mean, std = regressor.predict(X, return_std=True, **kwargs)
    std = std.clip(min=1e-9)  # Avoid division by zero

    PI = norm.cdf((mean - best_y - xi) / std)

    return PI


def expected_improvement_query(regressor, X, best_y, xi=0.01, **kwargs):
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
    regressor : estimator object
        Regressor with `predict(X, return_std=True)`.
    X : np.array
        Pool of examples.
    best_y : float
        Best observed value so far.
    xi : float
        Exploration-exploitation tradeoff parameter.
    kwargs : keyword arguments
        Additional keyword arguments to pass to the regressor's `predict` method.

    Returns
    -------
    np.array
        Expected improvement values for each instance in `X`.

    References
    ----------
    .. [1] Jones, D. R., Schonlau, M., & Welch, W. J. (1998). Efficient global optimization of expensive black-box
    functions. Journal of Global Optimization, 13(4), 455–492.

    """
    mean, std = regressor.predict(X, return_std=True, **kwargs)
    std = std.clip(min=1e-9)  # Avoid division by zero

    improvement = mean - best_y - xi
    Z = improvement / std
    EI = improvement * norm.cdf(Z) + std * norm.pdf(Z)

    return EI


def upper_confidence_bound_query(regressor, X, beta=2.0, **kwargs):
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
    regressor : estimator object
        Regressor with `predict(X, return_std=True)`.
    X : np.array
        Pool of examples.
    beta : float
        Tradeoff parameter (higher = more exploration).
    kwargs : keyword arguments
        Additional keyword arguments to pass to the regressor's `predict` method.

    Returns
    -------
    np.array
        Upper confidence bound values for each instance in `X`.

    References
    ----------
    .. [1] Srinivas, N., Krause, A., Kakade, S. M., & Seeger, M. (2010). Gaussian Process Optimization in the Bandit
    Setting: No Regret and Experimental Design. ICML.

    """
    mean, std = regressor.predict(X, return_std=True, **kwargs)

    ucb = mean + beta * std  # Exploration-exploitation balance

    return ucb


_QUERY_STRATEGIES = {
    "max-uncertainty-reduction": max_uncertainty_reduction_query,
    "exploitation": exploitation_query,
    "upper-confidence-bound": upper_confidence_bound_query,  # `beta` should be configurable
    # "random": random_query,
}
