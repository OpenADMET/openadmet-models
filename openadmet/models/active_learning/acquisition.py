import numpy as np
from scipy.stats import norm


def max_uncertainty_reduction_query(regressor, X, n_instances=1):
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

    Returns
    -------
    index : int
        Query index.
    X_i : np.array
        Query instance.

    References
    ----------
    .. [1] Cohn, D., Ghahramani, Z., & Jordan, M. I. (1996). Active Learning with Statistical Models.
    Journal of Artificial Intelligence Research, 4, 129–145.

    """
    # Predict on available points
    _, std = regressor.predict(X, return_std=True)

    # Take largest N standard devs
    query_idx = np.argsort(std)[-n_instances:]

    return query_idx, X[query_idx]


def exploitation_query(regressor, X, n_instances=1):
    r"""Returns the instances within `X` with highest predicted values.

    Parameters
    ----------
    regressor : estimator object
        Regressor with `predict(X, return_std=True)`.
    X : np.array
        Pool of examples.
    n_instances : int
        Number of instances.

    Returns
    -------
    index : int
        Query index.
    X_i : np.array
        Query instance.

    """
    # Predict on available points
    preds, _ = regressor.predict(X, return_std=True)

    # Take largest N predictions
    query_idx = np.argsort(preds)[-n_instances:]

    return query_idx, X[query_idx]


def probability_improvement_query(regressor, X, best_y, n_instances=1, xi=0.01):
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
    n_instances : int
        Number of instances to select.
    xi : float
        Exploration-exploitation tradeoff parameter.

    Returns
    -------
    query_idx : int
        Query index.
    X_i : np.array
        Query instance.

    References
    ----------
    .. [1] Kushner, H. J. (1964). A new method of locating the maximum point of an arbitrary multipeak curve in the
    presence of noise. Journal of Basic Engineering, 86(1), 97–106.

    """
    mean, std = regressor.predict(X, return_std=True)
    std = std.clip(min=1e-9)  # Avoid division by zero

    PI = norm.cdf((mean - best_y - xi) / std)

    query_idx = np.argsort(PI)[-n_instances:]

    return query_idx, X[query_idx]


def expected_improvement_query(regressor, X, best_y, n_instances=1, xi=0.01):
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
    n_instances : int
        Number of instances to select.
    xi : float
        Exploration-exploitation tradeoff parameter.

    Returns
    -------
    query_idx : int
        Query index.
    X_i : np.array
        Query instance.

    References
    ----------
    .. [1] Jones, D. R., Schonlau, M., & Welch, W. J. (1998). Efficient global optimization of expensive black-box
    functions. Journal of Global Optimization, 13(4), 455–492.

    """
    mean, std = regressor.predict(X, return_std=True)
    std = std.clip(min=1e-9)  # Avoid division by zero

    improvement = mean - best_y - xi
    Z = improvement / std
    EI = improvement * norm.cdf(Z) + std * norm.pdf(Z)

    query_idx = np.argsort(EI)[-n_instances:]

    return query_idx, X[query_idx]


def upper_confidence_bound_query(regressor, X, n_instances=1, beta=2.0):
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
    n_instances : int
        Number of instances to select.
    beta : float
        Tradeoff parameter (higher = more exploration).

    Returns
    -------
    query_idx : int
        Query index.
    X_i : np.array
        Query instance.

    References
    ----------
    .. [1] Srinivas, N., Krause, A., Kakade, S. M., & Seeger, M. (2010). Gaussian Process Optimization in the Bandit
    Setting: No Regret and Experimental Design. ICML.

    """
    mean, std = regressor.predict(X, return_std=True)

    ucb = mean + beta * std  # Exploration-exploitation balance

    query_idx = np.argsort(ucb)[-n_instances:]

    return query_idx, X[query_idx]


def random_query(regressor, X, n_instances=1):
    r"""
    Random acquisition function. Randomly selects points from the pool. Useful as null model.

    Parameters
    ----------
    regressor : estimator object
        Ignored.
    X : np.array
        Pool of examples.
    n_instances : int
        Number of instances to select.

    Returns
    -------
    query_idx : int
        Query index.ex
    X_i : np.array
        Query instance.

    """
    query_idx = np.random.choice(X.shape[0], n_instances, replace=False)

    return query_idx, X[query_idx]
