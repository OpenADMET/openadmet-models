"""
Custom training criteria for ChemProp regression models.

This module provides loss functions that plug into ChemProp's metric system
(:class:`chemprop.nn.metrics.ChempropMetric`). They are bound to a
:class:`chemprop.nn.RegressionFFN` via its ``criterion`` argument and therefore
participate unchanged in ChemProp's per-sample weighting, per-task weighting,
and finite-target masking.
"""

from numpy.typing import ArrayLike
import torch
from chemprop.nn.metrics import ChempropMetric


def _normal_log_cdf(x: torch.Tensor) -> torch.Tensor:
    r"""
    Numerically stable log of the standard normal CDF.

    Computes :math:`\log \Phi(x)` where :math:`\Phi` is the standard normal CDF via
    :func:`torch.special.log_ndtr`, whose asymptotics keep the gradient correct deep
    in the censored tail. A naive ``log(Phi(x) + eps)`` instead floors the value and
    sends the gradient to zero as :math:`x \to -\infty`, which would silence the loss
    exactly when a left-censored compound is predicted far above its bound, the case
    the censored branch exists to penalize.

    Parameters
    ----------
    x : torch.Tensor
        Standardized argument of any shape.

    Returns
    -------
    torch.Tensor
        ``log Phi(x)``, same shape and dtype as ``x``.

    """
    return torch.special.log_ndtr(x)


class CensoredRegressionLoss(ChempropMetric):
    r"""
    Tobit (censored regression) loss for mixed exact and inequality labels.

    Adapted from the censored loss in the ``moal`` pipeline, re-expressed as a
    :class:`ChempropMetric` so it consumes ChemProp's native inequality masks
    (``lt_mask`` / ``gt_mask``) instead of a list of label records. Two corrections
    relative to that source: the exact branch carries the Gaussian one-half factor so
    the branches share one likelihood scale (see below), and :math:`\log \Phi` is
    evaluated with :func:`torch.special.log_ndtr` so the censored-tail gradient does
    not vanish. Each element's branch is chosen by its mask, with the recorded
    ``target`` acting as the censoring bound :math:`T`:

    .. math::

        L = \begin{cases}
            -\log \Phi\!\big((T - \hat{y}) / \sigma\big) & \text{lt\_mask (true value} < T) \\
            -\log \Phi\!\big((\hat{y} - T) / \sigma\big) & \text{gt\_mask (true value} > T) \\
            \tfrac{1}{2}\big((\hat{y} - T) / \sigma\big)^2 & \text{otherwise (exact)}
        \end{cases}

    A left-censored observation (``lt_mask``) contributes no penalty once the
    prediction sits well below the bound and a smoothly growing penalty as it rises
    above it, so a value recorded only as "below the assay detection limit" stops
    anchoring the fit at its nominal number. The exact branch carries the one-half
    factor of the Gaussian negative log-likelihood (the additive constant
    :math:`\log(\sigma\sqrt{2\pi})` is dropped as it has no gradient): this keeps the
    exact and censored branches on the same likelihood scale, so a censored row's
    gradient is weighted against an exact row's exactly as Tobit maximum likelihood
    prescribes. Omitting it would double the exact branch's relative weight and bias
    the fit toward the dense exact rows, blunting the very de-anchoring the censored
    branch is meant to provide.

    Per-sample weights and per-task weights are applied by
    :meth:`ChempropMetric.update` after this method returns, so the unreduced loss
    here is unweighted; ChemProp's finite-target masking is likewise handled by the
    base class.

    Parameters
    ----------
    sigma : ArrayLike
        Noise scale :math:`\sigma` in the residual's space, broadcastable to the
        prediction shape ``b x t``. A scalar applies one scale to every task. When
        targets are normalized for training, this must be the raw-unit scale divided
        by the per-task target standard deviation (the model wiring performs that
        conversion). Every entry must be strictly positive.
    task_weights : ArrayLike, default=1.0
        Per-task weights of shape ``t`` or ``1 x t``, forwarded to
        :class:`ChempropMetric`.

    Notes
    -----
    ChemProp represents an inequality with a single bound per element, so the
    interval-censored branch of the original moal loss (a primary-screen hit known
    to lie in ``[T, U]``) has no representation here and is out of scope; this port
    covers the exact, left-censored, and right-censored cases.

    """

    def __init__(self, sigma: ArrayLike, task_weights: ArrayLike = 1.0):
        """Store the per-task noise scale as a buffer after validating it."""
        super().__init__(task_weights=task_weights)
        sigma_tensor = torch.as_tensor(sigma, dtype=torch.float).view(1, -1)
        if torch.any(sigma_tensor <= 0):
            raise ValueError(
                f"sigma must be strictly positive, got {sigma_tensor.tolist()}"
            )
        self.register_buffer("sigma", sigma_tensor)

    def _calc_unreduced_loss(
        self, preds: torch.Tensor, targets: torch.Tensor, *args
    ) -> torch.Tensor:
        """Return the per-element censored loss, branching on the inequality masks."""
        # ChempropMetric.update passes (mask, weights, lt_mask, gt_mask); the masks default
        # to all-False there, so an uncensored batch (e.g. validation) takes the exact branch
        lt_mask = (
            args[2] if len(args) > 2 else torch.zeros_like(targets, dtype=torch.bool)
        )
        gt_mask = (
            args[3] if len(args) > 3 else torch.zeros_like(targets, dtype=torch.bool)
        )

        # exact branch: Gaussian negative log-likelihood (the 1/2 puts it on the same
        # likelihood scale as the -log Phi censored branches; the additive constant is dropped)
        exact = 0.5 * ((preds - targets) / self.sigma) ** 2
        # left-censored: penalize only as the prediction climbs above the lower bound T
        left = -_normal_log_cdf((targets - preds) / self.sigma)
        # right-censored: penalize only as the prediction falls below the upper bound T
        right = -_normal_log_cdf((preds - targets) / self.sigma)

        loss = torch.where(lt_mask, left, exact)
        return torch.where(gt_mask, right, loss)

    def extra_repr(self) -> str:
        """Append the per-task noise scale to the metric's repr."""
        return f"{super().extra_repr()}, sigma={self.sigma.tolist()}"
