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


class NoiseThresholdedMSE(ChempropMetric):
    r"""
    Epsilon-insensitive squared error with a per-task noise threshold.

    Residuals smaller in magnitude than the per-task threshold ``delta``
    contribute no loss; residuals larger than ``delta`` contribute the squared
    excess beyond it:

    .. math::

        L = \big(\max(|\hat{y} - y| - \delta,\ 0)\big)^2

    The loss and its first derivative are both zero at :math:`|r| = \delta`, so
    there is no gradient discontinuity at the threshold boundary. At
    ``delta = 0`` the loss is exactly mean squared error, which makes the
    threshold a strict generalization of the default ChemProp criterion.

    The threshold is a fixed measurement-noise floor, not a learned parameter.
    It must be supplied in the same space as the residuals seen during training.
    ChemProp's ``UnscaleTransform`` is a no-op in training mode, so training
    residuals are in normalized target space; a threshold specified in raw
    target units must be divided by the per-task target standard deviation
    before being passed here.

    Parameters
    ----------
    delta : ArrayLike
        Per-task noise threshold in normalized target space, broadcastable to
        the prediction shape ``b x t``. A scalar applies the same threshold to
        every task. Every entry must be non-negative.
    task_weights : ArrayLike, default=1.0
        Per-task weights of shape ``t`` or ``1 x t``, forwarded to
        :class:`ChempropMetric`.

    """

    def __init__(self, delta: ArrayLike, task_weights: ArrayLike = 1.0):
        """Store the per-task threshold as a buffer after validating it."""
        super().__init__(task_weights=task_weights)
        delta_tensor = torch.as_tensor(delta, dtype=torch.float).view(1, -1)
        if torch.any(delta_tensor < 0):
            raise ValueError(f"delta must be non-negative, got {delta_tensor.tolist()}")
        self.register_buffer("delta", delta_tensor)

    def _calc_unreduced_loss(
        self, preds: torch.Tensor, targets: torch.Tensor, *args
    ) -> torch.Tensor:
        """Return the per-element squared excess of the residual beyond delta."""
        excess = (preds - targets).abs() - self.delta
        excess = excess.clamp_min(0.0)
        return excess * excess

    def extra_repr(self) -> str:
        """Append the per-task threshold to the metric's repr."""
        return f"{super().extra_repr()}, delta={self.delta.tolist()}"
