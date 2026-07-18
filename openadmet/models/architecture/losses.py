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


class QuantileLoss(ChempropMetric):
    r"""
    Pinball (quantile) loss for asymmetric penalization of the regression error.

    For a target quantile :math:`\tau \in (0, 1)` and error
    :math:`e = y - \hat{y}`, the per-element loss is

    .. math::

        L = \max(\tau\, e,\ (\tau - 1)\, e)
          = \begin{cases} \tau\, e & e \ge 0 \;(\text{under-prediction}) \\
                          (1 - \tau)\, |e| & e < 0 \;(\text{over-prediction})
            \end{cases}

    At :math:`\tau = 0.5` this is half the mean absolute error and is symmetric.
    For :math:`\tau > 0.5` under-prediction (predicting below the target) is
    penalized more than over-prediction, which pushes predictions upward; this is
    the intended behavior for confirmed-potent compounds whose magnitude the model
    otherwise compresses toward the mean. The minimizer of the expected loss is the
    conditional :math:`\tau`-quantile of the target.

    The loss is scale-free in the quantile sense: :math:`\tau` is a probability,
    not a target-unit quantity, so unlike a raw-unit threshold it needs no
    conversion through the training-target scaler. It operates on normalized
    training residuals exactly as written.

    Parameters
    ----------
    tau : float
        Target quantile, strictly between 0 and 1. Values above 0.5 penalize
        under-prediction more heavily.
    task_weights : ArrayLike, default=1.0
        Per-task weights of shape ``t`` or ``1 x t``, forwarded to
        :class:`ChempropMetric`.

    """

    def __init__(self, tau: float, task_weights: ArrayLike = 1.0):
        """Store the quantile after validating it lies strictly in (0, 1)."""
        super().__init__(task_weights=task_weights)
        if not 0.0 < float(tau) < 1.0:
            raise ValueError(f"tau must be strictly between 0 and 1, got {tau}")
        self.tau = float(tau)

    def _calc_unreduced_loss(
        self, preds: torch.Tensor, targets: torch.Tensor, *args
    ) -> torch.Tensor:
        """Return the per-element pinball loss for the stored quantile."""
        error = targets - preds
        return torch.maximum(self.tau * error, (self.tau - 1.0) * error)

    def extra_repr(self) -> str:
        """Append the quantile to the metric's repr."""
        return f"{super().extra_repr()}, tau={self.tau}"
