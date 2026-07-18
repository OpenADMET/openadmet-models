import numpy as np
import pytest
import torch
from sklearn.preprocessing import StandardScaler

from openadmet.models.architecture.chemprop import ChemPropModel
from openadmet.models.architecture.losses import QuantileLoss


def _make_scaler(scale: list[float]) -> StandardScaler:
    scaler = StandardScaler()
    scaler.scale_ = np.array(scale)
    scaler.mean_ = np.zeros(len(scale))
    return scaler


def test_quantile_half_is_symmetric_half_absolute_error():
    """At tau=0.5 over- and under-prediction of equal size cost the same: 0.5*|r|."""
    over = QuantileLoss(tau=0.5)._calc_unreduced_loss(
        torch.tensor([[1.0]]), torch.tensor([[0.0]])
    )
    under = QuantileLoss(tau=0.5)._calc_unreduced_loss(
        torch.tensor([[-1.0]]), torch.tensor([[0.0]])
    )

    assert over.item() == pytest.approx(0.5)
    assert under.item() == pytest.approx(0.5)


def test_high_tau_penalizes_under_prediction_more():
    """At tau=0.9 an under-prediction costs 0.9 while an equal over-prediction costs 0.1."""
    under = QuantileLoss(tau=0.9)._calc_unreduced_loss(
        torch.tensor([[0.0]]), torch.tensor([[1.0]])
    )
    over = QuantileLoss(tau=0.9)._calc_unreduced_loss(
        torch.tensor([[1.0]]), torch.tensor([[0.0]])
    )

    assert under.item() == pytest.approx(0.9)
    assert over.item() == pytest.approx(0.1)


def test_high_tau_gradient_pushes_prediction_upward():
    """When the model under-predicts at tau=0.9 the gradient is -tau, so descent raises preds."""
    preds = torch.tensor([[0.0]], requires_grad=True)
    targets = torch.tensor([[1.0]])

    QuantileLoss(tau=0.9)._calc_unreduced_loss(preds, targets).sum().backward()

    assert preds.grad.item() == pytest.approx(-0.9)


def test_quantile_tau_out_of_range_rejected():
    """A quantile outside (0, 1) is invalid and raises."""
    with pytest.raises(ValueError, match="between 0 and 1"):
        QuantileLoss(tau=1.0)


def test_quantile_tau_builds_quantile_criterion():
    """Setting quantile_tau gives the regression head a QuantileLoss with that quantile."""
    model = ChemPropModel(n_tasks=1, quantile_tau=0.9)
    model.build(scaler=_make_scaler([2.0]))

    assert isinstance(model.estimator.criterion, QuantileLoss)
    assert model.estimator.criterion.tau == pytest.approx(0.9)


def test_quantile_val_metric_shares_quantile():
    """The val_loss metric carries the same quantile as the training criterion."""
    model = ChemPropModel(n_tasks=1, quantile_tau=0.9)
    model.build(scaler=_make_scaler([2.0]))

    val_metric = model.estimator.metrics[-1]

    assert isinstance(val_metric, QuantileLoss)
    assert val_metric.tau == pytest.approx(0.9)


def test_quantile_tau_out_of_range_rejected_at_model_validation():
    """A quantile outside (0, 1) fails model validation."""
    with pytest.raises(ValueError, match="between 0 and 1"):
        ChemPropModel(quantile_tau=1.5)
