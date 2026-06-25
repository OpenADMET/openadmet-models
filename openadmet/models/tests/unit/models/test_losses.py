import numpy as np
import pytest
import torch
from chemprop.nn.metrics import MSE
from sklearn.preprocessing import StandardScaler

from openadmet.models.architecture.chemprop import ChemPropModel
from openadmet.models.architecture.losses import NoiseThresholdedMSE


def test_zero_delta_equals_mse():
    """At delta=0 the criterion reduces to mean squared error exactly."""
    preds = torch.tensor([[1.0, 3.0], [-2.0, 0.5]])
    targets = torch.tensor([[0.0, 0.0], [0.0, 0.0]])

    criterion = NoiseThresholdedMSE(delta=0.0)
    criterion.update(preds, targets)
    reference = MSE()
    reference.update(preds, targets)

    assert torch.allclose(criterion.compute(), reference.compute())


def test_zero_delta_matches_hand_computed_mean():
    """delta=0 over residuals 1 and 3 gives mean squared error (1+9)/2."""
    preds = torch.tensor([[1.0, 3.0]])
    targets = torch.tensor([[0.0, 0.0]])

    criterion = NoiseThresholdedMSE(delta=0.0)
    criterion.update(preds, targets)

    assert criterion.compute().item() == pytest.approx(5.0)


def test_residuals_within_threshold_contribute_no_loss():
    """A residual smaller than delta yields zero unreduced loss."""
    preds = torch.tensor([[0.3]])
    targets = torch.tensor([[0.0]])

    loss = NoiseThresholdedMSE(delta=0.5)._calc_unreduced_loss(preds, targets)

    assert loss.item() == 0.0


def test_residuals_beyond_threshold_use_squared_excess():
    """A residual of 1.0 with delta 0.5 contributes (1.0 - 0.5)**2 = 0.25."""
    preds = torch.tensor([[1.0]])
    targets = torch.tensor([[0.0]])

    loss = NoiseThresholdedMSE(delta=0.5)._calc_unreduced_loss(preds, targets)

    assert loss.item() == pytest.approx(0.25)


def test_per_task_delta_applies_independently():
    """A per-task delta thresholds each task with its own value."""
    preds = torch.tensor([[1.0, 1.0]])
    targets = torch.tensor([[0.0, 0.0]])

    loss = NoiseThresholdedMSE(delta=[0.0, 0.5])._calc_unreduced_loss(preds, targets)

    assert loss[0, 0].item() == pytest.approx(1.0)
    assert loss[0, 1].item() == pytest.approx(0.25)


def test_gradient_is_zero_within_threshold():
    """No gradient flows for residuals inside the threshold."""
    preds = torch.tensor([[0.2]], requires_grad=True)
    targets = torch.tensor([[0.0]])

    NoiseThresholdedMSE(delta=0.5)._calc_unreduced_loss(preds, targets).sum().backward()

    assert preds.grad.item() == 0.0


def test_gradient_beyond_threshold_is_squared_excess_derivative():
    """Outside the threshold the gradient is 2*(|r| - delta) in the residual direction."""
    preds = torch.tensor([[1.0]], requires_grad=True)
    targets = torch.tensor([[0.0]])

    NoiseThresholdedMSE(delta=0.5)._calc_unreduced_loss(preds, targets).sum().backward()

    assert preds.grad.item() == pytest.approx(2.0 * (1.0 - 0.5))


def test_negative_delta_rejected():
    """A negative threshold is invalid and raises."""
    with pytest.raises(ValueError, match="non-negative"):
        NoiseThresholdedMSE(delta=-0.1)


def test_experimental_uncertainty_zero_keeps_default_mse():
    """experimental_uncertainty=0 leaves the ChemProp default MSE criterion in place."""
    model = ChemPropModel(n_tasks=2, experimental_uncertainty=0.0)
    model.build(scaler=_make_scaler([2.0, 0.5]))

    assert isinstance(model.estimator.criterion, MSE)
    assert not isinstance(model.estimator.criterion, NoiseThresholdedMSE)


def test_experimental_uncertainty_scales_delta_per_task():
    """A raw-unit threshold is divided by each task's target std to reach loss space."""
    model = ChemPropModel(n_tasks=2, experimental_uncertainty=0.3)
    model.build(scaler=_make_scaler([2.0, 0.5]))

    delta = model.estimator.criterion.delta.flatten().tolist()

    assert delta == pytest.approx([0.15, 0.6])


def test_experimental_uncertainty_without_scaler_uses_raw_threshold():
    """With no scaler the threshold is used directly for every task."""
    model = ChemPropModel(
        n_tasks=2, experimental_uncertainty=0.3, normalized_targets=False
    )
    model.build(scaler=None)

    delta = model.estimator.criterion.delta.flatten().tolist()

    assert delta == pytest.approx([0.3, 0.3])


def test_val_loss_metric_shares_threshold():
    """The metric used for val_loss carries the same per-task threshold as the criterion."""
    model = ChemPropModel(n_tasks=2, experimental_uncertainty=0.3)
    model.build(scaler=_make_scaler([2.0, 0.5]))

    val_metric = model.estimator.metrics[-1]

    assert isinstance(val_metric, NoiseThresholdedMSE)
    assert torch.allclose(val_metric.delta, model.estimator.criterion.delta)


def test_negative_experimental_uncertainty_rejected():
    """A negative noise floor fails model validation."""
    with pytest.raises(ValueError, match="non-negative"):
        ChemPropModel(experimental_uncertainty=-1.0)


def _make_scaler(scale: list[float]) -> StandardScaler:
    scaler = StandardScaler()
    scaler.scale_ = np.array(scale)
    scaler.mean_ = np.zeros(len(scale))
    return scaler
