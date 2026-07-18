import numpy as np
import pytest
import torch
from sklearn.preprocessing import StandardScaler

from openadmet.models.architecture.chemprop import ChemPropModel
from openadmet.models.architecture.losses import CensoredRegressionLoss


def _masks(shape: tuple[int, int], lt: bool = False, gt: bool = False):
    """Return (mask, weights, lt_mask, gt_mask) for a direct _calc_unreduced_loss call."""
    rows = shape[0]
    return (
        torch.ones(shape, dtype=torch.bool),
        torch.ones(rows),
        torch.full(shape, lt, dtype=torch.bool),
        torch.full(shape, gt, dtype=torch.bool),
    )


def _make_scaler(scale: list[float]) -> StandardScaler:
    scaler = StandardScaler()
    scaler.scale_ = np.array(scale)
    scaler.mean_ = np.zeros(len(scale))
    return scaler


def test_censored_exact_branch_is_half_squared_standardized_residual():
    """With no inequality flag the loss is the Gaussian NLL: half the squared standardized residual."""
    preds = torch.tensor([[1.0]])
    targets = torch.tensor([[0.0]])

    loss = CensoredRegressionLoss(sigma=0.5)._calc_unreduced_loss(
        preds, targets, *_masks((1, 1))
    )

    assert loss.item() == pytest.approx(0.5 * (1.0 / 0.5) ** 2)


def test_censored_left_branch_below_bound_is_near_zero():
    """A left-censored row predicted far below its bound contributes almost no loss."""
    preds = torch.tensor([[-5.0]])
    targets = torch.tensor([[0.0]])

    loss = CensoredRegressionLoss(sigma=1.0)._calc_unreduced_loss(
        preds, targets, *_masks((1, 1), lt=True)
    )

    assert loss.item() == pytest.approx(0.0, abs=1e-6)


def test_censored_left_branch_penalizes_prediction_above_bound():
    """A left-censored row costs more when predicted above its bound than below it."""
    targets = torch.tensor([[0.0]])
    criterion = CensoredRegressionLoss(sigma=1.0)

    above = criterion._calc_unreduced_loss(
        torch.tensor([[2.0]]), targets, *_masks((1, 1), lt=True)
    )
    below = criterion._calc_unreduced_loss(
        torch.tensor([[-2.0]]), targets, *_masks((1, 1), lt=True)
    )

    assert above.item() > below.item()


def test_censored_left_branch_gradient_pushes_prediction_down():
    """The left-censored gradient is positive, so descent lowers an over-high prediction."""
    preds = torch.tensor([[2.0]], requires_grad=True)
    targets = torch.tensor([[0.0]])

    CensoredRegressionLoss(sigma=1.0)._calc_unreduced_loss(
        preds, targets, *_masks((1, 1), lt=True)
    ).sum().backward()

    assert preds.grad.item() > 0.0


def test_censored_left_branch_target_below_bound_does_not_anchor():
    """A row whose target sits at the bound is not pulled toward it the way exact MSE is."""
    preds = torch.tensor([[-3.0]], requires_grad=True)
    targets = torch.tensor([[0.0]])

    # exact MSE at the same point would impose a large gradient toward the target; the
    # left-censored branch, knowing only "below the bound", leaves a far smaller pull
    censored = CensoredRegressionLoss(sigma=1.0)._calc_unreduced_loss(
        preds, targets, *_masks((1, 1), lt=True)
    )
    censored.sum().backward()

    assert abs(preds.grad.item()) < 2.0 * abs(-3.0 - 0.0)


def test_censored_left_branch_deep_overshoot_keeps_gradient():
    """A left-censored row predicted far above its bound retains a strong corrective gradient."""
    preds = torch.tensor([[8.0]], requires_grad=True)
    targets = torch.tensor([[0.0]])

    CensoredRegressionLoss(sigma=1.0)._calc_unreduced_loss(
        preds, targets, *_masks((1, 1), lt=True)
    ).sum().backward()

    # the log_ndtr asymptote gives a gradient near |bound - pred| / sigma here; a naive
    # log(Phi + eps) would have floored this to ~0 and silenced the correction
    assert preds.grad.item() > 1.0


def test_censored_right_branch_mirrors_left():
    """A right-censored row penalizes predictions below its bound, the mirror of left."""
    targets = torch.tensor([[0.0]])
    criterion = CensoredRegressionLoss(sigma=1.0)

    below = criterion._calc_unreduced_loss(
        torch.tensor([[-2.0]]), targets, *_masks((1, 1), gt=True)
    )
    above = criterion._calc_unreduced_loss(
        torch.tensor([[2.0]]), targets, *_masks((1, 1), gt=True)
    )

    assert below.item() > above.item()


def test_censored_exact_equals_scaled_mse_through_update():
    """Through the reduction path, an uncensored batch is half-MSE scaled by 1/sigma squared."""
    preds = torch.tensor([[1.0], [3.0]])
    targets = torch.tensor([[0.0], [0.0]])

    criterion = CensoredRegressionLoss(sigma=2.0)
    criterion.update(preds, targets)

    # 0.5 * (1^2 + 3^2) / 2 / sigma^2 = 0.5 * 5 / 4
    assert criterion.compute().item() == pytest.approx(0.5 * 5.0 / 4.0)


def test_censored_sigma_nonpositive_rejected():
    """A non-positive noise scale is invalid and raises."""
    with pytest.raises(ValueError, match="strictly positive"):
        CensoredRegressionLoss(sigma=0.0)


def test_censored_sigma_builds_censored_criterion():
    """Setting censored_sigma gives the regression head a CensoredRegressionLoss."""
    model = ChemPropModel(n_tasks=1, censored_sigma=0.5)
    model.build(scaler=_make_scaler([2.0]))

    assert isinstance(model.estimator.criterion, CensoredRegressionLoss)


def test_censored_sigma_scales_per_task():
    """A raw-unit noise scale is divided by each task's target std to reach loss space."""
    model = ChemPropModel(n_tasks=2, censored_sigma=0.6)
    model.build(scaler=_make_scaler([2.0, 0.5]))

    sigma = model.estimator.criterion.sigma.flatten().tolist()

    assert sigma == pytest.approx([0.3, 1.2])


def test_censored_sigma_without_scaler_uses_raw_scale():
    """With no scaler the noise scale is used directly for every task."""
    model = ChemPropModel(n_tasks=2, censored_sigma=0.6, normalized_targets=False)
    model.build(scaler=None)

    sigma = model.estimator.criterion.sigma.flatten().tolist()

    assert sigma == pytest.approx([0.6, 0.6])


def test_censored_val_metric_shares_sigma():
    """The metric used for val_loss carries the same per-task sigma as the criterion."""
    model = ChemPropModel(n_tasks=2, censored_sigma=0.6)
    model.build(scaler=_make_scaler([2.0, 0.5]))

    val_metric = model.estimator.metrics[-1]

    assert isinstance(val_metric, CensoredRegressionLoss)
    assert torch.allclose(val_metric.sigma, model.estimator.criterion.sigma)


def test_censored_sigma_nonpositive_rejected_at_model_validation():
    """A non-positive noise scale fails model validation."""
    with pytest.raises(ValueError, match="strictly positive"):
        ChemPropModel(censored_sigma=0.0)
