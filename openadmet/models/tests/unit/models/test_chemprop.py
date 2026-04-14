import numpy as np
import pytest
import torch

from openadmet.models.architecture.chemprop import ChemPropModel


def test_chemprop_hyperparameters_overrides():
    """Test that ChemPropModel accepts overrides."""
    model = ChemPropModel(
        max_lr=1e-3,
        mpnn_lr=1e-5,  # Override
        ffn_lr=5e-4,  # Override
        weight_decay=1e-6,
        mpnn_weight_decay=1e-5,  # Override
        ffn_weight_decay=1e-4,  # Override
        scheduler="plateau",
        reduce_lr_factor=0.5,
        reduce_lr_patience=5,
    )

    assert model.max_lr == 1e-3
    assert model.mpnn_lr == 1e-5
    assert model.ffn_lr == 5e-4
    assert model.weight_decay == 1e-6
    assert model.mpnn_weight_decay == 1e-5
    assert model.ffn_weight_decay == 1e-4
    assert model.scheduler == "plateau"
    assert model.reduce_lr_factor == 0.5
    assert model.reduce_lr_patience == 5


def test_chemprop_hyperparameters_defaults():
    """Test that ChemPropModel cascades defaults."""
    model = ChemPropModel(max_lr=1e-3, weight_decay=1e-5, scheduler="noam")

    # LRs should inherit max_lr or derived values
    assert model.mpnn_lr == 1e-3
    assert model.ffn_lr == 1e-3
    assert model.init_lr == 1e-4  # max_lr * 0.1
    assert model.final_lr == 1e-5  # max_lr * 0.01

    # Weight decays should inherit global weight_decay
    assert model.mpnn_weight_decay == 1e-5
    assert model.ffn_weight_decay == 1e-5


def test_chemprop_hyperparameters_partial_overrides():
    """Test that component overrides only affect explicitly provided fields."""
    model = ChemPropModel(
        max_lr=1e-3,
        scheduler="noam",
        mpnn_lr=1e-5,
        mpnn_weight_decay=0.01,
    )

    assert model.mpnn_lr == 1e-5
    assert model.ffn_lr == 1e-3
    assert model.mpnn_weight_decay == 0.01
    assert model.ffn_weight_decay == 0.0
    assert model.init_lr == 1e-4
    assert model.final_lr == 1e-5


def test_chemprop_invalid_scheduler_value():
    """Test scheduler field validator for allowed values."""
    with pytest.raises(
        ValueError, match="Scheduler must be either 'noam' or 'plateau'"
    ):
        ChemPropModel(scheduler="reduce_on_plateau")


def test_chemprop_scheduler_mutual_exclusivity():
    """Test mutual exclusivity of scheduler parameters."""

    # Test plateau with noam param
    with pytest.raises(
        ValueError, match="warmup_epochs is not compatible with plateau scheduler"
    ):
        ChemPropModel(scheduler="plateau", warmup_epochs=5)

    # Test noam with plateau param
    with pytest.raises(
        ValueError, match="reduce_lr_factor is not compatible with noam scheduler"
    ):
        ChemPropModel(scheduler="noam", reduce_lr_factor=0.5)

    # Test noam with plateau param
    with pytest.raises(
        ValueError, match="reduce_lr_patience is not compatible with noam scheduler"
    ):
        ChemPropModel(scheduler="noam", reduce_lr_patience=5)

    # Test plateau factor validity
    with pytest.raises(ValueError, match="reduce_lr_factor must be < 1.0"):
        ChemPropModel(scheduler="plateau", reduce_lr_factor=1.5)


def test_chemprop_configure_optimizers_plateau():
    """Test configure_optimizers with Plateau scheduler."""
    model = ChemPropModel(
        max_lr=1e-3, scheduler="plateau", reduce_lr_factor=0.5, reduce_lr_patience=5
    )

    model.build()

    # Mock trainer not needed for plateau config setup usually, but let's be safe
    # model.estimator.trainer = ...

    optimizer_config = model.estimator.configure_optimizers()
    opt = optimizer_config["optimizer"]
    scheduler_config = optimizer_config["lr_scheduler"]

    # Check optimizer groups
    assert len(opt.param_groups) == 2
    # Group 0 is MPNN (inherits max_lr=1e-3)
    assert opt.param_groups[0]["lr"] == 1e-3

    # Check scheduler
    assert isinstance(
        scheduler_config["scheduler"], torch.optim.lr_scheduler.ReduceLROnPlateau
    )
    assert scheduler_config["monitor"] == "val_loss"
    assert scheduler_config["interval"] == "epoch"
    assert scheduler_config["scheduler"].factor == 0.5
    assert scheduler_config["scheduler"].patience == 5


def test_chemprop_configure_optimizers_noam():
    """Test configure_optimizers with Noam scheduler."""
    model = ChemPropModel(max_lr=1e-4, scheduler="noam")
    model.build()

    # Need to mock trainer attributes for Noam scheduler calculation
    class MockTrainer:
        train_dataloader = None
        num_training_batches = 100
        max_epochs = 10
        estimated_stepping_batches = 1000

    model.estimator.trainer = MockTrainer()

    optimizer_config = model.estimator.configure_optimizers()
    scheduler_config = optimizer_config["lr_scheduler"]

    # Check scheduler type
    assert isinstance(scheduler_config["scheduler"], torch.optim.lr_scheduler.LambdaLR)
    assert scheduler_config["interval"] == "step"


def test_chemprop_validation():
    """Test validation of messages and aggregation parameters."""
    # Test valid inputs
    ChemPropModel(messages="bond", aggregation="mean")
    ChemPropModel(messages="atom", aggregation="norm")

    # Test invalid messages
    with pytest.raises(ValueError, match="Messages must be either 'bond' or 'atom'"):
        ChemPropModel(messages="invalid")

    # Test invalid aggregation
    with pytest.raises(ValueError, match="Aggregation must be either 'mean' or 'norm'"):
        ChemPropModel(aggregation="invalid")


def test_chemprop_set_n_tasks():
    """Test that set_n_tasks correctly updates _n_tasks."""
    model = ChemPropModel(n_tasks=5)
    assert model._n_tasks == 5

    model2 = ChemPropModel(n_tasks=10)
    assert model2._n_tasks == 10


def test_chemprop_get_output_transform():
    """Test _get_output_transform logic."""
    from chemprop import nn

    model = ChemPropModel(n_tasks=1)

    # Case 1: Scaler provided
    class MockScaler:
        mean_ = np.array([0.5])
        scale_ = np.array([2.0])
        n_features_in_ = 1

    scaler = MockScaler()
    transform = model._get_output_transform(scaler)
    assert isinstance(transform, nn.UnscaleTransform)

    # Case 2: normalized_targets=True (default), no scaler
    transform = model._get_output_transform(None)
    assert isinstance(transform, nn.UnscaleTransform)

    # Case 3: normalized_targets=False, no scaler
    model.normalized_targets = False
    transform = model._get_output_transform(None)
    assert transform is None


def test_chemprop_predict_untrained():
    """Test that predict raises AttributeError when model is not trained."""
    model = ChemPropModel()
    with pytest.raises(AttributeError, match="Model not trained"):
        model.predict(np.array([[1]]))


def test_chemprop_freeze_weights():
    """Test freeze_weights functionality."""
    model = ChemPropModel(ffn_num_layers=2)
    model.build()

    # Initial state: everything requires grad
    for p in model.estimator.parameters():
        assert p.requires_grad is True

    # Freeze MPNN
    model.freeze_weights(message_passing=True, batch_norm=False, ffn_layers=0)

    for p in model.estimator.message_passing.parameters():
        assert p.requires_grad is False

    # FFN should still require grad
    for p in model.estimator.predictor.parameters():
        assert p.requires_grad is True

    # Test invalid FFN layers
    with pytest.raises(
        ValueError, match="Requested to freeze 3 feedforward network layer"
    ):
        model.freeze_weights(ffn_layers=3)

    # Freeze 1 FFN layer
    model.freeze_weights(message_passing=False, batch_norm=False, ffn_layers=1)

    # Check layer 0 of FFN is frozen
    for p in model.estimator.predictor.ffn[0].parameters():
        assert p.requires_grad is False
