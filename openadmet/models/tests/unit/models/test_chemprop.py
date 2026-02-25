import numpy as np
import pytest
import torch

from openadmet.models.architecture.chemprop import ChemPropModel


def test_chemprop_hyperparameters():
    """Test that ChemPropModel accepts new hyperparameters."""
    model = ChemPropModel(
        mpnn_lr=1e-3,
        ffn_lr=5e-4,
        mpnn_weight_decay=1e-5,
        ffn_weight_decay=1e-4,
        reduce_lr_on_plateau=True,
        reduce_lr_factor=0.5,
        reduce_lr_patience=5,
    )

    assert model.mpnn_lr == 1e-3
    assert model.ffn_lr == 5e-4
    assert model.mpnn_weight_decay == 1e-5
    assert model.ffn_weight_decay == 1e-4
    assert model.reduce_lr_on_plateau is True
    assert model.reduce_lr_factor == 0.5
    assert model.reduce_lr_patience == 5


def test_chemprop_configure_optimizers():
    """Test that configure_optimizers creates correct parameter groups."""
    model = ChemPropModel(
        mpnn_lr=1e-3,
        ffn_lr=5e-4,
        mpnn_weight_decay=1e-5,
        ffn_weight_decay=1e-4,
        reduce_lr_on_plateau=True,
    )

    # Need to call build() to initialize the estimator (mpnn)
    # build() might require downloading CheMeleon or fail if no internet?
    # No, by default it builds a fresh MPNN unless `from_chemeleon=True`.
    model.build()

    # Mock trainer attributes needed for configure_optimizers
    # But since we use reduce_lr_on_plateau=True, we don't need trainer attributes
    # (except maybe for logging if verbose=True, but that's fine).

    # Call the bound method on the estimator
    optimizer_config = model.estimator.configure_optimizers()

    opt = optimizer_config["optimizer"]
    scheduler_config = optimizer_config["lr_scheduler"]

    # Check optimizer groups
    assert len(opt.param_groups) == 2
    # Identify which group is which based on lr/weight_decay
    group1 = opt.param_groups[0]
    group2 = opt.param_groups[1]

    # Based on implementation:
    # Group 0 is MPNN (mpnn_lr, mpnn_weight_decay)
    # Group 1 is FFN (ffn_lr, ffn_weight_decay)

    assert group1["lr"] == 1e-3
    assert group1["weight_decay"] == 1e-5

    assert group2["lr"] == 5e-4
    assert group2["weight_decay"] == 1e-4

    # Check scheduler
    assert isinstance(
        scheduler_config["scheduler"], torch.optim.lr_scheduler.ReduceLROnPlateau
    )
    assert scheduler_config["monitor"] == "val_loss"
    assert scheduler_config["interval"] == "epoch"


def test_chemprop_configure_optimizers_noam():
    """Test configure_optimizers with default Noam scheduler."""
    model = ChemPropModel(mpnn_lr=1e-4, ffn_lr=1e-4, reduce_lr_on_plateau=False)
    model.build()

    # Need to mock trainer attributes for Noam scheduler calculation
    class MockTrainer:
        train_dataloader = None  # Trigger estimated_stepping_batches logic if needed, but easier to set num_training_batches directly
        num_training_batches = 100
        max_epochs = 10
        estimated_stepping_batches = 1000  # Just in case

    model.estimator.trainer = MockTrainer()

    optimizer_config = model.estimator.configure_optimizers()
    scheduler_config = optimizer_config["lr_scheduler"]

    # Check scheduler type (it's a LambdaLR from build_NoamLike_LRSched)
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

    # Should be called by validator
    model.n_tasks = 3
    # Manually trigger validation/update if needed, or create new instance
    # Pydantic models validate on init, assignment validation depends on config
    # Let's just check init behavior which is what the validator decorates
    model2 = ChemPropModel(n_tasks=10)
    assert model2._n_tasks == 10


def test_chemprop_get_output_transform():
    """Test _get_output_transform logic."""
    from chemprop import nn

    model = ChemPropModel(n_tasks=1)

    # Case 1: Scaler provided
    class MockScaler:
        pass

    scaler = MockScaler()
    scaler.mean_ = np.array([0.5])
    scaler.scale_ = np.array([2.0])
    scaler.n_features_in_ = 1

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

    # Check layer 0 of FFN is frozen (assuming it's the first layer)
    for p in model.estimator.predictor.ffn[0].parameters():
        assert p.requires_grad is False
