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
        reduce_lr_patience=5
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
        reduce_lr_on_plateau=True
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
    assert isinstance(scheduler_config["scheduler"], torch.optim.lr_scheduler.ReduceLROnPlateau)
    assert scheduler_config["monitor"] == "val_loss"
    assert scheduler_config["interval"] == "epoch"

def test_chemprop_configure_optimizers_noam():
    """Test configure_optimizers with default Noam scheduler."""
    model = ChemPropModel(
        mpnn_lr=1e-4,
        ffn_lr=1e-4,
        reduce_lr_on_plateau=False
    )
    model.build()
    
    # Need to mock trainer attributes for Noam scheduler calculation
    class MockTrainer:
        train_dataloader = None # Trigger estimated_stepping_batches logic if needed, but easier to set num_training_batches directly
        num_training_batches = 100
        max_epochs = 10
        estimated_stepping_batches = 1000 # Just in case

    model.estimator.trainer = MockTrainer()
    
    optimizer_config = model.estimator.configure_optimizers()
    scheduler_config = optimizer_config["lr_scheduler"]
    
    # Check scheduler type (it's a LambdaLR from build_NoamLike_LRSched)
    assert isinstance(scheduler_config["scheduler"], torch.optim.lr_scheduler.LambdaLR)
    assert scheduler_config["interval"] == "step"
