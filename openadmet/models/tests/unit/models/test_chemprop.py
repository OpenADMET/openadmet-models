import types

import numpy as np
import pytest
import torch

from openadmet.models.architecture.chemprop import (
    ChemPropModel,
    _resolve_noam_steps_per_epoch,
)


@pytest.fixture
def make_noam_trainer():
    """Return a factory for Noam-scheduler trainer stubs.

    Defaults match a typical small run (100 batches/epoch, 10 epochs).
    Override any attribute to exercise edge-case schedule shapes.
    """

    def _factory(
        num_training_batches=100,
        max_epochs=10,
        estimated_stepping_batches=1000,
        accumulate_grad_batches=1,
    ):
        return types.SimpleNamespace(
            num_training_batches=num_training_batches,
            max_epochs=max_epochs,
            estimated_stepping_batches=estimated_stepping_batches,
            accumulate_grad_batches=accumulate_grad_batches,
        )

    return _factory


@pytest.fixture
def make_val_trainer():
    """Return a factory for val-dataloader trainer stubs used by the plateau warning check."""

    def _factory(num_val_batches=()):
        return types.SimpleNamespace(num_val_batches=list(num_val_batches))

    return _factory


def test_resolve_noam_steps_per_epoch_uses_num_training_batches(make_noam_trainer):
    """num_training_batches is used directly when the trainer reports a finite value."""
    trainer = make_noam_trainer(num_training_batches=50)

    assert _resolve_noam_steps_per_epoch(trainer) == 50


def test_resolve_noam_steps_per_epoch_converts_estimated_stepping_batches(
    make_noam_trainer,
):
    """estimated_stepping_batches is converted back to raw batch units via grad accumulation."""
    trainer = make_noam_trainer(
        num_training_batches=float("inf"),
        max_epochs=10,
        estimated_stepping_batches=500,
        accumulate_grad_batches=2,
    )

    # (500 estimated * 2 grad_accum) // 10 epochs = 100 raw batches/epoch
    assert _resolve_noam_steps_per_epoch(trainer) == 100


def test_resolve_noam_steps_per_epoch_falls_back_to_1000(make_noam_trainer):
    """Falls back to 1000 and warns when neither trainer field is usable."""
    from loguru import logger

    trainer = make_noam_trainer(
        num_training_batches=float("inf"), estimated_stepping_batches=float("inf")
    )

    captured = []
    handler_id = logger.add(
        lambda msg: captured.append(msg.record["message"]), level="WARNING"
    )
    try:
        steps_per_epoch = _resolve_noam_steps_per_epoch(trainer)
    finally:
        logger.remove(handler_id)

    assert steps_per_epoch == 1000
    assert any("falling back to 1000" in w for w in captured)


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


def test_chemprop_scheduler_defaults_are_scheduler_specific():
    """Test that unset cross-scheduler params stay None and scheduler defaults are filled."""
    noam = ChemPropModel(scheduler="noam")
    assert noam.warmup_epochs == 2
    assert noam.reduce_lr_factor is None
    assert noam.reduce_lr_patience is None

    plateau = ChemPropModel(scheduler="plateau")
    assert plateau.warmup_epochs is None
    assert plateau.reduce_lr_factor == 0.5
    assert plateau.reduce_lr_patience == 5


def test_chemprop_configure_optimizers_plateau():
    """Test configure_optimizers with Plateau scheduler."""
    model = ChemPropModel(
        max_lr=1e-3, scheduler="plateau", reduce_lr_factor=0.5, reduce_lr_patience=5
    )

    model.build()

    optimizer_config = model.estimator.configure_optimizers()
    opt = optimizer_config["optimizer"]
    scheduler_config = optimizer_config["lr_scheduler"]

    # Check optimizer groups
    assert len(opt.param_groups) == 2
    # Group 0 is MPNN (inherits max_lr=1e-3)
    assert opt.param_groups[0]["lr"] == 1e-3

    # Check scheduler
    sched = scheduler_config["scheduler"]
    assert isinstance(sched, torch.optim.lr_scheduler.ReduceLROnPlateau)
    assert scheduler_config["monitor"] == "val_loss"
    assert scheduler_config["interval"] == "epoch"
    assert sched.factor == 0.5
    assert sched.patience == 5

    # min_lrs are per-group floors: group_lr * (final_lr / max_lr)
    # Both groups have max_lr=1e-3, final_lr=1e-5, so floor = 1e-3 * 0.01 = 1e-5
    assert sched.min_lrs == pytest.approx([1e-5, 1e-5])

    # scheduler is recorded in hparams; warmup_epochs is absent for plateau runs
    assert model.estimator.hparams.get("scheduler") == "plateau"
    assert "warmup_epochs" not in model.estimator.hparams


def test_chemprop_configure_optimizers_noam(make_noam_trainer):
    """Test configure_optimizers with Noam scheduler."""
    model = ChemPropModel(max_lr=1e-4, scheduler="noam")
    model.build()

    model.estimator._trainer = make_noam_trainer()

    optimizer_config = model.estimator.configure_optimizers()
    scheduler_config = optimizer_config["lr_scheduler"]

    # Check scheduler type
    assert isinstance(scheduler_config["scheduler"], torch.optim.lr_scheduler.LambdaLR)
    assert scheduler_config["interval"] == "step"

    # scheduler is recorded in hparams alongside Noam-specific keys
    assert model.estimator.hparams.get("scheduler") == "noam"
    assert "warmup_epochs" in model.estimator.hparams


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

    # Case 2: normalized_targets=True (default), no scaler — must be identity, not constant
    transform = model._get_output_transform(None)
    assert isinstance(transform, nn.UnscaleTransform)
    transform.eval()
    x = torch.tensor([[3.5]])
    assert transform(x) == pytest.approx(3.5), (
        "no-scaler transform must pass predictions through unchanged"
    )

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


def test_chemprop_load_weights_invalid_path():
    """Test that load_weights raises FileNotFoundError for invalid path."""
    model = ChemPropModel(from_foundation="doesnt_exist.pt")
    with pytest.raises(
        FileNotFoundError, match="Foundation model not found at doesnt_exist.pt"
    ):
        model.build()


# Minimal valid BondMessagePassing hyperparameters for foundation-file tests
_FOUNDATION_HPARAMS = {
    "d_h": 8,
    "depth": 1,
    "dropout": 0.0,
    "bias": False,
    "activation": "relu",
    "undirected": False,
    "d_v": 72,
    "d_e": 14,
    "d_vd": None,
    "V_d_transform": None,
    "graph_transform": None,
}


@pytest.mark.parametrize(
    "payload",
    [
        {"hyper_parameters": _FOUNDATION_HPARAMS},
        {"hyper_parameters": _FOUNDATION_HPARAMS, "state_dict": {}},
    ],
    ids=["missing_state_dict", "empty_state_dict"],
)
def test_chemprop_foundation_file_without_weights_raises(tmp_path, payload):
    """Test that build raises RuntimeError when a foundation file carries no weights."""
    path = tmp_path / "foundation.pt"
    torch.save(payload, str(path))
    model = ChemPropModel(from_foundation=str(path))
    with pytest.raises(RuntimeError, match="state_dict"):
        model.build()


def test_chemprop_chemeleon_and_foundation_mutual_exclusivity():
    """Test that from_chemeleon and from_foundation are mutually exclusive."""
    with pytest.raises(
        ValueError,
        match="Cannot specify both from_chemeleon and user-specified from_foundation",
    ):
        ChemPropModel(from_chemeleon=True, from_foundation="custom_model")


def test_chemprop_from_chemeleon_compat_success():
    """Test that from_chemeleon=True correctly maps to from_foundation='chemeleon'."""
    with pytest.warns(DeprecationWarning, match="from_chemeleon is deprecated"):
        model = ChemPropModel(from_chemeleon=True)
    assert model.from_foundation == "chemeleon"


def test_chemprop_from_chemeleon_deprecation_warning():
    """Test that using from_chemeleon emits a DeprecationWarning."""
    with pytest.warns(DeprecationWarning, match="from_chemeleon is deprecated"):
        ChemPropModel(from_chemeleon=True)


def test_chemprop_load_weights(tmp_path):
    """Test that load_weights correctly loads state dict."""

    # Create a ChemProp model and save it to a temporary file
    source_model = ChemPropModel(
        depth=3, message_hidden_dim=300, ffn_hidden_dim=200, dropout=0.1
    )
    source_model.build()

    # Save the model in foundation format
    temp_weights_path = tmp_path / "test_foundation.pt"
    foundation_data = {
        "hyper_parameters": {
            "d_h": source_model.message_hidden_dim,
            "depth": source_model.depth,
            "dropout": source_model.dropout,
            "bias": False,
            "activation": "relu",
            "undirected": False,
            "d_v": 72,
            "d_e": 14,
            "d_vd": None,
            "V_d_transform": None,
            "graph_transform": None,
        },
        "aggregation": {},  # MeanAggregation expects empty dict
        "state_dict": {},
    }

    # Extract the message passing weights
    mp_state_dict = source_model.estimator.message_passing.state_dict()
    for key in mp_state_dict:
        # Map from full state dict keys to foundation format keys
        if key.endswith(".weight") or key.endswith(".bias"):
            foundation_data["state_dict"][key] = mp_state_dict[key]

    torch.save(foundation_data, str(temp_weights_path))

    # Now test loading weights from the temporary file
    model = ChemPropModel(from_foundation=str(temp_weights_path))
    model.build()

    # Verify that the weights were loaded correctly
    loaded_weights = torch.load(str(temp_weights_path), weights_only=True)
    for key in loaded_weights["state_dict"]:
        assert torch.all(
            model.estimator.state_dict()[f"message_passing.{key}"]
            == loaded_weights["state_dict"][key]
        )


def test_chemprop_noam_lambda_boundaries(make_noam_trainer):
    """Noam lambda reaches exactly max_lr at warmup_steps and final_lr at end of cooldown."""
    model = ChemPropModel(max_lr=1e-3, scheduler="noam", warmup_epochs=2)
    model.build()

    model.estimator._trainer = make_noam_trainer()

    opt_config = model.estimator.configure_optimizers()
    lr_sched = opt_config["lr_scheduler"]["scheduler"]

    warmup_steps = 2 * 100  # warmup_epochs * steps_per_epoch
    cooldown_steps = (10 - 2) * 100

    # Advance to the step just before the warmup peak; LR must still be below max_lr
    # Verifies the warmup branch owns the peak step via `warmup_steps > 0 and step <= warmup_steps`
    for _ in range(warmup_steps - 1):
        lr_sched.step()
    assert lr_sched.get_last_lr()[0] < 1e-3

    # At warmup_steps (inclusive), the warmup branch reaches exactly 1.0 * base_lr
    lr_sched.step()
    assert lr_sched.get_last_lr()[0] == pytest.approx(1e-3, rel=1e-5)

    for _ in range(cooldown_steps):
        lr_sched.step()
    assert lr_sched.get_last_lr()[0] == pytest.approx(1e-3 * 0.01, rel=1e-3)


def test_chemprop_noam_no_warmup_infinite_training_holds_max_lr(make_noam_trainer):
    """With warmup_epochs=0 and max_epochs=-1, Noam holds at max_lr and warns."""
    from loguru import logger

    model = ChemPropModel(max_lr=1e-3, scheduler="noam", warmup_epochs=0)
    model.build()

    model.estimator._trainer = make_noam_trainer(
        max_epochs=-1, estimated_stepping_batches=float("inf")
    )

    captured = []
    handler_id = logger.add(
        lambda msg: captured.append(msg.record["message"]), level="WARNING"
    )
    try:
        opt_config = model.estimator.configure_optimizers()
    finally:
        logger.remove(handler_id)

    assert any("cannot calibrate" in w for w in captured)

    # With warmup_steps=0 and cooldown_steps=0, lambda always returns 1.0
    lr_sched = opt_config["lr_scheduler"]["scheduler"]
    assert lr_sched.get_last_lr()[0] == pytest.approx(1e-3, rel=1e-5)
    lr_sched.step()
    assert lr_sched.get_last_lr()[0] == pytest.approx(1e-3, rel=1e-5)


def test_chemprop_noam_lambda_no_warmup_starts_at_max_lr(make_noam_trainer):
    """With warmup_epochs=0, Noam starts at max_lr immediately."""
    model = ChemPropModel(max_lr=1e-3, scheduler="noam", warmup_epochs=0)
    model.build()

    model.estimator._trainer = make_noam_trainer()

    opt_config = model.estimator.configure_optimizers()
    lr_sched = opt_config["lr_scheduler"]["scheduler"]

    # LambdaLR applies the lambda at last_epoch=0 during construction
    # With warmup_steps=0 and cooldown_steps>0, decay_frac=0 gives 1.0 at step 0
    assert lr_sched.get_last_lr()[0] == pytest.approx(1e-3, rel=1e-5)


def test_chemprop_noam_warmup_exceeds_max_epochs(make_noam_trainer):
    """A warning is emitted when warmup_epochs >= max_epochs, and LR drops immediately after warmup."""
    from loguru import logger

    model = ChemPropModel(max_lr=1e-3, scheduler="noam", warmup_epochs=5)
    model.build()

    model.estimator._trainer = make_noam_trainer(
        num_training_batches=50, max_epochs=3, estimated_stepping_batches=150
    )

    captured = []
    handler_id = logger.add(
        lambda msg: captured.append(msg.record["message"]), level="WARNING"
    )
    try:
        opt_config = model.estimator.configure_optimizers()
    finally:
        logger.remove(handler_id)

    assert any("warmup_epochs" in w and "max_epochs" in w for w in captured)

    # With cooldown_steps=0, the first step after warmup should drop directly to final_lr
    lr_sched = opt_config["lr_scheduler"]["scheduler"]
    warmup_steps = 5 * 50  # warmup_epochs * num_training_batches
    for _ in range(warmup_steps):
        lr_sched.step()
    # Still at peak after warmup_steps
    assert lr_sched.get_last_lr()[0] == pytest.approx(1e-3, rel=1e-5)
    lr_sched.step()
    # First post-warmup step drops to final_lr (cooldown_steps=0, no intermediate decay)
    assert lr_sched.get_last_lr()[0] == pytest.approx(1e-3 * 0.01, rel=1e-3)


def test_chemprop_freeze_weights_eval_mode():
    """freeze_weights sets requires_grad=False and eval mode on the same FFN layer."""
    model = ChemPropModel(ffn_num_layers=2)
    model.build()

    model.freeze_weights(message_passing=False, batch_norm=False, ffn_layers=1)

    frozen_layer = model.estimator.predictor.ffn[0]
    for p in frozen_layer.parameters():
        assert p.requires_grad is False
    assert frozen_layer.training is False


def test_chemprop_monitor_metric_mode():
    """Plateau scheduler respects monitor_metric_mode."""
    model = ChemPropModel(
        scheduler="plateau",
        reduce_lr_factor=0.5,
        reduce_lr_patience=5,
        monitor_metric_mode="max",
    )
    model.build()

    opt_config = model.estimator.configure_optimizers()
    sched = opt_config["lr_scheduler"]["scheduler"]

    assert isinstance(sched, torch.optim.lr_scheduler.ReduceLROnPlateau)
    assert sched.mode == "max"


def test_chemprop_monitor_metric_mode_invalid():
    """Invalid monitor_metric_mode raises ValueError."""
    with pytest.raises(
        ValueError, match="monitor_metric_mode must be either 'min' or 'max'"
    ):
        ChemPropModel(scheduler="plateau", monitor_metric_mode="median")


def test_chemprop_serialize_includes_resolved_lr(tmp_path):
    """serialize() includes resolved LR fields and structural fields even when not explicitly set."""
    import json

    model = ChemPropModel(max_lr=2e-3, scheduler="noam")
    model.build()

    param_path = tmp_path / "params.json"
    serial_path = tmp_path / "model.pth"
    model.serialize(param_path=str(param_path), serial_path=str(serial_path))

    with open(param_path) as f:
        saved = json.load(f)

    # Resolved LR fields
    assert saved["init_lr"] == pytest.approx(2e-4)  # max_lr * 0.1
    assert saved["final_lr"] == pytest.approx(2e-5)  # max_lr * 0.01
    assert saved["mpnn_lr"] == pytest.approx(2e-3)
    assert saved["ffn_lr"] == pytest.approx(2e-3)
    # Active scheduler fields persisted; inactive plateau fields absent
    assert saved["warmup_epochs"] == 2
    assert "reduce_lr_factor" not in saved
    assert "reduce_lr_patience" not in saved
    # Structural fields always present for checkpoint compatibility
    assert saved["n_tasks"] == 1
    assert saved["depth"] == 3
    assert saved["ffn_num_layers"] == 2
    assert saved["aggregation"] == "mean"
    assert saved["batch_norm"] is False


def test_chemprop_serialize_includes_plateau_resolved_fields(tmp_path):
    """serialize() includes resolved plateau-specific fields and excludes noam-specific fields."""
    import json

    model = ChemPropModel(max_lr=1e-3, scheduler="plateau")
    model.build()

    param_path = tmp_path / "params.json"
    serial_path = tmp_path / "model.pth"
    model.serialize(param_path=str(param_path), serial_path=str(serial_path))

    with open(param_path) as f:
        saved = json.load(f)

    assert saved["reduce_lr_factor"] == pytest.approx(0.5)
    assert saved["reduce_lr_patience"] == 5
    # warmup_epochs is None for plateau; must not appear in the artifact
    assert "warmup_epochs" not in saved


def test_chemprop_plateau_hparams_reflect_configured_lrs():
    """mpnn.hparams for plateau reflects the user's max_lr/final_lr, not MPNN's own defaults."""
    model = ChemPropModel(max_lr=5e-3, final_lr=5e-5, scheduler="plateau")
    model.build()

    assert model.estimator.hparams["max_lr"] == pytest.approx(5e-3)
    assert model.estimator.hparams["final_lr"] == pytest.approx(5e-5)
    assert "init_lr" not in model.estimator.hparams


def test_chemprop_resolved_fields_matches_tagged_declarations():
    """_resolved_fields() returns exactly the fields tagged resolved=True."""
    expected = {
        "scheduler",
        "n_tasks",
        "depth",
        "message_hidden_dim",
        "ffn_hidden_dim",
        "ffn_num_layers",
        "aggregation",
        "messages",
        "batch_norm",
        "dropout",
        "normalized_targets",
        "init_lr",
        "final_lr",
        "mpnn_lr",
        "ffn_lr",
        "mpnn_weight_decay",
        "ffn_weight_decay",
        "warmup_epochs",
        "reduce_lr_factor",
        "reduce_lr_patience",
    }

    assert ChemPropModel._resolved_fields() == expected


def test_chemprop_plateau_warns_without_val_dataloader(make_val_trainer):
    """Plateau scheduler warns when no validation dataloader is configured."""
    from loguru import logger

    model = ChemPropModel(scheduler="plateau", reduce_lr_factor=0.5)
    model.build()

    # on_train_start is the correct hook; configure_optimizers fires before
    # Lightning populates num_val_batches and would always see an empty list
    model.estimator._trainer = make_val_trainer()

    captured = []
    handler_id = logger.add(
        lambda msg: captured.append(msg.record["message"]), level="WARNING"
    )
    try:
        model.estimator.on_train_start()
    finally:
        logger.remove(handler_id)

    assert any("no validation dataloader" in w for w in captured)


def test_chemprop_plateau_no_spurious_warning_with_val(make_val_trainer):
    """Plateau scheduler emits no warning when a validation dataloader is present."""
    from loguru import logger

    model = ChemPropModel(scheduler="plateau", reduce_lr_factor=0.5)
    model.build()

    model.estimator._trainer = make_val_trainer(num_val_batches=[100])

    captured = []
    handler_id = logger.add(
        lambda msg: captured.append(msg.record["message"]), level="WARNING"
    )
    try:
        model.estimator.on_train_start()
    finally:
        logger.remove(handler_id)

    assert not any("no validation dataloader" in w for w in captured)


def test_chemprop_plateau_min_lr_per_group():
    """Per-group min_lr floors respect each group's peak LR, not global max_lr."""
    model = ChemPropModel(
        scheduler="plateau",
        max_lr=1e-3,
        mpnn_lr=5e-4,
        reduce_lr_factor=0.5,
    )
    model.build()

    opt_config = model.estimator.configure_optimizers()
    sched = opt_config["lr_scheduler"]["scheduler"]

    # final_lr = max_lr * 0.01 = 1e-5; ratio = final_lr / max_lr = 0.01
    # MPNN group: mpnn_lr * 0.01 = 5e-4 * 0.01 = 5e-6
    # FFN group:  ffn_lr * 0.01 = 1e-3 * 0.01 = 1e-5
    assert sched.min_lrs[0] == pytest.approx(5e-6)
    assert sched.min_lrs[1] == pytest.approx(1e-5)


def test_predict_embedding_unbuilt_raises():
    model = ChemPropModel(from_foundation="chemeleon")
    with pytest.raises(ValueError, match="has not been built"):
        model.predict_embedding(["CCO"])


def test_predict_embedding_shape_and_dtype():
    from openadmet.models.architecture.chemprop import ChemPropModel

    model = ChemPropModel(from_foundation="chemeleon-test")
    model.build()
    smiles = ["CCO", "CCN", "c1ccccc1"]
    emb = model.predict_embedding(smiles, batch_size=2)
    assert emb.shape == (3, 2048)
    assert emb.dtype == np.float32


def test_predict_embedding_safe_batch_size_no_drop():
    from openadmet.models.architecture.chemprop import ChemPropModel

    model = ChemPropModel(from_foundation="chemeleon-test")
    model.build()
    smiles = ["CCO", "CCN", "c1ccccc1"]
    emb = model.predict_embedding(smiles, batch_size=3)
    assert emb.shape[0] == len(smiles)


def test_predict_embedding_deterministic():
    from openadmet.models.architecture.chemprop import ChemPropModel

    model = ChemPropModel(from_foundation="chemeleon-test")
    model.build()
    smiles = ["CCO", "CCN"]
    e1 = model.predict_embedding(smiles, batch_size=2)
    e2 = model.predict_embedding(smiles, batch_size=2)
    assert np.array_equal(e1, e2)
