"""Accelerator-default guard for the Lightning trainer."""

from openadmet.models.trainer.lightning import LightningTrainer


def test_lightning_trainer_accelerator_defaults_to_auto():
    """The Lightning trainer field must default ``accelerator`` to ``"auto"``."""
    assert LightningTrainer.model_fields["accelerator"].default == "auto"
