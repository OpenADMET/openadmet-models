"""Guard the default accelerator of the prediction entry points.

``"gpu"`` is not a portable accelerator spelling: the value is forwarded
straight to the Lightning Trainer (it does not go through ``_resolve_device``),
and Lightning resolves ``"gpu"`` to CUDA, so on a machine with no GPU at all the
call raises ``MisconfigurationException``. Every prediction entry point must
therefore default to ``"auto"``, which lets Lightning pick TPU, MPS, CUDA or CPU
whichever is available -- matching ``predict_embedding``, which already does.
"""

import inspect

import pytest

from openadmet.models.architecture.chemprop import ChemPropModel
from openadmet.models.architecture.nepare import NeuralPairwiseRegressorModel
from openadmet.models.inference.inference import predict as inference_predict
from openadmet.models.trainer.lightning import LightningTrainer


@pytest.mark.parametrize(
    "func",
    [
        ChemPropModel.predict,
        NeuralPairwiseRegressorModel.predict,
        inference_predict,
    ],
    ids=["chemprop.predict", "nepare.predict", "inference.predict"],
)
def test_predict_accelerator_defaults_to_auto(func):
    """Every ``predict`` entry point must default ``accelerator`` to ``"auto"``."""
    assert inspect.signature(func).parameters["accelerator"].default == "auto"


def test_lightning_trainer_accelerator_defaults_to_auto():
    """The Lightning trainer field must default ``accelerator`` to ``"auto"``."""
    assert LightningTrainer.model_fields["accelerator"].default == "auto"


def test_predict_cli_accelerator_defaults_to_auto():
    """The ``--accelerator`` CLI option must default to ``"auto"``."""
    from openadmet.models.cli import predict as predict_cli_module

    option = next(
        param
        for param in predict_cli_module.predict.params
        if param.name == "accelerator"
    )
    assert option.default == "auto"
