"""Guard the default accelerator of the prediction entry points.

``"gpu"`` is not a portable accelerator spelling: Lightning resolves it to
CUDA, so on any machine without CUDA (an Ascend NPU box, a CPU-only runner, an
Apple MPS host) the default either raises ``MisconfigurationException`` or
silently ignores a usable accelerator. The prediction paths must therefore
default to ``"auto"`` and let Lightning pick the best available accelerator.
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
