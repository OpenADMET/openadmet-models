import shutil
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import os
import pytest
import yaml

from openadmet.models.inference.inference import load_anvil_model_and_metadata, predict
from openadmet.models.tests.unit.datafiles import (
    pred_test_data_csv,
    anvil_lgbm_trained_model_dir,
    anvil_chemprop_trained_model_dir,
)


@pytest.fixture
def anvil_lgbm():
    return anvil_lgbm_trained_model_dir


@pytest.fixture
def anvil_chemprop():
    return anvil_chemprop_trained_model_dir


@pytest.mark.skipif(
    os.getenv("RUNNER_OS") == "macOS", reason="MacOS runner not enough memory"
)
@pytest.mark.parametrize("model_dir", ["anvil_lgbm", "anvil_chemprop"])
def test_predict(model_dir, request):
    # Use the fixture to get the model directory
    model_dir = request.getfixturevalue(model_dir)
    # Test the predict function with a sample input
    input_path = pred_test_data_csv
    input_col = "MY_SMILES"
    model_dir = [model_dir]
    write_csv = False
    output_path = None
    debug = False

    result = predict(
        input_path,
        input_col,
        model_dir,
        write_csv,
        output_path,
        debug=False,
        accelerator="cpu",
    )

    # Check if the result is a DataFrame
    assert isinstance(result, pd.DataFrame)


def test_load_anvil_forces_shuffle_false(tmp_path):
    """A featurizer saved with shuffle=True during training must have shuffle forced to False at load time."""
    # Copy the existing fixture so we have a valid directory structure to mutate
    model_dir = tmp_path / "model"
    shutil.copytree(anvil_chemprop_trained_model_dir, model_dir)

    # Overwrite procedure.yaml to simulate a model saved with shuffle=True during training
    procedure = {
        "feat": {"type": "ChemPropFeaturizer", "params": {"shuffle": True}},
        "model": {"type": "ChemPropModel", "params": {}},
        "split": {
            "type": "ShuffleSplitter",
            "params": {"random_state": 42, "test_size": 0.3, "train_size": 0.7},
        },
        "train": {
            "type": "LightningTrainer",
            "params": {"gpus": 1, "max_epochs": 10, "use_wandb": False},
        },
    }
    (model_dir / "recipe_components" / "procedure.yaml").write_text(yaml.dump(procedure))

    # Patch model deserialization — loading weights is orthogonal to what we're testing
    with patch(
        "openadmet.models.architecture.model_base.LightningModelBase.deserialize",
        return_value=MagicMock(),
    ):
        _, feat, _, _ = load_anvil_model_and_metadata(model_dir)

    assert feat.shuffle is False
