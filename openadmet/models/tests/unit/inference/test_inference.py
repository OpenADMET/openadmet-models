from pathlib import Path
import pandas as pd
import os
import pytest

from openadmet.models.inference.inference import predict
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
