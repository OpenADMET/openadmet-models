from pathlib import Path
import pandas as pd

import pytest

from openadmet.models.inference.inference import predict
from openadmet.models.tests.datafiles import pred_test_data_csv, anvil_lgbm_trained_model_dir

def test_predict():
    # Test the predict function with a sample input
    input_path = pred_test_data_csv
    input_col = "SMILES"
    model_dir = [anvil_lgbm_trained_model_dir]
    write_csv = False
    output_path = None
    debug = False

    result = predict(input_path, input_col, model_dir, write_csv, output_path, debug)

    # Check if the result is a DataFrame
    assert isinstance(result, pd.DataFrame)
