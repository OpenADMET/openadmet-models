import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import openadmet.models.inference.inference as inference_module
from openadmet.models.tests.unit.datafiles import (
    pred_test_data_csv,
    anvil_lgbm_trained_model_dir,
    anvil_chemprop_trained_model_dir,
)


class DummyPairwiseFeaturizer:
    def __init__(self, how_to_pair="ut", featurize_spy=None):
        self.how_to_pair = how_to_pair
        self._featurize_spy = featurize_spy

    def featurize(self, smiles):
        if self._featurize_spy is not None:
            self._featurize_spy(list(smiles))
        return np.zeros((len(smiles), 1)), np.arange(len(smiles))


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

    result = inference_module.predict(
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


def test_generate_pairwise_df_uses_task_idx_column():
    data = pd.DataFrame({"SMILES": ["CCO", "CCN"]})
    predictions = np.array([[1.0, 11.0], [2.0, 12.0], [3.0, 13.0]])
    std = np.array([[0.1, 1.1], [0.2, 1.2], [0.3, 1.3]])
    feat = SimpleNamespace(how_to_pair="ut")

    pairwise_df = inference_module._generate_pairwise_df(
        data=data,
        input_col="SMILES",
        feat=feat,
        predictions=predictions,
        std=std,
        predictions_tag="pred",
        std_tag="std",
        task_idx=1,
    )

    assert pairwise_df["pred"].tolist() == [11.0, 12.0, 13.0]
    assert pairwise_df["std"].tolist() == [1.1, 1.2, 1.3]


def test_predict_single_task_pairwise_uses_column_zero(mocker):
    model = mocker.Mock()
    model.estimator = "dummy"
    model.predict.return_value = np.array([[7.0], [8.0], [9.0]])

    mocker.patch.object(
        inference_module, "PairwiseFeaturizer", new=DummyPairwiseFeaturizer
    )
    mock_load = mocker.patch.object(
        inference_module,
        "load_anvil_model_and_metadata",
        autospec=True,
        return_value=(
            model,
            DummyPairwiseFeaturizer(),
            SimpleNamespace(tag="PAIR"),
            SimpleNamespace(target_cols=["task0"]),
        ),
    )

    input_df = pd.DataFrame({"SMILES": ["CCO", "CCN"]})
    result = inference_module.predict(
        input_path=input_df,
        input_col="SMILES",
        model_dir="dummy_model",
        write_csv=False,
        output_csv=None,
        debug=False,
        accelerator="cpu",
        log=False,
    )

    pred_col = "OADMET_PRED_PAIR_task0"
    assert pred_col in result.columns
    assert result[pred_col].tolist() == [7.0, 8.0, 9.0]
    mock_load.assert_called_once()


def test_predict_pairwise_multitask_keeps_original_pairs(mocker):
    model = mocker.Mock()
    model.estimator = "dummy"
    model.predict.return_value = np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]])

    mocker.patch.object(
        inference_module, "PairwiseFeaturizer", new=DummyPairwiseFeaturizer
    )
    mock_load = mocker.patch.object(
        inference_module,
        "load_anvil_model_and_metadata",
        autospec=True,
        return_value=(
            model,
            DummyPairwiseFeaturizer(),
            SimpleNamespace(tag="PAIR"),
            SimpleNamespace(target_cols=["task0", "task1"]),
        ),
    )

    result = inference_module.predict(
        input_path=pd.DataFrame({"SMILES": ["CCO", "CCN"]}),
        input_col="SMILES",
        model_dir="dummy_model",
        write_csv=False,
        output_csv=None,
        debug=False,
        accelerator="cpu",
        log=False,
    )

    assert len(result) == 3
    assert result["SMILES"].tolist() == ["CCO - CCO", "CCO - CCN", "CCN - CCN"]
    assert result["OADMET_PRED_PAIR_task0"].tolist() == [1.0, 2.0, 3.0]
    assert result["OADMET_PRED_PAIR_task1"].tolist() == [10.0, 20.0, 30.0]
    assert result["OADMET_STD_PAIR_task0"].isna().all()
    assert result["OADMET_STD_PAIR_task1"].isna().all()
    mock_load.assert_called_once()


def test_predict_pairwise_multimodel_reuses_original_input(mocker):
    featurize_spy = mocker.Mock()
    first_model = mocker.Mock()
    first_model.estimator = "dummy_a"
    first_model.predict.return_value = np.array([[1.0], [2.0], [3.0]])
    second_model = mocker.Mock()
    second_model.estimator = "dummy_b"
    second_model.predict.return_value = np.array([[4.0], [5.0], [6.0]])

    mocker.patch.object(
        inference_module, "PairwiseFeaturizer", new=DummyPairwiseFeaturizer
    )
    mock_load = mocker.patch.object(
        inference_module,
        "load_anvil_model_and_metadata",
        autospec=True,
        side_effect=[
            (
                first_model,
                DummyPairwiseFeaturizer(featurize_spy=featurize_spy),
                SimpleNamespace(tag="PAIRA"),
                SimpleNamespace(target_cols=["task0"]),
            ),
            (
                second_model,
                DummyPairwiseFeaturizer(featurize_spy=featurize_spy),
                SimpleNamespace(tag="PAIRB"),
                SimpleNamespace(target_cols=["task0"]),
            ),
        ],
    )

    result = inference_module.predict(
        input_path=pd.DataFrame({"SMILES": ["CCO", "CCN"]}),
        input_col="SMILES",
        model_dir=["model_a", "model_b"],
        write_csv=False,
        output_csv=None,
        debug=False,
        accelerator="cpu",
        log=False,
    )

    assert len(result) == 3
    assert result["OADMET_PRED_PAIRA_task0"].tolist() == [1.0, 2.0, 3.0]
    assert result["OADMET_PRED_PAIRB_task0"].tolist() == [4.0, 5.0, 6.0]
    assert featurize_spy.call_args_list == [
        mocker.call(["CCO", "CCN"]),
        mocker.call(["CCO", "CCN"]),
    ]
    assert mock_load.call_count == 2
