from pathlib import Path

import pandas as pd
import pytest
from click.testing import CliRunner

from openadmet.models.cli import predict as predict_cli_module
from openadmet.models.cli.cli import cli
from openadmet.models.tests.test_utils import click_success
from openadmet.models.tests.unit.datafiles import dummy_null_anvil_yaml


@pytest.fixture
def runner():
    """Provide a Click CliRunner for testing CLI commands in isolation."""
    return CliRunner()


def test_toplevel_runnable(runner):
    """Ensure the top-level 'openadmet' command runs and displays help without error."""
    result = runner.invoke(cli, ["--help"])
    assert click_success(result)


@pytest.mark.parametrize(
    "args", [["anvil", "--help"], ["compare", "--help"], ["predict", "--help"]]
)
def test_subcommand_runnable(runner, args):
    """Verify that all major subcommands (anvil, compare, predict) are registered and runnable."""
    result = runner.invoke(cli, args)
    assert click_success(result)


def test_predict_cli_invokes_inference(tmp_path, runner, null_single_model_dir):
    """
    Validate that the 'predict' subcommand runs inference end-to-end and writes predictions.

    Uses a real NullFeaturizer + DummyRegressorModel fixture so that the CLI path,
    featurization, and prediction logic all execute without mocking any layer.
    """
    input_csv = tmp_path / "input.csv"
    input_csv.write_text("MY_SMILES\nCCO\n")
    output_csv = tmp_path / "predictions.csv"

    result = runner.invoke(
        cli,
        [
            "predict",
            "--input-path",
            str(input_csv),
            "--input-col",
            "MY_SMILES",
            "--model-dir",
            str(null_single_model_dir),
            "--output-csv",
            str(output_csv),
            "--accelerator",
            "cpu",
        ],
    )
    assert click_success(result)
    assert output_csv.exists()
    df = pd.read_csv(output_csv)
    assert "OADMET_PRED_UNIT_task_0" in df.columns
    assert "OADMET_STD_UNIT_task_0" in df.columns


def test_anvil_cli_invokes_workflow(tmp_path, runner):
    """
    Validate that the 'anvil' subcommand runs a lightweight workflow end-to-end.

    Uses the dummy_null_anvil recipe (NullFeaturizer + DummyRegressorModel) so the full
    parse → train → save pipeline executes without mocking any layer.
    """
    output_dir = tmp_path / "anvil_output"

    result = runner.invoke(
        cli,
        [
            "anvil",
            "--recipe-path",
            dummy_null_anvil_yaml,
            "--output-dir",
            str(output_dir),
        ],
    )

    assert click_success(result)
    assert output_dir.exists()
    assert (output_dir / "model.pkl").exists()
    assert (output_dir / "recipe_components").is_dir()


@pytest.mark.parametrize(
    "aq_fxns,beta,best_y,xi,expected",
    [
        (("ucb",), (2.0,), (), (), {"ucb": {"beta": 2.0}}),
        (
            ("ei", "pi"),
            (),
            (1.0, 2.0),
            (0.1, 0.2),
            {"ei": {"xi": 0.1, "best_y": 1.0}, "pi": {"xi": 0.2, "best_y": 2.0}},
        ),
    ],
)
def test_validate_aq_fxns_success(aq_fxns, beta, best_y, xi, expected):
    """
    Verify that valid combinations of acquisition function arguments are correctly parsed into a configuration dict.

    This tests the CLI argument validation logic for active learning parameters.
    """
    assert predict_cli_module._validate_aq_fxns(aq_fxns, beta, best_y, xi) == expected


@pytest.mark.parametrize(
    "aq_fxns,beta,best_y,xi,error_message",
    [
        (("ucb", "ucb"), (1.0, 2.0), (), (), "UCB can only be specified once"),
        (("ei",), (), (), (), "must be specified once per EI and/or PI acquisition"),
        (("ucb",), (), (), (), "Field `beta` must be specified for UCB acquisition"),
    ],
)
def test_validate_aq_fxns_errors(aq_fxns, beta, best_y, xi, error_message):
    """
    Ensure that invalid acquisition function arguments trigger appropriate validation errors.

    This prevents users from running predictions with ambiguous or incomplete active learning settings.
    """
    with pytest.raises(ValueError, match=error_message):
        predict_cli_module._validate_aq_fxns(aq_fxns, beta, best_y, xi)
