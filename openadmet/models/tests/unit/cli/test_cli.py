from pathlib import Path

import pytest
from click.testing import CliRunner

from openadmet.models.cli import anvil as anvil_cli_module
from openadmet.models.cli import predict as predict_cli_module
from openadmet.models.cli.cli import cli
from openadmet.models.tests.test_utils import click_success
from openadmet.models.tests.unit.datafiles import basic_anvil_yaml_cv


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


def test_predict_cli_invokes_inference(tmp_path, runner, mocker):
    """
    Validate that the 'predict' subcommand correctly parses arguments and calls the underlying inference function.

    We mock `inference_func` to avoid loading real models (which is heavy and requires trained artifacts).
    This ensures that the CLI layer correctly passes paths, column names, and flags to the logic layer.
    """
    input_csv = tmp_path / "input.csv"
    input_csv.write_text("MY_SMILES\nCCO\n")
    model_dir = tmp_path / "model_dir"
    model_dir.mkdir()

    mock_inference = mocker.patch.object(predict_cli_module, "inference_func")

    result = runner.invoke(
        cli,
        [
            "predict",
            "--input-path",
            input_csv,
            "--input-col",
            "MY_SMILES",
            "--model-dir",
            model_dir,
            "--output-csv",
            tmp_path / "predictions.csv",
            "--accelerator",
            "cpu",
        ],
    )
    assert click_success(result)
    mock_inference.assert_called_once()
    called = mock_inference.call_args.kwargs
    assert called["input_col"] == "MY_SMILES"
    assert called["accelerator"] == "cpu"
    assert called["write_csv"] is True
    assert list(called["model_dir"]) == [model_dir]


def test_anvil_cli_invokes_workflow(tmp_path, runner, mocker):
    """
    Validate that the 'anvil' subcommand correctly initializes and runs a workflow from a recipe.

    We mock the `AnvilSpecification` and workflow execution to verify that the CLI correctly handles
    recipe paths and output directories without actually running a full ML training job.
    """
    mock_spec = mocker.Mock()
    mock_from_recipe = mocker.patch.object(
        anvil_cli_module.AnvilSpecification, "from_recipe", return_value=mock_spec
    )

    result = runner.invoke(
        cli,
        [
            "anvil",
            "--recipe-path",
            basic_anvil_yaml_cv,
            "--output-dir",
            tmp_path / "anvil_output",
        ],
    )

    assert click_success(result)
    mock_from_recipe.assert_called_once_with(basic_anvil_yaml_cv)
    mock_spec.run.assert_called_once()
    called = mock_spec.run.call_args.kwargs
    assert Path(called["output_dir"]) == tmp_path / "anvil_output"
    assert called["debug"] is False


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
