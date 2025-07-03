from openadmet.models.cli.cli import cli
from openadmet.models.tests.test_utils import click_success
from openadmet.models.tests.datafiles import (
    anvil_lgbm_trained_model_dir,
    pred_test_data_csv,
)
import pytest
from click.testing import CliRunner



def test_toplevel_runnable():
    """Test the top-level CLI command"""

    runner = CliRunner()
    result = runner.invoke(cli, ["--help"])
    assert click_success(result)

@pytest.mark.parametrize(
    "args",
    [
        ["anvil", "--help"],
        ["compare", "--help"],
        ["predict", "--help"],
    ],
)
def test_subcommand_runnable(args):
    """Test the subcommands"""

    runner = CliRunner()
    result = runner.invoke(cli, args)
    assert click_success(result)


def test_predict_cli(tmp_path):
    """Test the predict CLI command"""

    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "predict",
            "--input-path",
            pred_test_data_csv,
            "--input-col",
            "MY_SMILES",
            "--model-dir",
            anvil_lgbm_trained_model_dir,
            "--output-path",
            tmp_path / "predictions.csv",
        ],
    )
    assert click_success(result)
