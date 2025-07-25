from openadmet.models.cli.cli import cli
from openadmet.models.tests.test_utils import click_success
import pytest
from click.testing import CliRunner

from openadmet.data. import lgbm_fp_prop_cv, lgbm_fp_cv, lgbm_prop_cv



@pytest.mark.parametrize("recipe_file", [
    lgbm_fp_cv,
    lgbm_fp_prop_cv,
    lgbm_prop_cv
])
class TestLGBM:
    def test_lgbm_configurations(self, config_file):
        runner = CliRunner()
        result = runner.invoke(
            cli,
                [
                    "predict",
                    "--recipe-path",
                    config_file,
                ]
        )
        assert click_success(result)