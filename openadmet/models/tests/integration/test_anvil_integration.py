from openadmet.models.cli.cli import cli
from openadmet.models.tests.test_utils import click_success
import pytest
from click.testing import CliRunner

from openadmet.models.tests.integration.datafiles import lgbm_fp_prop_cv, lgbm_fp_cv, lgbm_prop_cv



class TestCPUAnvilConfigs:

    @pytest.mark.cpu
    @pytest.mark.parametrize("recipe_file", [
        lgbm_fp_cv,
        lgbm_fp_prop_cv,
        lgbm_prop_cv
    ])
    def test_configs(self, recipe_file, tmp_path):
        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "anvil",
                "--recipe-path",
                recipe_file,
                "--output-dir",
                tmp_path / "output",
            ]
        )
        assert click_success(result)



def test_cuda_available():
    """Check if CUDA is available for GPU tests"""
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


class TestGPUAnvilConfigs:

    @pytest.mark.gpu
    @pytest.mark.skipif(not test_cuda_available(), reason="CUDA not available")
    @pytest.mark.parametrize("recipe_file", [
        None
    ])
    def test_gpu_configs(self, recipe_file, tmp_path):

        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "anvil",
                "--recipe-path",
                recipe_file,
                "--output-dir",
                tmp_path / "output",
            ]
        )
        assert click_success(result)