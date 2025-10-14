import numpy as np
import pandas as pd
import pytest
from click.testing import CliRunner

from openadmet.models.cli.cli import cli
from openadmet.models.tests.integration.datafiles import (
    catboost_prop_dissimilarity,
    chemeleon_MT,
    chemeleon_MT_ensemble,
    chemprop_AChE_finetune,
    chemprop_AChE_finetune_ensemble,
    chemprop_MT,
    chemprop_MT_cpu_single,
    chemprop_ST,
    dummy_fp,
    lgbm_fp_cv,
    lgbm_fp_ensemble,
    lgbm_fp_prop_cv,
    lgbm_mordred_cv_impute,
    lgbm_prop_cv,
    mtenn_anvil,
    pdb_folder,
    rf_scaffold_cv,
    tabpfn,
    xgboost_perimeter_cv,
    nepare_fp,
    cv_metrics_lgbm_descr,
    cv_metrics_lgbm_fp,
    cv_metrics_lgbm_combined,
)
from openadmet.models.tests.test_utils import click_success


class TestCPUAnvilConfigs:
    @pytest.mark.cpu
    @pytest.mark.parametrize(
        "recipe_file",
        [
            lgbm_fp_cv,
            lgbm_fp_prop_cv,
            lgbm_prop_cv,
            lgbm_fp_ensemble,
            chemprop_MT_cpu_single,
            xgboost_perimeter_cv,
            catboost_prop_dissimilarity,
            lgbm_mordred_cv_impute,
            rf_scaffold_cv,
            dummy_fp,
            chemprop_AChE_finetune,
            chemprop_AChE_finetune_ensemble,
            nepare_fp,
        ],
    )
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
            ],
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
    @pytest.mark.parametrize(
        "recipe_file",
        [
            chemprop_MT,
            chemprop_ST,
            chemeleon_MT,
            chemeleon_MT_ensemble,
            tabpfn,
        ],
    )
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
            ],
        )
        assert click_success(result)


class TestStructuralModelGPUAnvilConfigs:
    """
    Test for MTENN Anvil configuration with GPU support

    We need to generate the poses data dynamically for the test, as the original
    mtenn_anvil.yaml file expects a CSV with poses and a folder with PDB files in a static location.
    The test will create a temporary folder with PDB files and a CSV file pointing to
    those files. The MTENN model will then be tested with this data.
    """

    @pytest.mark.gpu
    @pytest.mark.skipif(not test_cuda_available(), reason="CUDA not available")
    @pytest.mark.parametrize("recipe_file, pdb_folder", [(mtenn_anvil, pdb_folder)])
    def test_mtenn_anvil(self, recipe_file, pdb_folder, tmp_path):
        # glob the pdb files recursively from the pdb_folder
        pdb_files = list((tmp_path / pdb_folder).glob("**/*.pdb"))
        # create a CSV file with the pdb files
        poses_df = pd.DataFrame(
            {"poses": [pdb_file.as_posix() for pdb_file in pdb_files]}
        )
        # add a dummy target column with random floats
        poses_df["y"] = np.random.rand()

        # save the CSV file to the temporary path
        poses_csv = tmp_path / "poses.csv"
        poses_df.to_csv(poses_csv, index=False)

        # replace the resource in the recipe file with the temporary CSV file
        with open(recipe_file) as file:
            recipe_content = file.read()
        recipe_content = recipe_content.replace(
            "{{ANVIL_DIR}}/poses.csv", poses_csv.as_posix()
        )

        # write the modified recipe content to a new file
        recipe_file = tmp_path / "mtenn_anvil.yaml"
        with open(recipe_file, "w") as file:
            file.write(recipe_content)

        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "anvil",
                "--recipe-path",
                recipe_file,
                "--output-dir",
                tmp_path / "output",
            ],
        )
        assert click_success(result)


class TestCPUPosthocConfigs:
    @pytest.mark.cpu
    def test_compare_all_cv_metrics(self, tmp_path):
        runner = CliRunner()
        cv_metrics_files = [
            cv_metrics_lgbm_fp,
            cv_metrics_lgbm_descr,
            cv_metrics_lgbm_combined,
        ]
        labels = [
            "LGBM_FP",
            "LGBM_DESCR",
            "LGBM_COMBINED",
        ]
        task_names = [
            "PXR_induction_DRC_summary_octant_in-house_pure: pEC50_estimate (-log10(molarity))",
            "PXR_induction_DRC_summary_octant_in-house_pure: pEC50_estimate (-log10(molarity))",
            "PXR_induction_DRC_summary_octant_in-house_pure: pEC50_estimate (-log10(molarity))",
        ]
        output_dir = tmp_path / "output"
        output_dir.mkdir(parents=True, exist_ok=True)  # <-- Ensure directory exists

        # Repeat each tag before each argument
        cli_args = ["compare"]
        for f in cv_metrics_files:
            cli_args.extend(["--model-stats-fns", f])
        for l in labels:
            cli_args.extend(["--labels", l])
        for t in task_names:
            cli_args.extend(["--task-names", t])
        cli_args.extend(["--output-dir", str(output_dir)])

        result = runner.invoke(cli, cli_args)
        assert click_success(result)
