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
    pca_fp_lgbm,
    pca_concat_lgbm,
    lgbm_fp_prop_cv,
    lgbm_mordred_cv_impute,
    lgbm_prop_cv,
    rf_scaffold_cv,
    chemprop_MT_cpu_single_train_test,
    tabpfn,
    xgboost_perimeter_cv,
    nepare_fp,
    cv_metrics_lgbm_descr,
    cv_metrics_lgbm_fp,
    cv_metrics_lgbm_combined,
    lgbm_fp_cv_train_test,
)
from openadmet.models.tests.test_utils import click_success


class TestCPUAnvilConfigs:
    @pytest.mark.cpu
    @pytest.mark.parametrize(
        "recipe_file",
        [
            lgbm_fp_cv,
            pca_fp_lgbm,
            pca_concat_lgbm,
            lgbm_fp_prop_cv,
            lgbm_prop_cv,
            lgbm_fp_ensemble,
            chemprop_MT_cpu_single,
            chemprop_MT_cpu_single_train_test,
            xgboost_perimeter_cv,
            lgbm_fp_cv_train_test,
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
            # tabpfn, TABPFN currently broken on this dataset for some reason
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
