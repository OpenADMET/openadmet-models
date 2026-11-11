import numpy as np
import pytest
from numpy.testing import assert_almost_equal

from openadmet.models.comparison.compare_base import get_comparison_class
from openadmet.models.comparison.posthoc import PostHocComparison
from openadmet.models.tests.unit.datafiles import (
    anvil_lgbm_trained_model_dir,
    cyp1a2_json,
    cyp2c9_json,
    cyp3a4_json,
    multi_task_json,
)


def test_get_comparison_class():
    """
    Test dynamic retrieval of comparison classes from the registry.

    Verifies that valid class names return the class and invalid names raise ValueError.
    """
    get_comparison_class("PostHoc")
    with pytest.raises(ValueError):
        get_comparison_class("NotARealClass")


def test_posthoc_fails_on_incorrect_inputs():
    """
    Test that posthoc comparison fails when given incorrect inputs.

    Inputs include:
    - No inputs
    - Only one of model_stats_fns, labels, or task_names
    - Mismatched lengths of model_stats_fns, labels, and task_names
    - Repeated labels
    - Incorrect labels and task_names for model_stats_fns

    This validation is critical to ensure that comparison tables and plots match models to their correct metadata.
    """
    comp_obj = PostHocComparison()
    with pytest.raises(ValueError):
        comp_obj.compare()
    with pytest.raises(ValueError):
        comp_obj.compare(model_stats_fns=[cyp2c9_json])
    with pytest.raises(ValueError):
        comp_obj.compare(labels=["model1"])
    with pytest.raises(ValueError):
        comp_obj.compare(task_names=["task1"])
    with pytest.raises(ValueError):
        comp_obj.compare(model_stats_fns=[cyp2c9_json], labels=["model1"])
    with pytest.raises(ValueError):
        comp_obj.compare(model_stats_fns=[cyp2c9_json], task_names=["task1"])
    with pytest.raises(ValueError):
        comp_obj.compare(labels=["model1"], task_names=["task1"])
    with pytest.raises(ValueError):
        comp_obj.compare(
            model_stats_fns=[cyp2c9_json, cyp3a4_json],
            labels=["model1"],
            task_names=["task1", "task2", "task3"],
        )


def test_posthoc_repeat_label_error():
    """Test that posthoc comparison fails when given multiple repeated labels."""
    model_stats = [cyp2c9_json, cyp3a4_json, cyp1a2_json]
    model_tags = [
        "openadmet-CYP2C9-pchembl-regression-testing-cv",
        "openadmet-CYP2C9-pchembl-regression-testing-cv",
        "openadmet-CYP1A2-pchembl-regression-testing-cv",
    ]
    task_tags = ["pchembl_value_mean"] * 3
    comp_obj = PostHocComparison()
    with pytest.raises(ValueError):
        comp_obj.compare(
            model_stats_fns=model_stats, labels=model_tags, task_names=task_tags
        )


def test_posthoc_comparison():
    """
    Test that posthoc comparison works correctly when given valid inputs.

    This verifies the calculation of statistical tests (Levene's test for equality of variances,
    Tukey's HSD for pairwise mean differences) based on loaded model metrics.
    """
    model_stats = [cyp2c9_json, cyp3a4_json, cyp1a2_json]
    model_tags = [
        "openadmet-CYP2C9-pchembl-regression-testing-cv",
        "openadmet-CYP3A4-pchembl-regression-testing-cv",
        "openadmet-CYP1A2-pchembl-regression-testing-cv",
    ]
    task_tags = ["pchembl_value_mean"] * 3
    comp_obj = PostHocComparison()
    levene, tukeys_df = comp_obj.compare(
        model_stats_fns=model_stats, labels=model_tags, task_names=task_tags
    )
    assert_almost_equal(levene["mse"]["stat"], 1.2975061710820235)
    assert_almost_equal(levene["ktau"]["stat"], 0.8835355632672074)
    assert_almost_equal(tukeys_df["metric_val"][0], 0.10937620875054632)
    assert_almost_equal(tukeys_df["pvalue"][14], 0.00321143)


@pytest.mark.parametrize(
    "label_types,expected_labels",
    [
        (["biotarget", "model", "tasks"], ["CYP3A4_LGBM_ST"]),
        (["feat"], ["mordred+ecfp:6"]),
    ],
)
def test_posthoc_comparison_anvil_reader_and_feature_label(
    label_types, expected_labels
):
    """
    Test that posthoc comparison can automatically extract labels from anvil-trained model directories.

    This ensures that metadata stored in `metadata.yaml` within model directories can be correctly
    parsed to generate readable labels for comparison plots.
    """
    comp_obj = PostHocComparison()
    model_stats_fns, labels, task_names = comp_obj.label_and_task_name_from_anvil(
        model_dirs=[anvil_lgbm_trained_model_dir], label_types=label_types
    )
    assert labels == expected_labels


@pytest.mark.parametrize(
    "label_types",
    [
        (["bad_label"]),
    ],
)
def test_posthoc_comparison_json_reader_fails(label_types):
    """Test that posthoc comparison fails when given incorrect label type."""
    comp_obj = PostHocComparison()
    with pytest.raises(ValueError):
        comp_obj.label_and_task_name_from_anvil(
            model_dirs=[anvil_lgbm_trained_model_dir], label_types=label_types
        )


def test_posthoc_comparison_json_reader():
    """
    Test that posthoc comparison handles both multi-task and single-task JSON result files.

    This verifies that the system can normalize results from different task types into a common
    format for statistical comparison.
    """
    model_stats = [multi_task_json, cyp3a4_json]
    model_tags = ["multitask", "single_task"]
    task_tags = ["cyp3a4_pchembl_value_mean", "pchembl_value_mean"]
    comp_obj = PostHocComparison()
    comp_obj.json_to_df(model_stats, model_tags, task_tags)
    levene, tukeys_df = comp_obj.compare(
        model_stats_fns=model_stats, labels=model_tags, task_names=task_tags
    )
    assert np.allclose(levene["mse"]["stat"], 2.483488460351842)
    assert np.allclose(levene["ktau"]["stat"], 1.0392615736603197)
    assert np.allclose(tukeys_df["metric_val"][0], -0.01037444780666702)
    assert np.allclose(tukeys_df["pvalue"][0], 0.2488307785417857)


def test_posthoc_comparison_printing(capsys):
    """
    Test that posthoc comparison prints results to console in a readable format.

    We capture stdout to verify that Levene's test and Tukey's HSD results are actually displayed to the user.
    """
    model_stats = [cyp2c9_json, cyp3a4_json, cyp1a2_json]
    model_tags = [
        "openadmet-CYP2C9-pchembl-regression-testing-cv",
        "openadmet-CYP3A4-pchembl-regression-testing-cv",
        "openadmet-CYP1A2-pchembl-regression-testing-cv",
    ]
    task_tags = ["pchembl_value_mean"] * 3
    comp_obj = PostHocComparison()
    levene, tukeys_df = comp_obj.compare(
        model_stats_fns=model_stats, labels=model_tags, task_names=task_tags
    )
    captured = capsys.readouterr()
    assert "Levene's test results" in captured.out
    assert "Tukey's HSD results" in captured.out
    assert "0.109376" in captured.out
    assert "0.304913" in captured.out
