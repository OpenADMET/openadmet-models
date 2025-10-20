import pytest
from numpy.testing import assert_almost_equal
import numpy as np
from openadmet.models.comparison.compare_base import get_comparison_class
from openadmet.models.comparison.posthoc import PostHocComparison
from openadmet.models.tests.unit.datafiles import (
    cyp2c9_json,
    cyp1a2_json,
    cyp3a4_json,
    multi_task_json,
    anvil_lgbm_trained_model_dir,
)


def test_get_comparison_class():
    """Test getting comparison class."""
    get_comparison_class("PostHoc")
    with pytest.raises(ValueError):
        get_comparison_class("NotARealClass")


def test_posthoc_fails_on_incorrect_inputs():
    """Test that posthoc comparison fails when given incorrect inputs.

    Inputs include:
    - No inputs
    - Only one of model_stats_fns, labels, or task_names
    - Mismatched lengths of model_stats_fns, labels, and task_names
    - Repeated labels
    - Incorrect labels and task_names for model_stats_fns
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
    """Test that posthoc comparison works when given correct inputs."""
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
    assert_almost_equal(levene["mse"][0], 1.2975061710820235)
    assert_almost_equal(levene["ktau"][0], 0.8835355632672074)
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
    """Test that posthoc comparison can read from anvil-trained model directories and features."""
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
    """Test that posthoc comparison can read multi vs single task from anvil file."""
    model_stats = [multi_task_json, cyp3a4_json]
    model_tags = ["multitask", "single_task"]
    task_tags = ["cyp3a4_pchembl_value_mean", "pchembl_value_mean"]
    comp_obj = PostHocComparison()
    comp_obj.json_to_df(model_stats, model_tags, task_tags)
    levene, tukeys_df = comp_obj.compare(
        model_stats_fns=model_stats, labels=model_tags, task_names=task_tags
    )
    assert levene["mse"][0] == 2.483488460351842
    assert levene["ktau"][0] == 1.0392615736603197
    assert tukeys_df["metric_val"][0] == -0.01037444780666702
    assert tukeys_df["pvalue"][0] == 0.2488307785417857


def test_posthoc_comparison_printing(capsys):
    """Test that posthoc comparison prints results to console."""
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
