import pytest
from numpy.testing import assert_almost_equal
import numpy as np
from openadmet.models.comparison.compare_base import get_comparison_class
from openadmet.models.comparison.posthoc import PostHocComparison
from openadmet.models.tests.unit.datafiles import cyp2c9_json, cyp1a2_json, cyp3a4_json, multi_task_json


def test_get_comparison_class():
    get_comparison_class("PostHoc")
    with pytest.raises(ValueError):
        get_comparison_class("NotARealClass")


def test_posthoc_comparison():
    model_stats = [cyp2c9_json, cyp3a4_json, cyp1a2_json]
    model_tags = ["openadmet-CYP2C9-pchembl-regression-testing-cv", "openadmet-CYP3A4-pchembl-regression-testing-cv", "openadmet-CYP1A2-pchembl-regression-testing-cv"]
    task_tags = ["pchembl_value_mean"]*3
    comp_obj = PostHocComparison()
    levene, tukeys_df = comp_obj.compare(model_stats, model_tags, task_tags)
    assert_almost_equal(levene["mse"][0], 1.2975061710820235)
    assert_almost_equal(levene["ktau"][0], 0.8835355632672074)
    assert_almost_equal(tukeys_df["metric_val"][0], 0.10937620875054632)
    assert_almost_equal(tukeys_df["pvalue"][14], 0.00321143)

def test_posthoc_comparison_multitask_reader():
    model_stats = [multi_task_json, cyp3a4_json]
    model_tags = ["multitask", "single_task"]
    task_tags = ["cyp3a4_pchembl_value_mean", "pchembl_value_mean"]
    comp_obj = PostHocComparison()
    comp_obj.json_to_df(model_stats, model_tags, task_tags)
    levene, tukeys_df = comp_obj.compare(model_stats, model_tags, task_tags)
    assert levene["mse"][0] == 2.483488460351842
    assert levene["ktau"][0] == 1.0392615736603197
    assert tukeys_df["metric_val"][0] == -0.01037444780666702
    assert tukeys_df["pvalue"][0] == 0.2488307785417857

def test_posthoc_comparison_printing(capsys):
    model_stats = [cyp2c9_json, cyp3a4_json, cyp1a2_json]
    model_tags = ["openadmet-CYP2C9-pchembl-regression-testing-cv", "openadmet-CYP3A4-pchembl-regression-testing-cv", "openadmet-CYP1A2-pchembl-regression-testing-cv"]
    task_tags = ["pchembl_value_mean"]*3
    comp_obj = PostHocComparison()
    levene, tukeys_df = comp_obj.compare(model_stats, model_tags, task_tags)
    captured = capsys.readouterr()
    assert "Levene's test results" in captured.out
    assert "Tukey's HSD results" in captured.out
    assert "0.109376" in captured.out
    assert "0.304913" in captured.out
