import pytest

from openadmet.models.comparison.compare_base import get_comparison_class
from openadmet.models.comparison.posthoc import PostHocComparison
from openadmet.models.tests.datafiles import combined_json, descr_json, fp_json, multi_task_json


def test_get_comparison_class():
    get_comparison_class("PostHoc")
    with pytest.raises(ValueError):
        get_comparison_class("NotARealClass")


# def test_posthoc_comparison():
#     model_stats = [descr_json, fp_json, combined_json]
#     model_tags = ["descr", "fp", "combined"]
#     comp_obj = PostHocComparison()
#     levene, tukeys_df = comp_obj.compare(model_stats, model_tags)
#     assert levene["mse"][0] == 0.29637389987684526
#     assert levene["ktau"][0] == 0.05033310952264555
#     assert tukeys_df["metric_val"][0] == 0.00705013739584795
#     assert tukeys_df["pvalue"][14] == 1.600273813462394e-08

def test_posthoc_comparison_multitask_reader():
    model_stats = [multi_task_json]
    model_tags = ["multitask"]
    task_tags = ["cyp3a4_pchembl_value_mean"]
    comp_obj = PostHocComparison()
    comp_obj.json_to_df(model_stats, model_tags, task_tags)

