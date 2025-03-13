from importlib import resources

import openadmet_models.tests.test_data  # noqa: F401

_data_ref = resources.files("openadmet_models.tests.test_data")


basic_anvil_yaml = (_data_ref / "basic_anvil.yaml").as_posix()
anvil_yaml_gridsearch = (_data_ref / "anvil_gridsearch.yaml").as_posix()
anvil_yaml_featconcat = (_data_ref / "anvil_featconcat.yaml").as_posix()
basic_anvil_yaml_cv = (_data_ref / "basic_anvil_cv.yaml").as_posix()

# individual sections for multi-yaml
metadata_yaml = (_data_ref / "basic_anvil_metadata.yaml").as_posix()
procedure_yaml = (_data_ref / "basic_anvil_procedure.yaml").as_posix()
data_yaml = (_data_ref / "basic_anvil_data.yaml").as_posix()
eval_yaml = (_data_ref / "basic_anvil_eval.yaml").as_posix()


# data for testing
intake_cat = (_data_ref / "example_intake.yaml").as_posix()
test_csv = (_data_ref / "test_data.csv").as_posix()
CYP3A4_chembl_pchembl = (_data_ref / "CYP3A4_chembl_pchembl.csv").as_posix()
AChE_CHEMBL4078_Landrum_maxcur = (
    _data_ref / "AChE_CHEMBL4078_Landrum_maxcur.csv"
).as_posix()

# posthoc anvil outputs for testing
descr_json = (_data_ref / "cross_validation_metrics_descr.json")
fp_json = (_data_ref / "cross_validation_metrics_fp.json")
combined_json = (_data_ref / "cross_validation_metrics_combined.json")
