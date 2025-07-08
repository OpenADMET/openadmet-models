from importlib import resources

import openadmet.models.tests.test_data  # noqa: F401

_data_ref = resources.files("openadmet.models.tests.test_data")


basic_anvil_yaml = (_data_ref / "basic_anvil.yaml").as_posix()
anvil_yaml_gridsearch = (_data_ref / "anvil_gridsearch.yaml").as_posix()
anvil_yaml_featconcat = (_data_ref / "anvil_featconcat.yaml").as_posix()
anvil_yaml_xgboost_cv = (_data_ref / "anvil_xgbm_cv_1_1.yaml").as_posix()

basic_anvil_yaml_cv = (_data_ref / "basic_anvil_cv.yaml").as_posix()
basic_anvil_yaml_classification = (
    _data_ref / "basic_anvil_classification.yaml"
).as_posix()
tabpfn_anvil_classification_yaml = (_data_ref / "tabpfn_ache.yaml").as_posix()

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

ACEH_chembl_pchembl = (_data_ref / "ACEH_chembl_pchembl.csv").as_posix()
acetylcholinesterase_anvil_chemprop_yaml = (_data_ref / "acetylcholinesterase_anvil_chemprop.yaml").as_posix()


# posthoc anvil outputs for testing
cyp2c9_json = (_data_ref / "cross_validation_metrics_2c9.json").as_posix()
cyp3a4_json = (_data_ref / "cross_validation_metrics_3a4.json").as_posix()
cyp1a2_json = (_data_ref / "cross_validation_metrics_1a2.json").as_posix()
multi_task_json = _data_ref / "cross_validation_metrics_multitask.json"

# ligand pose for reading and featurizing in MTENN
ligand_pose = (_data_ref / "7kvh_A.pdb").as_posix()


# stripped down anvil trained model directory for inference testing
anvil_lgbm_trained_model_dir = (_data_ref / "cyp3a4_anvil_lgbm_model_dir").as_posix()
anvil_chemprop_trained_model_dir = (_data_ref / "aceh_chemprop_anvil_model_dir").as_posix()
# test data for prediction
pred_test_data_csv = (_data_ref / "ligands.csv").as_posix()
