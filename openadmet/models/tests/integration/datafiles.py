from importlib import resources

import openadmet.models.tests.unit.test_data  # noqa: F401

_data_ref = resources.files("openadmet.models.tests.integration.test_data")

# CPU

# fingerprint and properties with hparam opt and cross-validation
lgbm_fp_prop_cv = (_data_ref / "lgbm_fp_prop_gridsearch_cv.yaml").as_posix()
# fingerprint only
lgbm_fp_cv = (_data_ref / "lgbm_fp_cv.yaml").as_posix()
# fingerprint and properties with Mordred features and cross-validation with imputation of missing values
lgbm_mordred_cv_impute = (_data_ref / "lgbm_mordred_cv_impute.yaml").as_posix()
# LGBM with properties and scaffold splitting and cross-validation
lgbm_prop_cv = (_data_ref / "lgbm_prop_scaffold_cv.yaml").as_posix()
# single epoch ChemProp with multitask
chemprop_MT_cpu_single = (_data_ref / "chemprop_MT_cpu_single.yaml").as_posix()

# xgboost perimeter split
xgboost_perimeter_cv = (_data_ref / "xgboost_prop_perimeter_cv.yaml").as_posix()

# CatBoost with properties and dissimilarity splitting
catboost_prop_dissimilarity = (_data_ref / "catboost_prop_dissimilarity.yaml").as_posix()

rf_scaffold_cv = (_data_ref / "rf_scaffold_cv.yaml").as_posix()

# GPU

# ChemProp with multitask and cross-validation
chemprop_MT = (_data_ref / "chemprop_MT.yaml").as_posix()
# ChemProp with single task and cross-validation
chemprop_ST = (_data_ref / "chemprop_ST.yaml").as_posix()
# Chemeleon with multitask and cross-validation
chemeleon_MT = (_data_ref / "chemeleon_MT.yaml").as_posix()
# TabPFN with multitask and cross-validation
tabpfn = (_data_ref / "tabpfn.yaml").as_posix()
# MTENN anvil
mtenn_anvil = (_data_ref / "mtenn_anvil.yaml").as_posix()


# poses data
poses_data = (_data_ref / "poses.csv").as_posix()
pdb_folder = (_data_ref / "pose_data").as_posix()
