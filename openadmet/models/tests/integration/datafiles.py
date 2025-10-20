from importlib import resources

import openadmet.models.tests.unit.test_data  # noqa: F401

_data_ref = resources.files("openadmet.models.tests.integration.test_data")

# CPU

# Fingerprint and properties with hparam opt and cross-validation
lgbm_fp_prop_cv = (_data_ref / "lgbm_fp_prop_gridsearch_cv.yaml").as_posix()

# Fingerprint only
lgbm_fp_cv = (_data_ref / "lgbm_fp_cv.yaml").as_posix()

# Fingerprint and properties with Mordred features and cross-validation with imputation of missing values
lgbm_mordred_cv_impute = (_data_ref / "lgbm_mordred_cv_impute.yaml").as_posix()

# LGBM with properties and scaffold splitting and cross-validation
lgbm_prop_cv = (_data_ref / "lgbm_prop_scaffold_cv.yaml").as_posix()

# LGBM model ensemble
lgbm_fp_ensemble = (_data_ref / "lgbm_fp_ensemble.yaml").as_posix()

# Single epoch ChemProp with multitask
chemprop_MT_cpu_single = (_data_ref / "chemprop_MT_cpu_single.yaml").as_posix()

# XGBoost perimeter split
xgboost_perimeter_cv = (_data_ref / "xgboost_prop_perimeter_cv.yaml").as_posix()

# CatBoost with properties and dissimilarity splitting
catboost_prop_dissimilarity = (
    _data_ref / "catboost_prop_dissimilarity.yaml"
).as_posix()

rf_scaffold_cv = (_data_ref / "rf_scaffold_cv.yaml").as_posix()

# Dummy models
dummy_fp = (_data_ref / "dummy_fp.yaml").as_posix()

# ChemProp finetuning on AChE data
chemprop_AChE_finetune = (_data_ref / "chemprop_AChE_finetune.yaml").as_posix()
chemprop_AChE_finetune_ensemble = (
    _data_ref / "chemprop_AChE_finetune_ensemble.yaml"
).as_posix()

# Nepare fingerprint model
nepare_fp = (_data_ref / "nepare_fp.yaml").as_posix()
# Cross-validation files for posthoc comparison testing
cv_metrics_lgbm_fp = (_data_ref / "cross_validation_metrics_fp.json").as_posix()
cv_metrics_lgbm_descr = (_data_ref / "cross_validation_metrics_descr.json").as_posix()
cv_metrics_lgbm_combined = (
    _data_ref / "cross_validation_metrics_combined.json"
).as_posix()

# Train/test split recipes
lgbm_fp_cv_train_test = (_data_ref / "lgbm_fp_cv_train_test.yaml").as_posix()
chemprop_MT_cpu_single_train_test = (
    _data_ref / "chemprop_MT_cpu_single_train_test.yaml"
).as_posix()

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
# Chemeleon ensemble
chemeleon_MT_ensemble = (_data_ref / "chemeleon_MT_ensemble.yaml").as_posix()

# Poses data
poses_data = (_data_ref / "poses.csv").as_posix()
pdb_folder = (_data_ref / "pose_data").as_posix()
