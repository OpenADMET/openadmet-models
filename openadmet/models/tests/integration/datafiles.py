from importlib import resources

import openadmet.models.tests.unit.test_data  # noqa: F401

_data_ref = resources.files("openadmet.models.tests.integration.test_data")

# fingerprint and properties with hparam opt and cross-validation
lgbm_fp_prop_cv = (_data_ref / "lgbm/lgbm_fp_prop_gridsearch_cv.yaml").as_posix()
# fingerprint only
lgbm_fp_cv = (_data_ref / "lgbm/lgbm_fp_cv.yaml").as_posix()
# LGBM with properties and scaffold splitting and cross-validation
lgbm_prop_cv = (_data_ref / "lgbm/lgbm_prop_scaffold_cv.yaml").as_posix()
