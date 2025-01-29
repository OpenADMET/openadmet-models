from importlib import resources

import openadmet_models.tests.test_data # noqa: F401

_data_ref = resources.files("openadmet_models.tests.test_data")


basic_anvil_yaml = (_data_ref / "basic_anvil.yaml").as_posix()

intake_cat = (_data_ref / "example_intake.yaml").as_posix()

test_csv = (_data_ref / "test_data.csv").as_posix()
