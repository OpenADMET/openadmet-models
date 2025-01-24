from importlib import resources
import openadmet_models.tests.test_data

_data_ref = resources.files("openadmet_models.tests.test_data")


basic_anvil_yaml = (
    _data_ref / "basic_anvil.yaml"
).as_posix()