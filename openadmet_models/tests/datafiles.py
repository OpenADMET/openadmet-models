from importlib import resources
import openadmet_models.test_data

_data_ref = resources.files("openadmet_models.test_data")


anvil_yaml = (
    _data_ref / "basic_anvil.yaml"
).as_posix()