import pytest
from openadmet_models.anvil.anvil_workflow import *
from openadmet_models.tests.datafiles import basic_anvil_yaml




def test_anvil_workflow_create():
    anvil_workflow = AnvilWorkflow.from_yaml(basic_anvil_yaml)
    print(anvil_workflow)
    raise Exception("Not implemented")



