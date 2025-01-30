from openadmet_models.anvil.anvil_workflow import AnvilWorkflow
from openadmet_models.tests.datafiles import basic_anvil_yaml


def test_anvil_workflow_create():
    anvil_workflow = AnvilWorkflow.from_yaml(basic_anvil_yaml)
    assert anvil_workflow


def test_anvil_workflow_run(tmp_path):
    anvil_workflow = AnvilWorkflow.from_yaml(basic_anvil_yaml)
    anvil_workflow.run(output_dir="tst_output")
    


