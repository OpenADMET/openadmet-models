from pathlib import Path
import pytest
from openadmet_models.anvil.anvil_workflow import AnvilWorkflow, AnvilSpecification, EvalSpec
from openadmet_models.tests.datafiles import (
    basic_anvil_yaml,
    anvil_yaml_featconcat,
    anvil_yaml_gridsearch,
)


def all_anvil_full_recipes():
    return  [basic_anvil_yaml, anvil_yaml_featconcat, anvil_yaml_gridsearch]



def test_anvil_spec_create():
    anvil_spec = AnvilSpecification.from_recipe(basic_anvil_yaml)
    assert anvil_spec

def test_anvil_spec_create_from_recipe_roundtrip(tmp_path):
    anvil_spec = AnvilSpecification.from_recipe(basic_anvil_yaml)
    assert anvil_spec
    anvil_spec.to_recipe(tmp_path/"tst.yaml")
    anvil_spec2 = AnvilSpecification.from_recipe(tmp_path/"tst.yaml")
    # these were created from different directories, so the anvil_dir will be different
    anvil_spec.data.anvil_dir = None
    anvil_spec2.data.anvil_dir = None

    assert anvil_spec == anvil_spec2



def test_anvil_spec_create_to_workflow():
    anvil_spec = AnvilSpecification.from_recipe(basic_anvil_yaml)
    anvil_workflow = anvil_spec.to_workflow()
    assert anvil_workflow

@pytest.mark.parametrize("anvil_full_recipie", all_anvil_full_recipes())
def test_anvil_workflow_run(tmp_path, anvil_full_recipie):
    anvil_workflow = AnvilSpecification.from_recipe(anvil_full_recipie).to_workflow()
    anvil_workflow.run(output_dir=tmp_path / "tst")
    assert Path(tmp_path / "tst" / "model.json").exists()
    assert Path(tmp_path / "tst" / "regression_metrics.json").exists()
    assert Path(tmp_path / "tst" / "regplot.png").exists()


def test_anvil_multiyaml(tmp_path):
    anvil_spec = AnvilSpecification.from_recipe(basic_anvil_yaml)
    anvil_spec.to_multi_yaml(metadata_yaml=tmp_path/"metadata.yaml", procedure_yaml=tmp_path/"procedure.yaml", data_yaml=tmp_path/"data.yaml", report_yaml=tmp_path/"eval.yaml")
    anvil_spec2 = AnvilSpecification.from_multi_yaml(metadata_yaml=tmp_path/"metadata.yaml", procedure_yaml=tmp_path/"procedure.yaml", data_yaml=tmp_path/"data.yaml", report_yaml=tmp_path/"eval.yaml")
    assert anvil_spec.data.anvil_dir == anvil_spec2.data.anvil_dir
    assert anvil_spec.dict() == anvil_spec2.dict()


