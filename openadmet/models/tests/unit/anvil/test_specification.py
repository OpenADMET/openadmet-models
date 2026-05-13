from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from openadmet.models.anvil.specification import (
    AnvilSpecification,
    DataSpec,
    EnsembleSpec,
    EvalSpec,
    FeatureSpec,
    Metadata,
    ModelSpec,
    ProcedureSpec,
    ReportSpec,
    SplitSpec,
    TrainerSpec,
    TransformSpec,
)
from openadmet.models.anvil.workflow import AnvilDeepLearningWorkflow, AnvilWorkflow
from openadmet.models.architecture.model_base import LightningModelBase

# --- DataSpec Tests ---


def test_dataspec_resource_and_train_test_mutually_exclusive():
    """Test that specifying both resource and train_resource raises ValueError."""
    with pytest.raises(ValueError, match="Specify either `resource` or"):
        DataSpec(
            type="csv",
            input_col="smiles",
            target_cols="target",
            resource="data.csv",
            train_resource="train.csv",
        )


def test_dataspec_requires_train_and_test_together():
    """Test that specifying train_resource without test_resource raises ValueError."""
    with pytest.raises(ValueError, match="must both be specified"):
        DataSpec(
            type="csv",
            input_col="smiles",
            target_cols="target",
            train_resource="train.csv",
        )


def test_dataspec_target_cols_string_normalized_to_list():
    """Test that a string target_cols is converted to a list."""
    spec = DataSpec(
        type="csv",
        input_col="smiles",
        target_cols="activity",
        resource="data.csv",
    )
    assert spec.target_cols == ["activity"]


def test_dataspec_template_anvil_dir_replaces_placeholder(tmp_path):
    """Test that {{ ANVIL_DIR }} is replaced in resource path."""
    spec = DataSpec(
        type="csv",
        input_col="smiles",
        target_cols="target",
        resource="{{ ANVIL_DIR }}/data.csv",
        anvil_dir="/tmp/mydir",
    )
    # The validator runs automatically if anvil_dir is set at init
    assert spec.resource == "/tmp/mydir/data.csv"

    # Test explicit method call
    spec2 = DataSpec(
        type="csv",
        input_col="smiles",
        target_cols="target",
        resource="{{ ANVIL_DIR }}/data.csv",
    )
    spec2.template_anvil_dir(Path("/other/dir"))
    assert spec2.resource == "/other/dir/data.csv"


def test_dataspec_read_single_resource_csv(tmp_path):
    """Test reading a single CSV resource."""
    csv_path = tmp_path / "data.csv"
    df = pd.DataFrame(
        {
            "smiles": ["CCO", "CC(C)O", "c1ccccc1"],
            "target": [1.0, 2.0, 3.0],
            "extra": ["a", "b", "c"],
        }
    )
    df.to_csv(csv_path, index=False)

    spec = DataSpec(
        type="csv",
        input_col="smiles",
        target_cols="target",
        resource=str(csv_path),
    )
    X, y = spec.read()

    assert isinstance(X, pd.Series)
    assert isinstance(y, pd.DataFrame)
    assert len(X) == 3
    assert len(y) == 3
    assert list(y.columns) == ["target"]
    assert X.iloc[0] == "CCO"
    assert y.iloc[0, 0] == 1.0


def test_dataspec_read_single_resource_dropna(tmp_path):
    """Test that rows with NaNs in target columns are dropped."""
    csv_path = tmp_path / "data_nan.csv"
    df = pd.DataFrame(
        {
            "smiles": ["CCO", "CC(C)O", "c1ccccc1", "C"],
            "target": [1.0, np.nan, 3.0, 4.0],
        }
    )
    df.to_csv(csv_path, index=False)

    spec = DataSpec(
        type="csv",
        input_col="smiles",
        target_cols="target",
        resource=str(csv_path),
        dropna=True,
    )
    X, y = spec.read()

    assert len(X) == 3
    assert len(y) == 3
    assert "CC(C)O" not in X.values


def test_dataspec_read_train_test_val_returns_correct_splits(tmp_path):
    """Test reading separate train, test, and val resources."""
    train_path = tmp_path / "train.csv"
    test_path = tmp_path / "test.csv"
    val_path = tmp_path / "val.csv"

    pd.DataFrame({"smiles": ["A", "B", "C"], "target": [1, 2, 3]}).to_csv(
        train_path, index=False
    )

    pd.DataFrame({"smiles": ["D", "E"], "target": [4, 5]}).to_csv(
        test_path, index=False
    )

    pd.DataFrame({"smiles": ["F"], "target": [6]}).to_csv(val_path, index=False)

    spec = DataSpec(
        type="csv",
        input_col="smiles",
        target_cols="target",
        train_resource=str(train_path),
        test_resource=str(test_path),
        val_resource=str(val_path),
    )

    # Returns: X_train, X_val, X_test, y_train, y_val, y_test, X, y
    X_train, X_val, X_test, y_train, y_val, y_test, X, y = spec.read()

    assert len(X_train) == 3
    assert len(X_test) == 2
    assert len(X_val) == 1
    assert len(X) == 6
    assert len(y) == 6

    # Verify content
    assert X_train.tolist() == ["A", "B", "C"]
    assert X_test.tolist() == ["D", "E"]
    assert X_val.tolist() == ["F"]


def test_dataspec_read_train_test_raises_on_split_column_in_file(tmp_path):
    """Test that a ValueError is raised if input files contain a '_split' column."""
    train_path = tmp_path / "train_bad.csv"
    test_path = tmp_path / "test_bad.csv"

    pd.DataFrame({"smiles": ["A"], "target": [1], "_split": ["train"]}).to_csv(
        train_path, index=False
    )

    pd.DataFrame({"smiles": ["B"], "target": [2]}).to_csv(test_path, index=False)

    spec = DataSpec(
        type="csv",
        input_col="smiles",
        target_cols="target",
        train_resource=str(train_path),
        test_resource=str(test_path),
    )

    with pytest.raises(ValueError, match="should not contain a '_split' column"):
        spec.read()


def test_dataspec_to_yaml_from_yaml_roundtrip(tmp_path):
    """Test roundtrip YAML serialization for DataSpec."""
    spec = DataSpec(
        type="csv",
        input_col="smiles",
        target_cols=["target1", "target2"],
        resource="data.csv",
        dropna=True,
    )
    yaml_path = tmp_path / "spec.yaml"
    spec.to_yaml(yaml_path)

    loaded_spec = DataSpec.from_yaml(yaml_path)
    assert loaded_spec.input_col == spec.input_col
    assert loaded_spec.target_cols == spec.target_cols
    assert loaded_spec.resource == spec.resource
    assert loaded_spec.dropna == spec.dropna


# --- Metadata Tests ---


def test_metadata_to_yaml_from_yaml_roundtrip(tmp_path):
    """Test roundtrip YAML serialization for Metadata."""
    meta = Metadata(
        version="v1",
        name="test-workflow",
        build_number=1,
        description="A test workflow",
        tag="v1.0.0",
        authors="Test Author",
        email="test@example.com",
        biotargets=["TargetA"],
        tags=["tag1", "tag2"],
    )
    yaml_path = tmp_path / "metadata.yaml"
    meta.to_yaml(yaml_path)

    loaded_meta = Metadata.from_yaml(yaml_path)
    assert loaded_meta.name == meta.name
    assert loaded_meta.biotargets == meta.biotargets
    assert loaded_meta.version == "v1"


# --- AnvilSection Tests ---


def test_anvilsection_to_class_dispatches_correctly():
    """Test that to_class returns the correct class instance."""
    # Using SplitSpec as a concrete example
    spec = SplitSpec(
        type="ShuffleSplitter", params={"train_size": 0.8, "test_size": 0.2}
    )
    splitter = spec.to_class()
    # Check if it has the attributes we expect from a splitter
    assert hasattr(splitter, "split")
    assert splitter.train_size == 0.8


# --- ModelSpec Tests ---


def test_modelspec_path_pairs_validation():
    """Test validation of param_path and serial_path pairs."""
    # Success cases
    ModelSpec(type="MyModel", param_path="p.pt", serial_path="s.pt")
    ModelSpec(type="MyModel")

    # Failure cases
    with pytest.raises(ValueError, match="must be provided together"):
        ModelSpec(type="MyModel", param_path="p.pt")

    with pytest.raises(ValueError, match="must be provided together"):
        ModelSpec(type="MyModel", serial_path="s.pt")


# --- EnsembleSpec Tests ---


def test_ensemblespec_n_models_minimum():
    """Test validation of n_models."""
    with pytest.raises(ValueError, match="Ensemble must have more than one model"):
        EnsembleSpec(type="Ensemble", n_models=1)

    EnsembleSpec(type="Ensemble", n_models=2)


def test_ensemblespec_path_count_validation():
    """Test validation of param_paths and serial_paths lengths."""
    # Length mismatch between paths
    with pytest.raises(ValueError, match="same length"):
        EnsembleSpec(
            type="Ensemble", n_models=2, param_paths=["p1", "p2"], serial_paths=["s1"]
        )

    # Length mismatch with n_models
    with pytest.raises(ValueError, match="match the number of models"):
        EnsembleSpec(
            type="Ensemble",
            n_models=3,
            param_paths=["p1", "p2"],
            serial_paths=["s1", "s2"],
        )

    # Success
    EnsembleSpec(
        type="Ensemble", n_models=2, param_paths=["p1", "p2"], serial_paths=["s1", "s2"]
    )


# --- AnvilSpecification Tests ---


def test_anvilspecification_from_recipe_resolves_anvil_dir(tmp_path):
    """Test that loading from a recipe resolves {{ ANVIL_DIR }}."""
    workflow_dir = tmp_path / "myworkflow"
    workflow_dir.mkdir()
    recipe_path = workflow_dir / "recipe.yaml"

    # Create minimal valid YAML
    recipe_content = {
        "metadata": {
            "version": "v1",
            "name": "test",
            "build_number": 0,
            "description": "d",
            "tag": "t",
            "authors": "a",
            "email": "a@b.com",
            "biotargets": [],
            "tags": [],
        },
        "data": {
            "type": "csv",
            "resource": "{{ ANVIL_DIR }}/data.csv",
            "input_col": "s",
            "target_cols": "t",
        },
        "procedure": {
            "split": {"type": "RandomSplitter"},
            "feat": {"type": "FingerprintFeaturizer"},
            "model": {"type": "LGBMRegressorModel"},
            "train": {"type": "SKLearnBasicTrainer"},
        },
        "report": {"eval": []},
    }

    with open(recipe_path, "w") as f:
        yaml.dump(recipe_content, f)

    spec = AnvilSpecification.from_recipe(recipe_path)
    # The resolved path should contain the temp dir path (fsspec adds file:// scheme)
    expected_path = (workflow_dir / "data.csv").as_uri()
    assert spec.data.resource == expected_path


def test_anvilspecification_to_multi_yaml_from_multi_yaml_roundtrip(tmp_path):
    """Test splitting spec into multiple YAMLs and reloading."""
    meta = Metadata(
        version="v1",
        name="test",
        build_number=0,
        description="d",
        tag="t",
        authors="a",
        email="a@b.com",
        biotargets=[],
        tags=[],
    )
    data = DataSpec(type="csv", resource="data.csv", input_col="s", target_cols="t")
    proc = ProcedureSpec(
        split=SplitSpec(type="RandomSplitter"),
        feat=FeatureSpec(type="FingerprintFeaturizer"),
        model=ModelSpec(type="LGBMRegressorModel"),
        train=TrainerSpec(type="SKLearnBasicTrainer"),
    )
    report = ReportSpec(eval=[])

    spec = AnvilSpecification(metadata=meta, data=data, procedure=proc, report=report)

    spec.to_multi_yaml(
        metadata_yaml=tmp_path / "meta.yaml",
        procedure_yaml=tmp_path / "proc.yaml",
        data_yaml=tmp_path / "data.yaml",
        report_yaml=tmp_path / "eval.yaml",
    )

    assert (tmp_path / "meta.yaml").exists()
    assert (tmp_path / "proc.yaml").exists()

    reloaded = AnvilSpecification.from_multi_yaml(
        metadata_yaml=tmp_path / "meta.yaml",
        procedure_yaml=tmp_path / "proc.yaml",
        data_yaml=tmp_path / "data.yaml",
        report_yaml=tmp_path / "eval.yaml",
    )

    assert reloaded.metadata.name == spec.metadata.name
    assert reloaded.data.resource == spec.data.resource


def test_anvilspecification_to_workflow_returns_correct_driver_type(mocker):
    """Test that to_workflow returns correct workflow class based on trainer driver."""

    def make_spec(trainer_type, feat_params=None):
        return AnvilSpecification(
            metadata=Metadata(
                version="v1",
                name="t",
                build_number=0,
                description="d",
                tag="t",
                authors="a",
                email="a@b.com",
                biotargets=[],
                tags=[],
            ),
            data=DataSpec(type="csv", resource="d.csv", input_col="s", target_cols="t"),
            procedure=ProcedureSpec(
                split=SplitSpec(type="ShuffleSplitter"),
                feat=FeatureSpec(
                    type="FingerprintFeaturizer",
                    params=feat_params or {"fp_type": "ecfp:4"},
                ),
                model=ModelSpec(type="LGBMRegressorModel"),
                train=TrainerSpec(type=trainer_type),
            ),
            report=ReportSpec(eval=[]),
        )

    # Case 1: SKLEARN driver — use real registered types; no mocking needed
    spec_sklearn = make_spec("SKLearnBasicTrainer")
    workflow_sklearn = spec_sklearn.to_workflow()
    assert isinstance(workflow_sklearn, AnvilWorkflow)

    # Case 2: LIGHTNING driver — mock section.to_class() at class level since no DL model is registered
    from openadmet.models.drivers import DriverType as _DriverType
    from openadmet.models.trainer.lightning import LightningTrainer as _LightningTrainer

    dl_model = mocker.create_autospec(LightningModelBase, instance=True)
    dl_model._n_tasks = 1
    dl_model._driver_type = _DriverType.LIGHTNING
    dl_trainer = mocker.create_autospec(_LightningTrainer, instance=True)
    dl_trainer._driver_type = _DriverType.LIGHTNING

    spec_dl = make_spec("LightningTrainer")

    # Patch only model and trainer to_class; split/feat use real registered types
    mocker.patch.object(ModelSpec, "to_class", autospec=True, return_value=dl_model)
    mocker.patch.object(TrainerSpec, "to_class", autospec=True, return_value=dl_trainer)

    workflow_dl = spec_dl.to_workflow()
    assert isinstance(workflow_dl, AnvilDeepLearningWorkflow)
    assert workflow_dl.model_kwargs == {
        "param_path": None,
        "serial_path": None,
        "freeze_weights": None,
    }


def test_anvilspecification_run_writes_provenance_to_resolved_output_dir(
    tmp_path, mocker
):
    """Test that run() writes the recipe to the output directory."""
    spec = AnvilSpecification(
        metadata=Metadata(
            version="v1",
            name="t",
            build_number=0,
            description="d",
            tag="tag_original",
            authors="a",
            email="a@b.com",
            biotargets=[],
            tags=[],
        ),
        data=DataSpec(type="csv", resource="d.csv", input_col="s", target_cols="t"),
        procedure=ProcedureSpec(
            split=SplitSpec(type="S"),
            feat=FeatureSpec(type="F"),
            model=ModelSpec(type="M"),
            train=TrainerSpec(type="SKLearnBasicTrainer"),
        ),
        report=ReportSpec(eval=[]),
    )

    # Mock workflow run to avoid real execution
    mock_workflow = mocker.Mock()
    mock_workflow.resolved_output_dir = tmp_path / "resolved"
    mock_workflow.run.return_value = None

    mocker.patch.object(
        AnvilSpecification, "to_workflow", autospec=True, return_value=mock_workflow
    )

    spec.run(output_dir=tmp_path / "out")

    # Check that provenance files were written
    assert (tmp_path / "resolved" / "anvil_recipe.yaml").exists()
    assert (tmp_path / "resolved" / "recipe_components" / "metadata.yaml").exists()


def test_anvilspecification_run_tag_override(tmp_path, mocker):
    """Test that providing a tag to run() overrides the metadata tag in provenance."""
    spec = AnvilSpecification(
        metadata=Metadata(
            version="v1",
            name="t",
            build_number=0,
            description="d",
            tag="tag_original",
            authors="a",
            email="a@b.com",
            biotargets=[],
            tags=[],
        ),
        data=DataSpec(type="csv", resource="d.csv", input_col="s", target_cols="t"),
        procedure=ProcedureSpec(
            split=SplitSpec(type="S"),
            feat=FeatureSpec(type="F"),
            model=ModelSpec(type="M"),
            train=TrainerSpec(type="SKLearnBasicTrainer"),
        ),
        report=ReportSpec(eval=[]),
    )

    mock_workflow = mocker.Mock()
    mock_workflow.resolved_output_dir = tmp_path / "resolved"
    mocker.patch.object(
        AnvilSpecification, "to_workflow", autospec=True, return_value=mock_workflow
    )

    spec.run(output_dir=tmp_path / "out", tag="new_tag")

    # Check the saved yaml has the new tag
    saved_yaml = tmp_path / "resolved" / "anvil_recipe.yaml"
    with open(saved_yaml) as f:
        saved_data = yaml.safe_load(f)
    assert saved_data["metadata"]["tag"] == "new_tag"

    # Ensure original object is not mutated
    assert spec.metadata.tag == "tag_original"


# --- DataSpec format/catalog tests (Refinement 5) ---


def test_dataspec_read_single_resource_yaml_raises_without_cat_entry(tmp_path):
    """Test that reading a YAML resource without cat_entry raises ValueError."""
    yaml_path = tmp_path / "catalog.yaml"
    yaml_path.write_text("sources: {}\n")

    spec = DataSpec(
        type="yaml",
        input_col="smiles",
        target_cols="target",
        resource=str(yaml_path),
    )
    with pytest.raises(ValueError, match="cat_entry must be specified"):
        spec.read()


def test_dataspec_read_single_resource_parquet(tmp_path):
    """Test reading a single Parquet resource returns correct data."""
    pq_path = tmp_path / "data.parquet"
    df = pd.DataFrame(
        {
            "smiles": ["CCO", "CC(C)O", "c1ccccc1"],
            "activity": [0.1, 0.5, 0.9],
        }
    )
    df.to_parquet(pq_path, index=False)

    spec = DataSpec(
        type="parquet",
        input_col="smiles",
        target_cols="activity",
        resource=str(pq_path),
    )
    X, y = spec.read()

    assert len(X) == 3
    assert len(y) == 3
    assert list(y.columns) == ["activity"]
    assert X.iloc[0] == "CCO"
    assert y.iloc[0, 0] == pytest.approx(0.1)


def test_dataspec_read_single_resource_unsupported_extension():
    """Test that reading a resource with unsupported extension raises ValueError."""
    spec = DataSpec(
        type="json",
        input_col="smiles",
        target_cols="target",
        resource="/some/file.json",
    )
    with pytest.raises(ValueError, match="Unsupported resource type"):
        spec.read()


def test_dataspec_read_train_test_yaml_raises():
    """Test that YAML resources raise ValueError for train/test split reads."""
    spec = DataSpec(
        type="yaml",
        input_col="smiles",
        target_cols="target",
        train_resource="data.yaml",
        test_resource="data2.yaml",
    )
    with pytest.raises(ValueError, match="YAML catalogs not supported"):
        spec.read()


# --- ModelSpec freeze_weights tests (Refinement 6) ---


def test_modelspec_freeze_weights_succeeds_when_supported(mocker):
    """Test ModelSpec instantiates without error when freeze_weights is supported."""
    mock_model = mocker.create_autospec(LightningModelBase, instance=True)
    mock_model.build.return_value = None
    mock_model.freeze_weights.return_value = None

    mocker.patch.object(ModelSpec, "to_class", autospec=True, return_value=mock_model)

    spec = ModelSpec(type="SomeModel", freeze_weights={"layer": "encoder"})
    assert spec is not None
    mock_model.build.assert_called_once()
    mock_model.freeze_weights.assert_called_once()


def test_modelspec_freeze_weights_raises_when_not_implemented(mocker):
    """Test ModelSpec raises ValueError when freeze_weights is not implemented."""
    mock_model = mocker.create_autospec(LightningModelBase, instance=True)
    mock_model.build.return_value = None
    mock_model.freeze_weights.side_effect = NotImplementedError("not implemented")

    mocker.patch.object(ModelSpec, "to_class", autospec=True, return_value=mock_model)

    with pytest.raises(ValueError, match="Weight freezing not implemented"):
        ModelSpec(type="SomeModel", freeze_weights={"layer": "encoder"})
