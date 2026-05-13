"""Unit tests for anvil/workflow.py — utility functions, class instantiation, Pydantic validators, and driver routing.

Scope is intentionally limited to construction-time behavior. No `.run()`, `_train()`, or execution
paths are exercised here; those belong in integration tests.
"""

import numpy as np
import pandas as pd
import pytest

from openadmet.models.active_learning.committee import CommitteeRegressor
from openadmet.models.anvil.specification import DataSpec, Metadata
from openadmet.models.anvil.workflow import (
    _DRIVER_TO_CLASS,
    AnvilDeepLearningWorkflow,
    AnvilWorkflow,
    _safe_to_numpy,
)
from openadmet.models.architecture.chemprop import ChemPropModel
from openadmet.models.architecture.dummy import DummyRegressorModel
from openadmet.models.drivers import DriverType
from openadmet.models.features.molfeat_fingerprint import FingerprintFeaturizer
from openadmet.models.split.sklearn import ShuffleSplitter
from openadmet.models.trainer.lightning import LightningTrainer
from openadmet.models.trainer.sklearn import SKlearnBasicTrainer
from openadmet.models.transforms.impute import ImputeTransform

# ---------------------------------------------------------------------------
# Module-scoped fixtures — constructed once per test session for performance
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def metadata():
    """Return a minimal real Metadata instance."""
    return Metadata(
        version="v1",
        driver="sklearn",
        name="test-workflow",
        build_number=0,
        description="Unit test workflow",
        tag="test-tag",
        authors="Test Author",
        email="test@example.com",
        biotargets=["target1"],
        tags=["unit-test"],
    )


@pytest.fixture(scope="module")
def data_spec():
    """Return a minimal real DataSpec instance with one target column."""
    return DataSpec(
        type="csv",
        input_col="smiles",
        target_cols=["target"],
        resource="data.csv",
    )


@pytest.fixture(scope="module")
def sklearn_feat():
    """Return a real FingerprintFeaturizer using ECFP4 (RDKit-only, no downloads)."""
    return FingerprintFeaturizer(fp_type="ecfp:4")


# ---------------------------------------------------------------------------
# Factory helpers — build workflows from fully real Pydantic components
# ---------------------------------------------------------------------------


def _make_anvil_workflow(
    metadata,
    data_spec,
    feat,
    *,
    split=None,
    ensemble=None,
    model_kwargs=None,
    ensemble_kwargs=None,
    transform=None,
):
    """Construct an AnvilWorkflow from real lightweight production components."""
    if split is None:
        split = ShuffleSplitter(train_size=0.8, val_size=0.0, test_size=0.2)
    return AnvilWorkflow(
        metadata=metadata,
        data_spec=data_spec,
        split=split,
        feat=feat,
        model=DummyRegressorModel(),
        trainer=SKlearnBasicTrainer(),
        evals=[],
        ensemble=ensemble,
        transform=transform,
        model_kwargs=model_kwargs or {},
        ensemble_kwargs=ensemble_kwargs or {},
    )


def _make_dl_workflow(
    metadata,
    data_spec,
    feat,
    *,
    split=None,
    ensemble=None,
    transform=None,
    model_kwargs=None,
    ensemble_kwargs=None,
):
    """Construct an AnvilDeepLearningWorkflow from real lightweight production components."""
    if split is None:
        split = ShuffleSplitter(train_size=0.8, val_size=0.0, test_size=0.2)
    return AnvilDeepLearningWorkflow(
        metadata=metadata,
        data_spec=data_spec,
        split=split,
        feat=feat,
        model=ChemPropModel(),
        trainer=LightningTrainer(),
        evals=[],
        ensemble=ensemble,
        transform=transform,
        model_kwargs=model_kwargs or {},
        ensemble_kwargs=ensemble_kwargs or {},
    )


# ---------------------------------------------------------------------------
# Section 1: _safe_to_numpy utility
# ---------------------------------------------------------------------------


def test_safe_to_numpy_series():
    """Test that _safe_to_numpy converts a pd.Series to a np.ndarray with correct values."""
    s = pd.Series([1.0, 2.0, 3.0])
    result = _safe_to_numpy(s)
    assert isinstance(result, np.ndarray)
    assert result.shape == (3,)
    np.testing.assert_array_equal(result, np.array([1.0, 2.0, 3.0]))


def test_safe_to_numpy_dataframe():
    """Test that _safe_to_numpy converts a pd.DataFrame to a np.ndarray with correct shape and values."""
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    result = _safe_to_numpy(df)
    assert isinstance(result, np.ndarray)
    assert result.shape == (2, 2)
    np.testing.assert_array_equal(result, df.to_numpy())


def test_safe_to_numpy_ndarray_passthrough():
    """Test that _safe_to_numpy returns a np.ndarray unchanged via identity check."""
    arr = np.array([1.0, 2.0, 3.0])
    result = _safe_to_numpy(arr)
    assert result is arr


# ---------------------------------------------------------------------------
# Section 2: _DRIVER_TO_CLASS routing dictionary
# ---------------------------------------------------------------------------


def test_driver_to_class_sklearn_routes_to_anvil_workflow():
    """Test that DriverType.SKLEARN maps to AnvilWorkflow."""
    assert _DRIVER_TO_CLASS[DriverType.SKLEARN] is AnvilWorkflow


def test_driver_to_class_lightning_routes_to_dl_workflow():
    """Test that DriverType.LIGHTNING maps to AnvilDeepLearningWorkflow."""
    assert _DRIVER_TO_CLASS[DriverType.LIGHTNING] is AnvilDeepLearningWorkflow


def test_driver_to_class_has_exactly_two_entries():
    """Test that _DRIVER_TO_CLASS contains exactly the two expected driver keys."""
    assert set(_DRIVER_TO_CLASS.keys()) == {DriverType.SKLEARN, DriverType.LIGHTNING}


# ---------------------------------------------------------------------------
# Section 3: AnvilWorkflow happy-path construction
# ---------------------------------------------------------------------------


def test_anvil_workflow_constructs_with_real_components(
    metadata, data_spec, sklearn_feat
):
    """Test that AnvilWorkflow can be constructed from real lightweight registered components."""
    wf = _make_anvil_workflow(metadata, data_spec, sklearn_feat)
    assert isinstance(wf, AnvilWorkflow)


def test_anvil_workflow_driver_type_is_sklearn(metadata, data_spec, sklearn_feat):
    """Test that AnvilWorkflow correctly exposes the SKLEARN driver type."""
    wf = _make_anvil_workflow(metadata, data_spec, sklearn_feat)
    assert wf._driver_type == DriverType.SKLEARN


# ---------------------------------------------------------------------------
# Section 4: AnvilWorkflow.check_if_val_needed validator
# ---------------------------------------------------------------------------


def test_anvil_workflow_ensemble_without_val_raises(metadata, data_spec, sklearn_feat):
    """Test that constructing an ensemble AnvilWorkflow without a validation split raises ValueError."""
    split = ShuffleSplitter(train_size=0.8, val_size=0.0, test_size=0.2)
    with pytest.raises(ValueError, match="Ensemble models require a validation set"):
        _make_anvil_workflow(
            metadata,
            data_spec,
            sklearn_feat,
            split=split,
            ensemble=CommitteeRegressor(),
            ensemble_kwargs={"n_models": 2},
        )


def test_anvil_workflow_val_without_ensemble_raises(metadata, data_spec, sklearn_feat):
    """Test that requesting a validation split without an ensemble raises ValueError."""
    split = ShuffleSplitter(train_size=0.7, val_size=0.1, test_size=0.2)
    with pytest.raises(ValueError, match="Validation set requested, but not used"):
        _make_anvil_workflow(
            metadata, data_spec, sklearn_feat, split=split, ensemble=None
        )


def test_anvil_workflow_ensemble_with_val_succeeds(metadata, data_spec, sklearn_feat):
    """Test that an ensemble AnvilWorkflow with a validation split constructs without error."""
    split = ShuffleSplitter(train_size=0.7, val_size=0.1, test_size=0.2)
    wf = _make_anvil_workflow(
        metadata,
        data_spec,
        sklearn_feat,
        split=split,
        ensemble=CommitteeRegressor(),
        ensemble_kwargs={"n_models": 2},
    )
    assert isinstance(wf, AnvilWorkflow)
    assert wf.ensemble is not None


# ---------------------------------------------------------------------------
# Section 5: AnvilWorkflow.check_no_finetuning validator
# ---------------------------------------------------------------------------


# Single-model branch: all triggering combinations of path kwargs
@pytest.mark.parametrize(
    "model_kwargs",
    [
        {"param_path": "model.json"},
        {"serial_path": "model.pkl"},
        {"param_path": "model.json", "serial_path": "model.pkl"},
    ],
    ids=["param-path-only", "serial-path-only", "both-paths"],
)
def test_anvil_workflow_single_model_finetuning_raises(
    metadata, data_spec, sklearn_feat, model_kwargs
):
    """Test that any finetuning path kwarg for a single model raises ValueError."""
    with pytest.raises(
        ValueError, match="Finetuning from serialized model is not supported"
    ):
        _make_anvil_workflow(
            metadata, data_spec, sklearn_feat, model_kwargs=model_kwargs
        )


# Single-model branch: safe kwargs that must never trigger the validator
@pytest.mark.parametrize(
    "model_kwargs",
    [
        {},
        {"n_estimators": 100},
    ],
    ids=["empty-kwargs", "unrelated-key"],
)
def test_anvil_workflow_single_model_no_finetuning_succeeds(
    metadata, data_spec, sklearn_feat, model_kwargs
):
    """Test that empty or unrelated model_kwargs do not trigger the finetuning validator."""
    wf = _make_anvil_workflow(
        metadata, data_spec, sklearn_feat, model_kwargs=model_kwargs
    )
    assert isinstance(wf, AnvilWorkflow)
    assert wf.model_kwargs == model_kwargs


# Ensemble branch: all triggering combinations of path kwargs
@pytest.mark.parametrize(
    "path_kwargs",
    [
        {"param_paths": ["p1.json", "p2.json"]},
        {"serial_paths": ["s1.pkl", "s2.pkl"]},
        {"param_paths": ["p1.json", "p2.json"], "serial_paths": ["s1.pkl", "s2.pkl"]},
    ],
    ids=["param-paths-only", "serial-paths-only", "both-path-types"],
)
def test_anvil_workflow_ensemble_finetuning_raises(
    metadata, data_spec, sklearn_feat, path_kwargs
):
    """Test that any finetuning path kwarg for an ensemble raises ValueError."""
    split = ShuffleSplitter(train_size=0.7, val_size=0.1, test_size=0.2)
    ensemble_kwargs = {"n_models": 2, **path_kwargs}
    with pytest.raises(
        ValueError, match="Finetuning from serialized ensemble models is not supported"
    ):
        _make_anvil_workflow(
            metadata,
            data_spec,
            sklearn_feat,
            split=split,
            ensemble=CommitteeRegressor(),
            ensemble_kwargs=ensemble_kwargs,
        )


# Ensemble branch: non-path ensemble_kwargs that must never trigger the validator
@pytest.mark.parametrize(
    "ensemble_kwargs",
    [
        {"n_models": 2},
        {"n_models": 2, "calibration_method": "isotonic-regression"},
    ],
    ids=["n-models-only", "with-calibration-method"],
)
def test_anvil_workflow_ensemble_no_finetuning_succeeds(
    metadata, data_spec, sklearn_feat, ensemble_kwargs
):
    """Test that ensemble_kwargs containing only non-path keys do not trigger the finetuning validator."""
    split = ShuffleSplitter(train_size=0.7, val_size=0.1, test_size=0.2)
    wf = _make_anvil_workflow(
        metadata,
        data_spec,
        sklearn_feat,
        split=split,
        ensemble=CommitteeRegressor(),
        ensemble_kwargs=ensemble_kwargs,
    )
    assert isinstance(wf, AnvilWorkflow)
    assert wf.ensemble_kwargs == ensemble_kwargs


# ---------------------------------------------------------------------------
# Section 6: AnvilDeepLearningWorkflow happy-path construction
# ---------------------------------------------------------------------------


def test_dl_workflow_constructs_with_real_components(metadata, data_spec, sklearn_feat):
    """Test that AnvilDeepLearningWorkflow can be constructed from real lightweight registered components."""
    wf = _make_dl_workflow(metadata, data_spec, sklearn_feat)
    assert isinstance(wf, AnvilDeepLearningWorkflow)


def test_dl_workflow_driver_type_is_lightning(metadata, data_spec, sklearn_feat):
    """Test that AnvilDeepLearningWorkflow correctly exposes the LIGHTNING driver type."""
    wf = _make_dl_workflow(metadata, data_spec, sklearn_feat)
    assert wf._driver_type == DriverType.LIGHTNING


# ---------------------------------------------------------------------------
# Section 7: AnvilDeepLearningWorkflow.check_no_transform validator
# ---------------------------------------------------------------------------


def test_dl_workflow_rejects_transform(metadata, data_spec, sklearn_feat):
    """Test that specifying a transform step in a DL workflow raises ValueError."""
    with pytest.raises(ValueError, match="Transform step is not supported"):
        _make_dl_workflow(
            metadata, data_spec, sklearn_feat, transform=ImputeTransform()
        )


def test_dl_workflow_accepts_no_transform(metadata, data_spec, sklearn_feat):
    """Test that a DL workflow without a transform step constructs successfully."""
    wf = _make_dl_workflow(metadata, data_spec, sklearn_feat, transform=None)
    assert wf.transform is None


# ---------------------------------------------------------------------------
# Section 8: AnvilDeepLearningWorkflow.check_if_val_needed validator
# ---------------------------------------------------------------------------


def test_dl_workflow_ensemble_requires_val_raises(metadata, data_spec, sklearn_feat):
    """Test that a DL ensemble workflow without a validation split raises ValueError."""
    split = ShuffleSplitter(train_size=0.8, val_size=0.0, test_size=0.2)
    with pytest.raises(ValueError, match="Ensemble models require a validation set"):
        _make_dl_workflow(
            metadata,
            data_spec,
            sklearn_feat,
            split=split,
            ensemble=CommitteeRegressor(),
        )


def test_dl_workflow_ensemble_with_val_succeeds(metadata, data_spec, sklearn_feat):
    """Test that a DL ensemble workflow with a validation split constructs successfully."""
    split = ShuffleSplitter(train_size=0.7, val_size=0.1, test_size=0.2)
    wf = _make_dl_workflow(
        metadata,
        data_spec,
        sklearn_feat,
        split=split,
        ensemble=CommitteeRegressor(),
    )
    assert isinstance(wf, AnvilDeepLearningWorkflow)
    assert wf.ensemble is not None


# ---------------------------------------------------------------------------
# Section 9: AnvilDeepLearningWorkflow.check_finetuning_paths validator
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "model_kwargs,match",
    [
        ({"param_path": "/nonexistent/model.json"}, "must be provided together"),
        ({"serial_path": "/nonexistent/model.pth"}, "must be provided together"),
        (
            {
                "param_path": "/nonexistent/model.json",
                "serial_path": "/nonexistent/model.pth",
            },
            "does not exist",
        ),
    ],
    ids=["param-path-only", "serial-path-only", "both-nonexistent"],
)
def test_dl_workflow_single_model_finetuning_path_raises(
    metadata, data_spec, sklearn_feat, model_kwargs, match
):
    """Test that mismatched or nonexistent single-model finetuning paths raise ValueError."""
    with pytest.raises(ValueError, match=match):
        _make_dl_workflow(metadata, data_spec, sklearn_feat, model_kwargs=model_kwargs)


def test_dl_workflow_single_model_finetuning_path_succeeds_no_paths(
    metadata, data_spec, sklearn_feat
):
    """Test that empty model_kwargs passes finetuning path validation."""
    wf = _make_dl_workflow(metadata, data_spec, sklearn_feat, model_kwargs={})
    assert isinstance(wf, AnvilDeepLearningWorkflow)
    assert wf.model_kwargs == {}


def test_dl_workflow_single_model_finetuning_path_succeeds_both_exist(
    metadata, data_spec, sklearn_feat, tmp_path
):
    """Test that both finetuning paths pointing to real files passes validation."""
    param_file = tmp_path / "model.json"
    serial_file = tmp_path / "model.pth"
    param_file.touch()
    serial_file.touch()

    model_kwargs = {"param_path": str(param_file), "serial_path": str(serial_file)}
    wf = _make_dl_workflow(metadata, data_spec, sklearn_feat, model_kwargs=model_kwargs)
    assert isinstance(wf, AnvilDeepLearningWorkflow)
    assert wf.model_kwargs == model_kwargs


@pytest.mark.parametrize(
    "path_kwargs,match",
    [
        (
            {"param_paths": ["/nonexistent/p1.json", "/nonexistent/p2.json"]},
            "must be provided together",
        ),
        (
            {"serial_paths": ["/nonexistent/s1.pth", "/nonexistent/s2.pth"]},
            "must be provided together",
        ),
        (
            {
                "param_paths": ["/nonexistent/p1.json", "/nonexistent/p2.json"],
                "serial_paths": ["/nonexistent/s1.pth"],
            },
            "equal length",
        ),
        (
            {
                "param_paths": ["/nonexistent/p1.json", "/nonexistent/p2.json"],
                "serial_paths": ["/nonexistent/s1.pth", "/nonexistent/s2.pth"],
            },
            "does not exist",
        ),
    ],
    ids=[
        "param-paths-only",
        "serial-paths-only",
        "unequal-lengths",
        "both-nonexistent",
    ],
)
def test_dl_workflow_ensemble_finetuning_path_raises(
    metadata, data_spec, sklearn_feat, path_kwargs, match
):
    """Test that mismatched, unequal-length, or nonexistent ensemble finetuning paths raise ValueError."""
    split = ShuffleSplitter(train_size=0.7, val_size=0.1, test_size=0.2)
    ensemble_kwargs = {"n_models": 2, **path_kwargs}
    with pytest.raises(ValueError, match=match):
        _make_dl_workflow(
            metadata,
            data_spec,
            sklearn_feat,
            split=split,
            ensemble=CommitteeRegressor(),
            ensemble_kwargs=ensemble_kwargs,
        )


def test_dl_workflow_ensemble_finetuning_path_succeeds_no_paths(
    metadata, data_spec, sklearn_feat
):
    """Test that ensemble_kwargs with no path keys passes finetuning path validation."""
    split = ShuffleSplitter(train_size=0.7, val_size=0.1, test_size=0.2)
    wf = _make_dl_workflow(
        metadata,
        data_spec,
        sklearn_feat,
        split=split,
        ensemble=CommitteeRegressor(),
        ensemble_kwargs={"n_models": 2},
    )
    assert isinstance(wf, AnvilDeepLearningWorkflow)
    assert wf.ensemble is not None


def test_dl_workflow_ensemble_finetuning_path_succeeds_both_exist(
    metadata, data_spec, sklearn_feat, tmp_path
):
    """Test that ensemble finetuning paths pointing to real files passes validation."""
    p1, p2 = tmp_path / "m0.json", tmp_path / "m1.json"
    s1, s2 = tmp_path / "m0.pth", tmp_path / "m1.pth"
    for f in [p1, p2, s1, s2]:
        f.touch()

    split = ShuffleSplitter(train_size=0.7, val_size=0.1, test_size=0.2)
    ensemble_kwargs = {
        "n_models": 2,
        "param_paths": [str(p1), str(p2)],
        "serial_paths": [str(s1), str(s2)],
    }
    wf = _make_dl_workflow(
        metadata,
        data_spec,
        sklearn_feat,
        split=split,
        ensemble=CommitteeRegressor(),
        ensemble_kwargs=ensemble_kwargs,
    )
    assert isinstance(wf, AnvilDeepLearningWorkflow)
    assert wf.ensemble_kwargs == ensemble_kwargs
