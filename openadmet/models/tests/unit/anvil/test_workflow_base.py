import pytest
from pydantic import ConfigDict

from openadmet.models.active_learning.committee import CommitteeRegressor
from openadmet.models.anvil.specification import DataSpec, Metadata
from openadmet.models.anvil.workflow_base import AnvilWorkflowBase
from openadmet.models.architecture.dummy import DummyRegressorModel
from openadmet.models.drivers import DriverType
from openadmet.models.eval.cross_validation import (
    PytorchLightningRepeatedKFoldCrossValidation,
    SKLearnRepeatedKFoldCrossValidation,
)
from openadmet.models.eval.regression import RegressionMetrics
from openadmet.models.features.molfeat_fingerprint import FingerprintFeaturizer
from openadmet.models.split.sklearn import ShuffleSplitter
from openadmet.models.trainer.lightning import LightningTrainer
from openadmet.models.trainer.sklearn import SKlearnBasicTrainer


# Concrete workflow used to test the abstract base validation logic
class ConcreteWorkflow(AnvilWorkflowBase):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    def run(self, output_dir="anvil_training", debug=False):
        return "ran"


# Minimal metadata for testing
def get_minimal_metadata():
    return Metadata(
        version="v1",
        driver="sklearn",
        name="test",
        build_number=0,
        description="desc",
        tag="tag",
        authors="auth",
        email="a@b.com",
        biotargets=[],
        tags=[],
    )


# Helper to build a workflow with real lightweight components as defaults
def build_workflow(
    *,
    model=None,
    trainer=None,
    evals=None,
    ensemble=None,
    target_cols=["target"],
):
    if model is None:
        model = DummyRegressorModel()
    if trainer is None:
        trainer = SKlearnBasicTrainer()
    if evals is None:
        evals = [RegressionMetrics()]
    return ConcreteWorkflow(
        metadata=get_minimal_metadata(),
        data_spec=DataSpec(
            type="csv",
            input_col="smiles",
            target_cols=target_cols,
            resource="data.csv",
        ),
        split=ShuffleSplitter(),
        feat=FingerprintFeaturizer(fp_type="ecfp"),
        model=model,
        trainer=trainer,
        evals=evals,
        ensemble=ensemble,
    )


# --- Tests ---


def test_multitask_check_passes_when_counts_match():
    """Test that validation passes when model n_tasks matches data target_cols."""
    model = DummyRegressorModel()
    model._n_tasks = 2
    workflow = build_workflow(model=model, target_cols=["t1", "t2"])
    assert workflow


def test_multitask_check_raises_when_counts_mismatch():
    """Test that validation raises ValueError when n_tasks does not match target_cols."""
    model = DummyRegressorModel()
    model._n_tasks = 2
    with pytest.raises(ValueError, match="tasks but the data specification has"):
        build_workflow(model=model, target_cols=["t1", "t2", "t3"])


def test_no_ensemble_cross_val_raises_when_both_present():
    """Test that using ensemble with cross-validation raises ValueError."""
    with pytest.raises(
        ValueError, match="Ensemble models cannot be used with cross-validation"
    ):
        build_workflow(
            ensemble=CommitteeRegressor(),
            evals=[SKLearnRepeatedKFoldCrossValidation()],
        )


def test_no_ensemble_cross_val_allows_cv_without_ensemble():
    """Test that cross-validation is allowed if no ensemble is present."""
    workflow = build_workflow(
        evals=[SKLearnRepeatedKFoldCrossValidation()], ensemble=None
    )
    assert workflow


def test_model_trainer_driver_mismatch_raises():
    """Test that mismatched model and trainer drivers raise ValueError."""
    with pytest.raises(ValueError, match="Model driver type .* does not match trainer"):
        build_workflow(trainer=LightningTrainer())


def test_model_trainer_driver_match_succeeds():
    """Test that matching model and trainer drivers succeed."""
    workflow = build_workflow(
        model=DummyRegressorModel(), trainer=SKlearnBasicTrainer()
    )
    assert workflow


def test_cv_trainer_compatibility_raises_on_driver_mismatch():
    """Test that a CV evaluator with a mismatched trainer driver raises ValueError."""
    with pytest.raises(
        ValueError, match="Trainer driver type .* does not match evaluation"
    ):
        build_workflow(
            trainer=SKlearnBasicTrainer(),
            evals=[PytorchLightningRepeatedKFoldCrossValidation()],
        )


def test_cv_trainer_compatibility_ignores_non_cv_evals():
    """Test that non-CV evaluators do not trigger driver mismatch checks."""
    workflow = build_workflow(
        trainer=SKlearnBasicTrainer(),
        evals=[RegressionMetrics()],
    )
    assert workflow
