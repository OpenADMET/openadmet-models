"""Base class for Anvil workflows."""

from abc import abstractmethod
from os import PathLike
from typing import Any, Optional

from pydantic import BaseModel

from openadmet.models.active_learning.ensemble_base import (
    EnsembleBase,
)
from openadmet.models.anvil.specification import AnvilSpecification, DataSpec, Metadata
from openadmet.models.architecture.model_base import ModelBase
from openadmet.models.eval.eval_base import EvalBase
from openadmet.models.features.feature_base import FeaturizerBase
from openadmet.models.registries import *  # noqa: F401, F403
from openadmet.models.split.split_base import SplitterBase
from openadmet.models.trainer.trainer_base import TrainerBase
from openadmet.models.transforms.transform_base import (
    TransformBase,
)


class AnvilWorkflowBase(BaseModel):
    """
    Base class for Anvil workflows.

    Attributes
    ----------
    metadata : Metadata
        Metadata for the workflow.
    data_spec : DataSpec
        Data specification for the workflow.
    transform : Optional[TransformBase]
        Optional transform step.
    split : SplitterBase
        Data splitting strategy.
    feat : FeaturizerBase
        Feature extraction method.
    model : ModelBase
        The model to be used.
    ensemble : Optional[EnsembleBase]
        Optional ensemble model.
    trainer : TrainerBase
        The trainer for the model.
    evals : list[EvalBase]
        List of evaluation metrics.
    parent_spec : AnvilSpecification
        The parent specification for the workflow.
    debug : bool
        Whether to run in debug mode.

    """

    metadata: Metadata
    data_spec: DataSpec
    transform: Optional[TransformBase] = None  # Optional transform step
    split: SplitterBase
    feat: FeaturizerBase
    model: ModelBase
    ensemble: EnsembleBase | None = None
    trainer: TrainerBase
    evals: list[EvalBase]
    parent_spec: AnvilSpecification
    debug: bool = False

    @abstractmethod
    def run(self, output_dir: PathLike = "anvil_training", debug: bool = False) -> Any:
        """
        Run the workflow.

        Parameters
        ----------
        output_dir : PathLike, optional
            Directory to save outputs, by default "anvil_training"
        debug : bool, optional
            Whether to run in debug mode, by default False

        Returns
        -------
        Any
            Result of the workflow run

        """
        ...
