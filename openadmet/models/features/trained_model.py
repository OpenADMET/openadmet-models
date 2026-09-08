"""Featurizer that emits the predictions of an already-trained Anvil model."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import ClassVar, Literal

import numpy as np
import yaml
from pydantic import Field, PrivateAttr, field_validator, model_validator

from openadmet.models.eval.utils import ensure_2d
from openadmet.models.features.feature_base import FeaturizerBase, featurizers


@featurizers.register("TrainedModelFeaturizer")
class TrainedModelFeaturizer(FeaturizerBase):
    """
    Featurize molecules with the predictions of an already-trained Anvil model.

    The referenced model is frozen: it is loaded from disk, run in inference
    mode, and never refitted. Its predictions become feature columns for a
    downstream model, which is how a model trained on an abundant low-fidelity
    endpoint (e.g. primary-screen log2 fold change) can inform a model trained
    on a scarce high-fidelity one (e.g. dose-response pEC50).

    The pretrained model brings its own featurizer, so this featurizer takes
    SMILES rather than features. Molecules that its featurizer drops are
    reported through the returned index array, the same as any other
    featurizer, so a FeatureConcatenator can intersect them.

    Output width is ``len(outputs) * n_tasks``, where the tasks are the
    pretrained model's target columns. Columns are laid out output-major and
    task-minor, so ``outputs=[mean, std]`` over two tasks emits
    ``mean_task0, mean_task1, std_task0, std_task1``.

    Attributes
    ----------
    type : ClassVar[str]
        The type of the featurizer.
    model_dir : Path
        Directory of the trained model, in the layout ``anvil`` writes: a
        ``recipe_components`` directory plus the serialized model files.
    outputs : list of str
        Per-task quantities to emit, in column order. 'mean' is the model's
        prediction; 'std' is the ensemble spread and requires the pretrained
        model to be an ensemble. Defaults to ['mean'].
    accelerator : str
        Accelerator passed to the pretrained model's predict, by default
        'cpu', since this runs inside another model's training loop.

    """

    type: ClassVar[str] = "TrainedModelFeaturizer"

    model_dir: Path = Field(
        ..., description="Directory of the trained model to featurize with"
    )
    outputs: list[Literal["mean", "std"]] = Field(
        default=["mean"],
        min_length=1,
        description="Per-task quantities to emit as feature columns, in column order",
    )
    accelerator: str = "cpu"

    # Cached (model, featurizer) so featurizing several partitions loads once
    _loaded: tuple | None = PrivateAttr(default=None)

    @field_validator("model_dir")
    @classmethod
    def validate_model_dir(cls, value: Path) -> Path:
        """
        Check the path looks like a model directory without loading the model.

        Parameters
        ----------
        value : Path
            The configured model directory.

        Returns
        -------
        Path
            The validated directory.

        Raises
        ------
        ValueError
            If the directory or its recipe components are missing.

        """
        value = Path(value)

        # Fail when the recipe is parsed, not part-way through featurization
        if not value.is_dir():
            raise ValueError(f"Model directory {value} does not exist.")

        if not (value / "recipe_components").is_dir():
            raise ValueError(
                f"Model directory {value} has no recipe_components directory, so it "
                "is not a trained Anvil model."
            )

        if not (value / "recipe_components" / "procedure.yaml").is_file():
            raise ValueError(
                f"Model directory {value} has no recipe_components/procedure.yaml, "
                "so it is not a trained Anvil model."
            )

        return value

    @field_validator("outputs")
    @classmethod
    def reject_duplicate_outputs(cls, value: list[str]) -> list[str]:
        """
        Reject a repeated output, which would emit the same columns twice.

        Parameters
        ----------
        value : list of str
            The configured outputs, in column order.

        Returns
        -------
        list of str
            The validated outputs.

        Raises
        ------
        ValueError
            If an output is listed more than once.

        """
        duplicates = sorted({name for name in value if value.count(name) > 1})
        if duplicates:
            raise ValueError(f"Duplicate outputs: {duplicates}.")

        return value

    @model_validator(mode="after")
    def check_std_is_available(self):
        """
        Check the pretrained model can produce a spread when 'std' is requested.

        Only an ensemble has a spread; a single model reports NaN, which would
        silently fill a feature column with missing values. The recipe names
        the ensemble, so this is answerable from YAML alone.

        Raises
        ------
        ValueError
            If 'std' is requested from a model whose recipe has no ensemble.

        """
        if "std" not in self.outputs:
            return self

        procedure_path = self.model_dir / "recipe_components" / "procedure.yaml"
        with open(procedure_path) as f:
            procedure = yaml.safe_load(f) or {}

        if procedure.get("ensemble") is None:
            raise ValueError(
                f"outputs includes 'std', but the model at {self.model_dir} is not an "
                "ensemble and has no spread to report. Use outputs: [mean], or point "
                "at an ensemble model."
            )

        return self

    def _load_pretrained_model(self) -> tuple:
        """
        Load the pretrained model and its featurizer, caching the result.

        Returns
        -------
        tuple
            The loaded (model, featurizer).

        """
        if self._loaded is None:
            # Defer import
            from openadmet.models.inference.inference import (
                load_anvil_model_and_metadata,
            )

            model, feat, _, _ = load_anvil_model_and_metadata(self.model_dir)
            self._loaded = (model, feat)

        return self._loaded

    def featurize(self, smiles: Iterable[str]) -> tuple[np.ndarray, np.ndarray]:
        """
        Featurize SMILES with the trained model's predictions.

        Parameters
        ----------
        smiles : Iterable[str]
            List or iterable of SMILES strings to featurize.

        Returns
        -------
        tuple
            Tuple of (features, indices). Features has shape
            (n_featurized, len(outputs) * n_tasks); indices are the positions
            in the input that the pretrained model's featurizer kept.

        """
        model, feat = self._load_pretrained_model()

        # Featurize using pretrained model's featurizer
        X_feat, indices = feat.featurize(smiles)[:2]

        # Report std if requested
        if "std" in self.outputs:
            mean, std = model.predict(
                X_feat, accelerator=self.accelerator, return_std=True
            )
        else:
            mean = model.predict(X_feat, accelerator=self.accelerator)
            std = None

        # One block of columns per requested output, in the order listed
        blocks = []
        for name in self.outputs:
            values = mean if name == "mean" else std
            blocks.append(ensure_2d(np.asarray(values)))

        return np.concatenate(blocks, axis=1).astype(np.float64), np.asarray(indices)
