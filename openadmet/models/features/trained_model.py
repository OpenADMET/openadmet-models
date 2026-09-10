"""Featurizer that emits the predictions of an already-trained Anvil model."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import ClassVar

import numpy as np
import yaml
from pydantic import Field, PrivateAttr, field_validator, model_validator

from openadmet.models.eval.utils import ensure_2d
from openadmet.models.features.feature_base import FeaturizerBase, featurizers
from openadmet.models.transforms.transform_base import transform_features


@featurizers.register("TrainedModelFeaturizer")
class TrainedModelFeaturizer(FeaturizerBase):
    """
    Featurize molecules with the predictions of an already-trained Anvil model.

    The referenced model is frozen: it is loaded from disk, run in inference
    mode, and never refitted. Its predictions become feature columns for a
    downstream model, which is how a model trained on an abundant low-fidelity
    endpoint (e.g. primary-screen log2 fold change) can inform a model trained
    on a scarce high-fidelity one (e.g. dose-response pEC50).

    The pretrained model brings its own featurizer and its own fitted transform,
    so this featurizer takes SMILES rather than features and reproduces the
    pretrained model's inference path in full. Molecules that its featurizer
    drops are reported through the returned index array, the same as any other
    featurizer, so a FeatureConcatenator can intersect them.

    One column per task is emitted, where the tasks are the pretrained model's
    target columns. Setting ``include_std`` doubles that, appending the standard
    deviation block after the prediction block, so two tasks emit
    ``pred_task0, pred_task1, std_task0, std_task1``.

    Attributes
    ----------
    type : ClassVar[str]
        The type of the featurizer.
    model_dir : Path
        Directory of the trained model, in the layout ``anvil`` writes: a
        ``recipe_components`` directory plus the serialized model files.
    include_std : bool
        Whether to append the standard deviation across ensemble members as a
        second block of columns, by default False. Only an ensemble has a
        standard deviation, so this is rejected for a single model.
    accelerator : str
        Accelerator passed to the pretrained model's predict, by default
        'auto', which uses a GPU where one is available and falls back to CPU.

    """

    type: ClassVar[str] = "TrainedModelFeaturizer"

    model_dir: Path = Field(
        ..., description="Directory of the trained model to featurize with"
    )
    include_std: bool = Field(
        default=False,
        description="Whether to append the ensemble standard deviation as extra feature columns",
    )
    accelerator: str = "auto"

    # Cached (model, featurizer, transform) so featurizing several partitions loads once
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
            If the directory, its recipe_components directory, or the
            procedure.yaml inside it is missing.

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

    @model_validator(mode="after")
    def check_std_is_available(self):
        """
        Check the pretrained model can produce a standard deviation when requested.

        Only an ensemble honours ``return_std``. A single model discards it
        through ``**kwargs`` and returns predictions alone, so unpacking the
        result into (prediction, std) either splits that array in two or raises,
        depending on the row count. The recipe names the ensemble, so this is
        answerable from YAML alone.

        Raises
        ------
        ValueError
            If a standard deviation is requested from a model whose recipe has
            no ensemble.

        """
        if not self.include_std:
            return self

        procedure_path = self.model_dir / "recipe_components" / "procedure.yaml"
        with open(procedure_path) as f:
            procedure = yaml.safe_load(f) or {}

        if procedure.get("ensemble") is None:
            raise ValueError(
                f"include_std is set, but the model at {self.model_dir} is not an "
                "ensemble and has no standard deviation to report. Leave include_std "
                "unset, or point at an ensemble model."
            )

        return self

    def _load_pretrained_model(self) -> tuple:
        """
        Load the pretrained model, its featurizer, and its transform, caching the result.

        Returns
        -------
        tuple
            The loaded (model, featurizer, transform), where the transform is
            None when the pretrained recipe has none.

        """
        if self._loaded is None:
            # Defer import
            from openadmet.models.inference.inference import (
                load_anvil_model_and_metadata,
            )

            model, feat, transform, _, _ = load_anvil_model_and_metadata(self.model_dir)
            self._loaded = (model, feat, transform)

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
            (n_featurized, n_tasks), doubled when ``include_std`` is set;
            indices are the positions in the input that the pretrained model's
            featurizer kept.

        """
        model, feat, transform = self._load_pretrained_model()

        # Featurize using pretrained model's featurizer
        feat_data = feat.featurize(smiles)

        # Featurizers return (features, indices) or (dataloader, indices, scaler, dataset)
        X_feat, indices = feat_data[0], feat_data[1]

        # The model was fitted on transformed features, so predict on them too
        if transform is not None:
            # Single-row featurizer output arrives 1D
            X_feat = transform_features(transform, np.atleast_2d(X_feat))
            if X_feat.shape[0] != len(indices):
                raise ValueError(
                    "Transform changed the row count "
                    f"({X_feat.shape[0]} rows vs {len(indices)} indices); "
                    "transforms must preserve rows."
                )

        # Report std if requested
        if self.include_std:
            prediction, std = model.predict(
                X_feat, accelerator=self.accelerator, return_std=True
            )
        else:
            prediction = model.predict(X_feat, accelerator=self.accelerator)

        # Standard deviation columns follow the prediction columns
        blocks = [ensure_2d(np.asarray(prediction))]
        if self.include_std:
            blocks.append(ensure_2d(np.asarray(std)))

        return np.concatenate(blocks, axis=1).astype(np.float64), np.asarray(indices)
