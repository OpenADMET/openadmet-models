"""TabICL model implementations."""

from typing import ClassVar, Literal

import numpy as np
from loguru import logger
from pydantic import ConfigDict, Field, field_validator

from openadmet.models.architecture.model_base import PickleableModelBase, models


class TabICLModelBase(PickleableModelBase):
    """
    Base class for TabICL models.

    Attributes
    ----------
    type : ClassVar[str]
        Model type identifier.
    accelerator : Literal["cpu", "gpu", "auto"]
        Device to use for training and prediction. Mapped to ``device`` for TabICL.
    random_seed : int
        Random seed for reproducibility. Mapped to ``random_state`` for TabICL.
    n_estimators : int
        Number of ensemble members.
    batch_size : int
        Ensemble members processed together.
    use_amp : str
        Automatic mixed precision mode.
    use_fa3 : str
        Flash Attention 3 usage mode.
    offload_mode : str
        Offload mode for memory management.

    """

    model_config = ConfigDict(extra="allow")
    type: ClassVar[str]

    accelerator: Literal["cpu", "gpu", "auto"] = Field(
        default="auto", description="The device to use for training and prediction."
    )
    random_seed: int = Field(default=42, description="Random seed for reproducibility.")
    n_estimators: int = Field(default=8, description="Number of ensemble members.")
    batch_size: int = Field(
        default=1, description="Ensemble batch size for memory control."
    )
    use_amp: str = Field(
        default="auto", description="Automatic mixed precision setting."
    )
    use_fa3: str = Field(default="auto", description="Flash Attention 3 setting.")
    offload_mode: str = Field(
        default="auto", description="Offload mode for large data."
    )

    @field_validator("accelerator")
    @classmethod
    def validate_accelerator(cls, value: str) -> str:
        """
        Validate the accelerator parameter.

        Parameters
        ----------
        value : str
            Accelerator value to validate.

        Returns
        -------
        str
            Validated accelerator value.

        """
        if value not in ["cpu", "gpu", "auto"]:
            raise ValueError("Accelerator must be 'cpu', 'gpu' or 'auto'")
        return value

    @classmethod
    def _get_estimator_class(cls) -> type:
        """Return the TabICL estimator class."""
        raise NotImplementedError

    def _build_kwargs(self) -> dict:
        """Collect kwargs for the underlying estimator."""
        accelerator = self.accelerator if self.accelerator != "gpu" else "cuda"
        data = self.model_dump()
        # Map public names to TabICL names
        kwargs = {
            "n_estimators": data.get("n_estimators", 8),
            "batch_size": data.get("batch_size", 1),
            "device": accelerator,
            "random_state": data.get("random_seed", 42),
            "use_amp": data.get("use_amp", "auto"),
            "use_fa3": data.get("use_fa3", "auto"),
            "offload_mode": data.get("offload_mode", "auto"),
        }
        # Allow extra fields but whitelist known TabICL params
        allowed_extra = {
            "norm_methods",
            "feat_shuffle_method",
            "class_shuffle_method",
            "outlier_threshold",
            "softmax_temperature",
            "average_logits",
            "support_many_classes",
            "kv_cache",
            "model_path",
            "allow_auto_download",
            "checkpoint_version",
            "disk_offload_dir",
            "n_jobs",
            "verbose",
            "inference_config",
        }
        for k, v in data.items():
            if k in {
                "accelerator",
                "random_seed",
                "n_estimators",
                "batch_size",
                "use_amp",
                "use_fa3",
                "offload_mode",
            }:
                continue
            if k in allowed_extra:
                kwargs[k] = v
            else:
                # Unknown extra field – ignore to avoid passing invalid args to TabICL
                pass
        return kwargs

    def build(self) -> None:
        """Prepare and build the model instance."""
        if not self.estimator:
            estimator_cls = self._get_estimator_class()
            kwargs = self._build_kwargs()
            self.estimator = estimator_cls(**kwargs)
        else:
            logger.warning("Model already exists, skipping build")

    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Train the model.

        Parameters
        ----------
        X : np.ndarray
            Training features.
        y : np.ndarray
            Training targets.

        """
        self.build()
        self.estimator = self.estimator.fit(X, y)

    def predict(self, X: np.ndarray, **kwargs) -> np.ndarray:
        """
        Predict using the model.

        Parameters
        ----------
        X : np.ndarray
            Input features.
        kwargs : dict
            Additional arguments for prediction. Must be empty; unknown kwargs raise TypeError.

        Returns
        -------
        np.ndarray
            Model predictions with shape (n_samples, 1).

        Raises
        ------
        ValueError
            If the model is not trained.
        TypeError
            If unknown prediction kwargs are supplied.

        """
        if not self.estimator:
            raise ValueError("Model not trained")
        if kwargs:
            raise TypeError("TabICL predict does not accept extra kwargs")
        return np.expand_dims(self.estimator.predict(X), axis=1)


@models.register("TabICLRegressorModel")
class TabICLRegressorModel(TabICLModelBase):
    """TabICL regression model."""

    type: ClassVar[str] = "TabICLRegressorModel"

    @classmethod
    def _get_estimator_class(cls) -> type:
        from tabicl import TabICLRegressor

        return TabICLRegressor


@models.register("TabICLClassifierModel")
class TabICLClassifierModel(TabICLModelBase):
    """TabICL classification model."""

    type: ClassVar[str] = "TabICLClassifierModel"

    @classmethod
    def _get_estimator_class(cls) -> type:
        from tabicl import TabICLClassifier

        return TabICLClassifier

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities.

        Parameters
        ----------
        X : np.ndarray
            Input features.

        Returns
        -------
        np.ndarray
            Predicted class probabilities.

        """
        if not self.estimator:
            raise ValueError("Model not trained")
        return self.estimator.predict_proba(X)
