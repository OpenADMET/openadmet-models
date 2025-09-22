"""ChemProp and Chemeleon model implementations."""

from pathlib import Path
from typing import ClassVar
from urllib.request import urlretrieve

import numpy as np
import torch
from chemprop import models, nn
from lightning import pytorch as pl
from loguru import logger
from pydantic import field_validator

from openadmet.models.architecture.model_base import LightningModelBase
from openadmet.models.architecture.model_base import models as model_registry

_METRIC_TO_LOSS = {
    "mse": nn.metrics.MSE(),
    "mae": nn.metrics.MAE(),
    "rmse": nn.metrics.RMSE(),
}


@model_registry.register("ChemPropModel")
class ChemPropModel(LightningModelBase):
    """
    ChemProp regression model.

    Attributes
    ----------
    type : ClassVar[str]
        The type of the model.
    batch_norm : bool
        Whether to use batch normalization.
    monitor_metric : str
        The metric to monitor during training.
    metric_list : list
        List of metrics to use for evaluation.
    mod_params : dict
        Parameters for the ChemProp model class, such as depth, message_hidden_dim,
        ffn_hidden_dim, ffn_num_layers, messages, aggregation, etc.
    from_chemeleon : bool
        Whether to initialize the model from the CheMeleon foundation model.
    depth : int
        The depth of the message passing network.
    message_hidden_dim : int
        The hidden dimension of the message passing network.
    ffn_hidden_dim : int
        The hidden dimension of the feed-forward network.
    ffn_num_layers : int
        The number of layers in the feed-forward network.
    messages : str
        The type of messages to use, either 'bond' or 'atom'.
    aggregation : str
        The type of aggregation to use, either 'mean' or 'norm'.
    normalized_targets : bool
        Whether the targets are normalized.
    dropout : float
        The dropout rate.
    n_tasks : int
        The number of tasks (outputs).

    """

    type: ClassVar[str] = "ChemPropModel"
    batch_norm: bool = False
    monitor_metric: str = "val_loss"
    metric_list: list = ["mse", "mae", "rmse"]
    mod_params: dict = {}
    from_chemeleon: bool = False
    depth: int = 3
    message_hidden_dim: int = 300
    ffn_hidden_dim: int = 300
    ffn_num_layers: int = 1
    messages: str = "bond"
    aggregation: str = "norm"
    normalized_targets: bool = True
    dropout: float = 0.0
    n_tasks: int = 1

    @field_validator("messages")
    @classmethod
    def validate_messages(cls, value):
        """Validate the messages parameter."""
        if value not in ["bond", "atom"]:
            raise ValueError("Messages must be either 'bond' or 'atom'")
        return value

    @field_validator("aggregation")
    @classmethod
    def validate_aggregation(cls, value):
        """Validate the aggregation parameter."""
        if value not in ["mean", "norm"]:
            raise ValueError("Aggregation must be either 'mean' or 'norm'")
        return value

    def _get_output_transform(self, scaler):
        """Set the output transform."""
        if scaler is not None:
            output_transform = nn.UnscaleTransform.from_standard_scaler(scaler)
        elif self.normalized_targets:
            # Expects the targets to be normalized, likely to be loaded from state dict
            output_transform = nn.UnscaleTransform(
                [1] * self.n_tasks, [0] * self.n_tasks
            )
        else:
            output_transform = None
        return output_transform

    @classmethod
    def from_params(cls, class_params: dict = {}, mod_params: dict = {}):
        """
        Create a model from parameters.

        Parameters
        ----------
        class_params: dict
            Parameters for the model class, such as type, mod_class, etc.
        mod_params: dict
            Parameters for the ChemProp model class, such as depth, message_hidden_dim,
            ffn_hidden_dim, ffn_num_layers, messages, aggregation, etc.

        Returns
        -------
        ChemPropModel
            An instance of the ChemPropModel class

        """
        instance = cls(**class_params, mod_params=mod_params)
        instance.build()
        return instance

    def make_new(self) -> "ChemPropModel":
        """Copy parameters to a new model instance without copying the estimator."""
        return self.__class__(**self.mod_params, **self.dict(exclude={"estimator"}))

    def train(self, dataloader, scaler=None):
        """
        Train the model.

        Parameters
        ----------
        dataloader: torch.utils.data.DataLoader
            DataLoader for training data
        scaler: sklearn.preprocessing.StandardScaler, optional
            Scaler for target normalization, if applicable

        Returns
        -------
        None

        """
        raise NotImplementedError(
            "Training not implemented in model class, use a trainer"
        )

    def build(self, scaler=None):
        """
        Prepare the model.

        Parameters
        ----------
        scaler: sklearn.preprocessing.StandardScaler, optional
            Scaler for target normalization, if applicable

        Returns
        -------
        ChemPropModel
            The built ChemPropModel instance

        """
        if not self.estimator:
            metric_list = [_METRIC_TO_LOSS[metric] for metric in self.metric_list]

            if self.from_chemeleon:
                logger.info(
                    "Please cite DOI: 10.48550/arXiv.2506.15792 when using CheMeleon in published work"
                )
                ckpt_dir = Path().home() / ".chemprop"
                ckpt_dir.mkdir(exist_ok=True)
                model_path = ckpt_dir / "chemeleon_mp.pt"
                if not model_path.exists():
                    logger.info(
                        f"Downloading CheMeleon Foundation model from Zenodo (https://zenodo.org/records/15460715) to {model_path}"
                    )
                    urlretrieve(
                        r"https://zenodo.org/records/15460715/files/chemeleon_mp.pt",
                        model_path,
                    )
                else:
                    logger.info(f"Loading cached CheMeleon from {model_path}")
                aggr = nn.MeanAggregation()
                chemeleon_mp = torch.load(model_path, weights_only=True)
                mp = nn.BondMessagePassing(**chemeleon_mp["hyper_parameters"])
                mp.load_state_dict(chemeleon_mp["state_dict"])
                self.message_hidden_dim = mp.output_dim
                logger.warning(
                    "Using CheMeleon overrides settings for depth, message_hidden_dim, messages, and aggregation"
                )
            else:
                aggregation_cls = (
                    nn.MeanAggregation
                    if self.aggregation == "mean"
                    else nn.NormAggregation
                )
                message_cls = (
                    nn.BondMessagePassing
                    if self.messages == "bond"
                    else nn.AtomMessagePassing
                )

                # Create the model
                mp = message_cls(
                    d_h=self.message_hidden_dim, depth=self.depth, dropout=self.dropout
                )
                aggr = aggregation_cls()

            ffn = nn.RegressionFFN(
                n_tasks=self.n_tasks,
                input_dim=self.message_hidden_dim,
                hidden_dim=self.ffn_hidden_dim,
                n_layers=self.ffn_num_layers,
                output_transform=self._get_output_transform(scaler),
                dropout=self.dropout,
            )

            # Create the MPNN model
            mpnn = models.MPNN(mp, aggr, ffn, self.batch_norm, metric_list)

            # Pass monitor metric from "model" to "module"
            mpnn.monitor_metric = self.monitor_metric
            self.estimator = mpnn

        else:
            logger.warning("Model already exists, skipping build")

        return self

    def predict(
        self, X: np.ndarray, accelerator="gpu", devices=1, **kwargs
    ) -> np.ndarray:
        """
        Predict using the model.

        Parameters
        ----------
        X: np.ndarray
            Featurized data to predict on
        accelerator: str
            Accelerator to use for prediction, options are: 'cpu', 'gpu'
        devices: int
            Number of devices to use for prediction
        kwargs: dict
            Additional keyword arguments for prediction

        Returns
        -------
        np.ndarray
            Predictions as a numpy array

        """
        if not self.estimator:
            raise AttributeError("Model not trained")

        self.estimator.eval()

        with torch.inference_mode():
            trainer = pl.Trainer(
                logger=None,
                enable_progress_bar=False,
                accelerator=accelerator,
                devices=devices,
            )
            preds = trainer.predict(self.estimator, X)
        return torch.cat(preds).numpy()
