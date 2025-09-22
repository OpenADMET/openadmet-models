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

    This class implements a ChemProp-based regression model using message passing neural networks (MPNNs)
    for molecular property prediction. It supports various configurations for message passing, aggregation,
    and feed-forward network (FFN) layers. Can be initialized from the CheMeleon foundation model [REF], overriding
    settings for depth, message hidden dim, messages, and aggregation.

    Attributes
    ----------
    type : str
        The type of the model.
    n_tasks : int
        Number of prediction tasks.
    messages : str
        Type of message passing ("bond" or "atom").
    aggregation : str
        Aggregation method ("mean" or "norm").
    depth : int
        Number of message passing steps.
    message_hidden_dim : int
        Hidden dimension size for message passing.
    ffn_hidden_dim : int
        Hidden dimension size for the feed-forward network.
    ffn_num_layers : int
        Number of layers in the feed-forward network.
    scaler : object
        Scaler used for target normalization.
    normalized_targets : bool
        Whether targets are normalized.
    batch_norm : bool
        Whether to use batch normalization.
    dropout : float
        Dropout rate.
    from_chemeleon : bool
        Whether to use the CheMeleon foundation model.
    monitor_metric : str
        The metric to monitor during training.
    metric_list : list
        List of metrics to use for evaluation.
    warmup_epochs : int
        Number of warmup epochs for learning rate scheduling.
    init_lr : float
        Initial learning rate.
    max_lr : float
        Maximum learning rate.
    final_lr : float
        Final learning rate.

    """

    type: ClassVar[str] = "ChemPropModel"
    n_tasks: int = 1
    messages: str = "bond"
    aggregation: str = "norm"
    depth: int = 3
    message_hidden_dim: int = 300
    ffn_hidden_dim: int = 300
    ffn_num_layers: int = 1
    scaler: object = None
    normalized_targets: bool = True
    batch_norm: bool = False
    dropout: float = 0.0
    from_chemeleon: bool = False
    monitor_metric: str = "val_loss"
    metric_list: list = ["mse", "mae", "rmse"]
    warmup_epochs: int = 2
    init_lr: float = 1e-4
    max_lr: float = 1e-3
    final_lr: float = 1e-4

    @field_validator("messages")
    @classmethod
    def validate_messages(cls, value):
        """
        Validate the messages parameter.

        Parameters
        ----------
        value : str
            The value to validate.

        Returns
        -------
        str
            The validated value.

        Raises
        ------
        ValueError
            If value is not "bond" or "atom".

        """
        if value not in ["bond", "atom"]:
            raise ValueError("Messages must be either 'bond' or 'atom'")
        return value

    @field_validator("aggregation")
    @classmethod
    def validate_aggregation(cls, value):
        """
        Validate the aggregation parameter.

        Parameters
        ----------
        value : str
            The value to validate.

        Returns
        -------
        str
            The validated value.

        Raises
        ------
        ValueError
            If value is not "mean" or "norm".

        """
        if value not in ["mean", "norm"]:
            raise ValueError("Aggregation must be either 'mean' or 'norm'")
        return value

    def __init__(self, *args, **kwargs):
        """
        Initialize the ChemPropModel.

        Parameters
        ----------
        *args : tuple
            Positional arguments for parent class.
        **kwargs : dict
            Keyword arguments for parent class and model configuration.

        """
        super().__init__(*args, **kwargs)
        self.build()

    def _get_output_transform(self):
        """
        Set the output transform for predictions.

        Returns
        -------
        nn.UnscaleTransform or None
            The output transform to apply to predictions.

        """
        if self.scaler is not None:
            output_transform = nn.UnscaleTransform.from_standard_scaler(self.scaler)
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

        .. deprecated:: 1.0
            Use direct initialization instead.

        Parameters
        ----------
        class_params : dict, optional
            Class-level parameters.
        mod_params : dict, optional
            Model-specific parameters.

        Raises
        ------
        DeprecationWarning
            Always raised as this method is deprecated.

        """
        raise DeprecationWarning("Deprecating `from_params` method.")

    def make_new(self) -> "ChemPropModel":
        """
        Copy parameters to a new model instance without copying the estimator.

        Returns
        -------
        ChemPropModel
            A new instance of ChemPropModel with the same parameters.

        """
        return self.__class__(**self.model_dump(exclude={"estimator"}))

    def train(self, dataloader, scaler=None):
        """
        Train the model.

        Parameters
        ----------
        dataloader : DataLoader
            DataLoader for training data.
        scaler : object, optional
            Scaler for target normalization.

        Raises
        ------
        NotImplementedError
            Always raised. Use a trainer for training.

        """
        raise NotImplementedError(
            "Training not implemented in model class, use a trainer"
        )

    def build(self):
        """
        Prepare and build the ChemProp model.

        Downloads and loads the CheMeleon foundation model if specified, otherwise
        constructs a new MPNN model with the given configuration.

        Returns
        -------
        self : ChemPropModel
            The current instance with the estimator built.

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
                output_transform=self._get_output_transform(),
                dropout=self.dropout,
            )

            # Create the MPNN model
            mpnn = models.MPNN(
                message_passing=mp,
                agg=aggr,
                predictor=ffn,
                batch_norm=self.batch_norm,
                metrics=metric_list,
                warmup_epochs=self.warmup_epochs,
                init_lr=self.init_lr,
                max_lr=self.max_lr,
                final_lr=self.final_lr,
            )

            # Pass monitor metric from "model" to "module"
            # This is necessary to support subclasses of LightningModuleBase, as `monitor_metric`
            # is needed at the "module" level for use in both `configure_optimizers` and `LightningTrainer`
            mpnn.monitor_metric = self.monitor_metric
            self.estimator = mpnn

        else:
            logger.warning("Model already exists, skipping build")

        return self

    def predict(
        self, X: np.ndarray, accelerator="gpu", devices=1, **kwargs
    ) -> np.ndarray:
        """
        Predict using the trained model.

        Parameters
        ----------
        X : np.ndarray
            Input data for prediction.
        accelerator : str, optional
            Accelerator type to use ("gpu" or "cpu").
        devices : int, optional
            Number of devices to use for prediction.
        **kwargs
            Additional keyword arguments for the trainer.

        Returns
        -------
        np.ndarray
            Model predictions.

        Raises
        ------
        AttributeError
            If the model is not trained or built.

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


def freeze_weights(
    self, message_passing: bool = True, batch_norm: bool = True, ffn_layers: int = 1
):
    """
    Freeze parts of the model for transfer learning or fine-tuning.

    Parameters
    ----------
    message_passing : bool, optional
        If True, freeze the message passing layers. Default is True.
    batch_norm : bool, optional
        If True, freeze the batch normalization layers. Default is True.
    ffn_layers : int, optional
        Number of feed-forward network (FFN) layers to freeze. Default is 1.

    Notes
    -----
    This method sets the `requires_grad` attribute of the specified layers to False,
    preventing their weights from being updated during training. It also sets these
    layers to evaluation mode.

    """
    # Freeze message passing
    if message_passing:
        self.estimator.message_passing.apply(
            lambda module: module.requires_grad_(False)
        )
        self.estimator.message_passing.eval()

    # Freeze batch norm
    if batch_norm:
        self.estimator.bn.apply(lambda module: module.requires_grad_(False))
        self.estimator.bn.eval()

    # Freeze feedforward network
    for idx in range(ffn_layers):
        self.estimator.predictor.ffn[idx].requires_grad_(False)
        self.estimator.predictor.ffn[idx + 1].eval()
        # What if index out of range?

    # TODO: better logging
    logger.info(
        f"Model weights for {message_passing=}, {batch_norm=}, {ffn_layers=} frozen"
    )
