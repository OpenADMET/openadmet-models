"""ChemProp and Chemeleon model implementations."""

import json
import types
from pathlib import Path
from typing import ClassVar
from urllib.request import urlretrieve

import numpy as np
import torch
from chemprop import models, nn
from chemprop.models.model import build_NoamLike_LRSched
from lightning import pytorch as pl
from loguru import logger
from pydantic import PrivateAttr, field_validator, model_validator

from openadmet.models.architecture.model_base import LightningModelBase
from openadmet.models.architecture.model_base import models as model_registry

_METRIC_TO_LOSS = {
    "mse": nn.metrics.MSE(),
    "mae": nn.metrics.MAE(),
    "rmse": nn.metrics.RMSE(),
}


def configure_optimizers(self):
    """
    Configure optimizers and learning rate schedulers.

    Returns
    -------
    dict
        A dictionary containing the optimizer and learning rate scheduler configurations.

    """
    # Separate parameters into MPNN and FFN groups.
    mpnn_params = []
    ffn_params = []

    for name, param in self.named_parameters():
        if "predictor" in name:
            ffn_params.append(param)
        else:
            mpnn_params.append(param)

    # Set the optimizer base learning rates to their peak values.
    param_groups = [
        {
            "params": mpnn_params,
            "lr": self.mpnn_lr,
            "weight_decay": self.mpnn_weight_decay,
        },
        {
            "params": ffn_params,
            "lr": self.ffn_lr,
            "weight_decay": self.ffn_weight_decay,
        },
    ]

    opt = torch.optim.AdamW(param_groups)

    if self.scheduler == "plateau":
        # Configure the reduce on plateau scheduler.
        lr_sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt,
            mode="min",
            factor=self.reduce_lr_factor,
            patience=self.reduce_lr_patience,
            min_lr=self.final_lr,
        )

        lr_sched_config = {
            "scheduler": lr_sched,
            "monitor": self.monitor_metric,
            "interval": "epoch",
            "frequency": 1,
        }
    elif self.scheduler == "noam":
        # Calculate steps per epoch safely using trainer properties.
        if isinstance(
            self.trainer.estimated_stepping_batches, int
        ) and self.trainer.estimated_stepping_batches != float("inf"):
            total_steps = self.trainer.estimated_stepping_batches
            steps_per_epoch = total_steps // max(1, self.trainer.max_epochs)
        else:
            # Fallback for infinite training or uninitialized dataloaders.
            steps_per_epoch = getattr(self.trainer, "num_training_batches", 1000)

        warmup_steps = self.warmup_epochs * steps_per_epoch

        if self.trainer.max_epochs == -1:
            logger.warning(
                "Setting cooldown epochs to 100 times the warmup epochs for infinite training."
            )
            cooldown_steps = 100 * warmup_steps
        else:
            cooldown_epochs = self.trainer.max_epochs - self.warmup_epochs
            cooldown_steps = cooldown_epochs * steps_per_epoch

        # Convert vanilla absolute learning rates into relative scaling factors.
        init_factor = self.init_lr / self.max_lr
        final_factor = self.final_lr / self.max_lr

        # Define the lambda function using relative factors scaling up to 1.0 at peak.
        def lr_lambda(step: int):
            if step < warmup_steps:
                warmup_slope = (1.0 - init_factor) / max(1, warmup_steps)
                return init_factor + (step * warmup_slope)

            elif warmup_steps <= step < warmup_steps + cooldown_steps:
                decay_steps = step - warmup_steps
                cooldown_slope = final_factor ** (1.0 / max(1, cooldown_steps))
                return 1.0 * (cooldown_slope**decay_steps)

            else:
                return final_factor

        lr_sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
        lr_sched_config = {"scheduler": lr_sched, "interval": "step"}

    return {"optimizer": opt, "lr_scheduler": lr_sched_config}


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
    normalized_targets : bool
        Whether targets are normalized.
    batch_norm : bool
        Whether to use batch normalization.
    dropout : float
        Dropout rate.
    from_chemeleon : bool
        Whether to use the CheMeleon foundation model.
    monitor_metric : str
        The metric to monitor during training. Default is "val_loss".
    metric_list : list
        List of metrics to use for evaluation. Default is ["mse", "mae", "rmse"].
    scheduler : str
        Learning rate scheduler ("noam" or "plateau"). Default is "noam".
    warmup_epochs : int
        Number of warmup epochs for learning rate scheduling (Noam scheduler only). Default is 2.
    init_lr : float, optional
        Initial learning rate. If None, defaults to max_lr * 0.1.
    max_lr : float
        Maximum learning rate (Global default). Default is 1e-3.
    final_lr : float, optional
        Final learning rate. If None, defaults to max_lr * 0.01.
    weight_decay : float
        Global weight decay. Default is 0.0.
    mpnn_lr : float, optional
        Learning rate for MPNN. If None, defaults to max_lr.
    ffn_lr : float, optional
        Learning rate for FFN. If None, defaults to max_lr.
    mpnn_weight_decay : float, optional
        Weight decay for MPNN. If None, defaults to weight_decay.
    ffn_weight_decay : float, optional
        Weight decay for FFN. If None, defaults to weight_decay.
    reduce_lr_factor : float
        Factor by which the learning rate will be reduced (Plateau scheduler only). Default is 0.1.
    reduce_lr_patience : int
        Number of epochs with no improvement after which learning rate will be reduced (Plateau scheduler only). Default is 10.

    """

    # Meta parameters for this class
    type: ClassVar[str] = "ChemPropModel"

    # ChemProp parameters
    n_tasks: int = 1
    messages: str = "bond"
    aggregation: str = "norm"
    depth: int = 3
    message_hidden_dim: int = 300
    ffn_hidden_dim: int = 300
    ffn_num_layers: int = 1
    normalized_targets: bool = True
    batch_norm: bool = False
    dropout: float = 0.0
    from_chemeleon: bool = False
    monitor_metric: str = "val_loss"
    metric_list: list = ["mse", "mae", "rmse"]

    # Select scheduler among "noam" or "plateau"
    scheduler: str = "noam"

    # Global defaults (master values)
    max_lr: float = 1e-3
    weight_decay: float = 0.0

    # Component overrides (optional - inherit from masters if None)
    mpnn_lr: float | None = None
    ffn_lr: float | None = None
    mpnn_weight_decay: float | None = None
    ffn_weight_decay: float | None = None

    # Scheduler specifics (optional - inherit from max_lr if None)
    init_lr: float | None = None
    final_lr: float | None = None

    # Noam-only parameters
    warmup_epochs: int = 2

    # Plateau-only parameters
    reduce_lr_factor: float = 0.1
    reduce_lr_patience: int = 10

    _n_tasks: int = 1
    _explicit_init_fields: set[str] = PrivateAttr(default_factory=set)

    def __init__(self, **data):
        """Initialize the model and track explicitly provided fields."""
        explicit_init_fields = set(data)
        super().__init__(**data)
        self._explicit_init_fields = explicit_init_fields.intersection(
            type(self).model_fields.keys()
        )

    @model_validator(mode="after")
    def resolve_hyperparameters(self) -> "ChemPropModel":
        """
        Resolve hyperparameters using global defaults and component overrides pattern.

        Logic:
        - Resolve learning rates:
            - init_lr -> max_lr * 0.1
            - final_lr -> max_lr * 0.01
            - mpnn_lr -> max_lr
            - ffn_lr -> max_lr
        - Resolve weight decays:
            - mpnn_weight_decay -> weight_decay
            - ffn_weight_decay -> weight_decay
        """
        # Resolve LRs
        if self.init_lr is None:
            self.init_lr = self.max_lr * 0.1
        if self.final_lr is None:
            self.final_lr = self.max_lr * 0.01
        if self.mpnn_lr is None:
            self.mpnn_lr = self.max_lr
        if self.ffn_lr is None:
            self.ffn_lr = self.max_lr

        # Resolve weight decays
        if self.mpnn_weight_decay is None:
            self.mpnn_weight_decay = self.weight_decay
        if self.ffn_weight_decay is None:
            self.ffn_weight_decay = self.weight_decay

        return self

    @model_validator(mode="after")
    def validate_scheduler_params(self) -> "ChemPropModel":
        """Ensure scheduler-specific parameters are valid for the chosen scheduler."""
        if self.scheduler == "noam":
            # Check for plateau params
            if "reduce_lr_factor" in self.model_fields_set:
                raise ValueError(
                    "reduce_lr_factor is not compatible with noam scheduler"
                )
            if "reduce_lr_patience" in self.model_fields_set:
                raise ValueError(
                    "reduce_lr_patience is not compatible with noam scheduler"
                )
        elif self.scheduler == "plateau":
            # Check for noam params
            if "warmup_epochs" in self.model_fields_set:
                raise ValueError(
                    "warmup_epochs is not compatible with plateau scheduler"
                )
            if self.reduce_lr_factor >= 1.0:
                raise ValueError("reduce_lr_factor must be < 1.0 for plateau scheduler")
        return self

    @model_validator(mode="after")
    def set_n_tasks(self) -> "ChemPropModel":
        """
        Set the number of tasks for the model.

        Returns
        -------
        ChemPropModel
            The updated model instance.

        """
        self._n_tasks = self.n_tasks
        return self

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

        """
        if value not in ["mean", "norm"]:
            raise ValueError("Aggregation must be either 'mean' or 'norm'")
        return value

    @field_validator("scheduler")
    @classmethod
    def validate_scheduler(cls, value):
        """
        Validate the scheduler parameter.

        Parameters
        ----------
        value : str
            The value to validate.

        Returns
        -------
        str
            The validated value.

        """
        if value not in ["noam", "plateau"]:
            raise ValueError("Scheduler must be either 'noam' or 'plateau'")
        return value

    def _get_output_transform(self, scaler):
        """
        Convert scaler to the output transform needed for predictions.

        Parameters
        ----------
        scaler : object
            The scaler to use for unscaling predictions.

        Returns
        -------
        nn.UnscaleTransform or None
            The output transform to apply to predictions.

        """
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

    def build(self, scaler=None):
        """
        Prepare and build the ChemProp model.

        Downloads and loads the CheMeleon foundation model if specified, otherwise
        constructs a new MPNN model with the given configuration.

        Parameters
        ----------
        scaler : object, optional
            Scaler for target normalization.

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
                output_transform=self._get_output_transform(scaler),
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

            # Attach custom optimization parameters to the MPNN instance
            mpnn.mpnn_weight_decay = self.mpnn_weight_decay
            mpnn.ffn_weight_decay = self.ffn_weight_decay
            mpnn.mpnn_lr = self.mpnn_lr
            mpnn.ffn_lr = self.ffn_lr
            mpnn.reduce_lr_factor = self.reduce_lr_factor
            mpnn.reduce_lr_patience = self.reduce_lr_patience
            mpnn.scheduler = self.scheduler  # Propagate scheduler choice

            # Bind the custom configure_optimizers method
            mpnn.configure_optimizers = types.MethodType(configure_optimizers, mpnn)

            self.estimator = mpnn

        else:
            logger.warning("Model already exists, skipping build")

        return self

    def train(self, dataloader, scaler=None):
        """
        Train the model.

        Parameters
        ----------
        dataloader : DataLoader
            DataLoader for training data.
        scaler : object, optional
            Scaler for target normalization.

        """
        raise NotImplementedError(
            "Training not implemented in model class, use a trainer"
        )

    def serialize(self, param_path="model.json", serial_path="model.pth"):
        """
        Save the model with only explicitly provided parameter fields.

        Parameters
        ----------
        param_path: PathLike
            Path to save the model parameters to
        serial_path: PathLike
            Path to save the serialized model to

        """
        explicit_params = self.model_dump(include=self._explicit_init_fields)
        with open(param_path, "w") as f:
            json.dump(explicit_params, f, indent=2)
        self.save(serial_path)

    def make_new(self) -> "ChemPropModel":
        """Copy parameters to a new model instance without copying the estimator."""
        explict_params = self.model_dump(
            include=self._explicit_init_fields, exclude={"estimator"}
        )
        return self.__class__(**explict_params)

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
        self, message_passing: bool = True, batch_norm: bool = True, ffn_layers: int = 0
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
            Number of feed-forward network (FFN) layers to freeze. Default is 0.

        Notes
        -----
        This method sets the `requires_grad` attribute of the specified layers to False,
        preventing their weights from being updated during training. It also sets these
        layers to evaluation mode.

        """
        # Check number of layers
        if ffn_layers > self.ffn_num_layers:
            raise ValueError(
                f"Requested to freeze {ffn_layers} feedforward network layer(s), "
                f"but only {self.ffn_num_layers} available."
            )

        # Freeze message passing
        if message_passing:
            # No gradient updates
            self.estimator.message_passing.apply(
                lambda module: module.requires_grad_(False)
            )
            # Set to evaluation mode
            self.estimator.message_passing.eval()

            # Log for message passing
            logger.info(f"Model weights for message passing frozen.")

        # Freeze batch norm
        if batch_norm:
            # No gradient updates
            self.estimator.bn.apply(lambda module: module.requires_grad_(False))
            # Evaluation mode
            self.estimator.bn.eval()
            # Log for batch normalization
            logger.info(f"Model weights for batch normalization frozen.")

        # Freeze feedforward network
        if ffn_layers > 0:
            for idx in range(ffn_layers):
                # No gradient updates
                self.estimator.predictor.ffn[idx].requires_grad_(False)
                # Evaluation mode
                self.estimator.predictor.ffn[idx + 1].eval()

            # Log for feedforward network
            logger.info(
                f"Model weights for {ffn_layers} feedforward network layer(s) frozen."
            )
