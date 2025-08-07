from typing import Any, ClassVar, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger
from pydantic import field_validator
from torch_geometric.data import Batch
from torch_geometric.nn import (
    GATv2Conv,
    global_add_pool,
    global_max_pool,
    global_mean_pool,
)

from openadmet.models.architecture.model_base import (
    LightningModelBase,
    LightningModuleBase,
)
from openadmet.models.architecture.model_base import models as model_registry

_POOLING = {"mean": global_mean_pool, "max": global_max_pool, "add": global_add_pool}

# TODO: unify with the one in chemprop.py
_METRIC_TO_LOSS = {
    "mse": nn.MSELoss(),
    "mae": nn.L1Loss(),
    "huber": nn.HuberLoss(),
    "bce": nn.BCEWithLogitsLoss(),
    "cross_entropy": nn.CrossEntropyLoss(),
}


class GATv2Module(LightningModuleBase):
    """
    Graph Attention Network v2 (GATv2) Model
    """

    # Model architecture hyperparameters
    input_dim: int = 8 # must match that of GATGraphFeaturizer TODO: make this dynamic
    hidden_dim: int = 64
    num_layers: int = 3
    num_heads: int = 8
    dropout: float = 0.2
    pooling: str = "mean"
    output_dim: int = 1
    edge_dim: Optional[int] = 4 # must match that of GATGraphFeaturizer TODO: make this dynamic
    concat_heads: bool = True
    add_self_loops: bool = True
    share_weights: bool = True
    bias: bool = True

    # Training hyperparameters
    loss_function: str = "mse"
    optimizer: str = "adamw"
    optimizer_lr: float = 1e-3
    optimizer_weight_decay: float = 1e-5
    scheduler: str = "cosine"
    scheduler_factor: float = 0.5
    scheduler_patience: int = 10
    monitor_metric: str = "val_loss"

    @field_validator("pooling")
    @classmethod
    def validate_pooling(cls, value):
        """Validate pooling method"""
        allowed = ["mean", "max", "add"]
        if value not in allowed:
            raise ValueError(f"Pooling must be one of {allowed}")
        return value

    @field_validator("loss_function")
    @classmethod
    def validate_loss_function(cls, value):
        """Validate loss function"""
        allowed = ["mse", "mae", "huber", "bce", "cross_entropy"]
        if value not in allowed:
            raise ValueError(f"Loss function must be one of {allowed}")
        return value


    def __init__(
        self,
        input_dim: int = 8,
        hidden_dim: int = 64,
        num_layers: int = 3,
        num_heads: int = 8,
        dropout: float = 0.2,
        pooling: str = "mean",
        output_dim: int = 1,
        edge_dim: Optional[int] = 4,
        concat_heads: bool = True,
        add_self_loops: bool = True,
        share_weights: bool = True,
        bias: bool = True,
        loss_function: str = "mse",
        optimizer: str = "adamw",
        optimizer_lr: float = 1e-3,
        optimizer_weight_decay: float = 1e-5,
        scheduler: str = "cosine",
        scheduler_factor: float = 0.5,
        scheduler_patience: int = 10,
        monitor_metric: str = "val_loss",
    ):

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        self.pooling = pooling
        self.output_dim = output_dim
        self.edge_dim = edge_dim
        self.concat_heads = concat_heads
        self.add_self_loops = add_self_loops
        self.share_weights = share_weights
        self.bias = bias
        self.loss_function = loss_function
        self.optimizer = optimizer
        self.optimizer_lr = optimizer_lr
        self.optimizer_weight_decay = optimizer_weight_decay
        self.scheduler = scheduler
        self.scheduler_factor = scheduler_factor
        self.scheduler_patience = scheduler_patience
        self.monitor_metric = monitor_metric

        # Initialize super class
        super().__init__()

        # Input projection layer
        self.input_projection = nn.Linear(self.input_dim, self.hidden_dim)

        # GAT layers
        self.gat_layers = nn.ModuleList()

        for i in range(self.num_layers):
            # First and intermediate layers
            if i < self.num_layers - 1:
                in_channels = (
                    self.hidden_dim
                    if i == 0
                    else (
                        self.hidden_dim * self.num_heads
                        if self.concat_heads
                        else self.hidden_dim
                    )
                )
                out_channels = self.hidden_dim
                concat = self.concat_heads
            # Last layer
            else:
                in_channels = (
                    self.hidden_dim * self.num_heads
                    if self.concat_heads
                    else self.hidden_dim
                )
                out_channels = self.hidden_dim
                concat = False  # Don't concatenate in the last layer

            self.gat_layers.append(
                GATv2Conv(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    heads=self.num_heads,
                    concat=concat,
                    dropout=self.dropout,
                    edge_dim=self.edge_dim,
                    add_self_loops=self.add_self_loops,
                    share_weights=self.share_weights,
                    bias=self.bias,
                )
            )

        # Batch normalization layers
        self.batch_norms = nn.ModuleList()
        for i in range(self.num_layers):
            if i < self.num_layers - 1 and self.concat_heads:
                bn_dim = self.hidden_dim * self.num_heads
            else:
                bn_dim = self.hidden_dim
            self.batch_norms.append(nn.BatchNorm1d(bn_dim))

        # Output layers
        self.output_layers = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.BatchNorm1d(self.hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim // 2, self.output_dim),
        )

        # Pooling function
        self.pool = _POOLING[self.pooling]

    def forward(self, data):
        """
        Forward pass

        Args:
            data: PyTorch Geometric data object containing:
                - x: Node features [num_nodes, input_dim]
                - edge_index: Edge indices [2, num_edges]
                - batch: Batch indices [num_nodes]
                - edge_attr (optional): Edge features [num_edges, edge_dim]

        Returns:
            Graph-level predictions [batch_size, output_dim]
        """
        x, edge_index, batch = data.x, data.edge_index, data.batch
        edge_attr = getattr(data, "edge_attr", None)

        # Input projection
        x = self.input_projection(x)
        x = F.relu(x)

        # GAT layers
        for i, (gat_layer, bn) in enumerate(zip(self.gat_layers, self.batch_norms)):
            residual = x if i > 0 and x.size(-1) == gat_layer.out_channels else None

            x = gat_layer(x, edge_index, edge_attr=edge_attr)
            x = bn(x)

            if i < len(self.gat_layers) - 1:  # Don't apply activation on last layer
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)

            # Residual connection (if dimensions match)
            if residual is not None and residual.size(-1) == x.size(-1):
                x = x + residual

        # Graph-level pooling
        x = self.pool(x, batch)

        # Output layers
        out = self.output_layers(x)

        return out

    def training_step(self, batch: Batch, batch_idx: int):
        """Training step"""
        target = batch.y
        pred = self.forward(batch)

        if pred.ndim > 1 and pred.shape[1] == 1:
            pred = pred.squeeze(-1)
        if target.ndim > 1 and target.shape[1] == 1:
            target = target.squeeze(-1)

        loss = _METRIC_TO_LOSS[self.loss_function](pred, target)

        self.log(
            "train_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            batch_size=batch.num_graphs,
        )

        return loss

    def validation_step(self, batch: Batch, batch_idx: int):
        """Validation step"""
        target = batch.y

        pred = self.forward(batch)

        if pred.ndim > 1 and pred.shape[1] == 1:
            pred = pred.squeeze(-1)
        if target.ndim > 1 and target.shape[1] == 1:
            target = target.squeeze(-1)

        loss = _METRIC_TO_LOSS[self.loss_function](pred, target)

        self.log(
            "val_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=batch.num_graphs,
        )

        return loss

    def predict_step(self, batch: Batch, batch_idx: int):
        """Prediction step"""
        data = batch
        pred = self.forward(data)
        return pred


@model_registry.register("GATv2Model")
class GATv2Model(LightningModelBase):
    """
    GATv2 model wrapper inheriting from TorchModelBase
    """

    type: ClassVar[str] = "GATv2Model"
    scaler: Optional[Any] = None

    mod_params: dict = {}

    # Model architecture hyperparameters
    input_dim: Optional[int] = 8
    hidden_dim: int = 64
    num_layers: int = 3
    num_heads: int = 8
    dropout: float = 0.2
    pooling: str = "mean"
    output_dim: int = 1
    edge_dim: Optional[int] = 4
    concat_heads: bool = True
    add_self_loops: bool = True
    share_weights: bool = False
    bias: bool = True

    # Training hyperparameters
    loss_function: str = "mse"
    optimizer: str = "adamw"
    optimizer_lr: float = 1e-3
    optimizer_weight_decay: float = 1e-5
    scheduler: str = "cosine"
    scheduler_factor: float = 0.5
    scheduler_patience: int = 10
    monitor_metric: str = "val_loss"

    @classmethod
    def from_params(cls, class_params: dict = None, mod_params: dict = None):
        """
        Create model instance from parameters
        """

        instance = cls(**class_params, mod_params=mod_params)
        instance.build()
        return instance

    def build(self, scaler=None, **kwargs):
        """
        Builds the GATv2 model.
        'input_dim' is a mandatory parameter.
        """
        self.scaler = scaler

        if self.input_dim is None:
            raise ValueError("'input_dim' must be provided to build the GATv2 model.")

        if not self.estimator:
            # Build core GAT model
            self.estimator = GATv2Module(
                input_dim=self.input_dim,
                hidden_dim=self.hidden_dim,
                num_layers=self.num_layers,
                num_heads=self.num_heads,
                dropout=self.dropout,
                pooling=self.pooling,
                output_dim=self.output_dim,
                edge_dim=self.edge_dim,
                concat_heads=self.concat_heads,
                add_self_loops=self.add_self_loops,
                share_weights=self.share_weights,
                bias=self.bias,
                loss_function=self.loss_function,
                optimizer=self.optimizer,
                optimizer_lr=self.optimizer_lr,
                optimizer_weight_decay=self.optimizer_weight_decay,
                scheduler=self.scheduler,
                scheduler_factor=self.scheduler_factor,
                scheduler_patience=self.scheduler_patience,
                monitor_metric=self.monitor_metric,
            )

            logger.info(f"Built GATv2Model with config: {self.estimator.__dict__}")
        else:
            logger.warning("Model already exists, skipping build")


    def make_new(self) -> "GATv2Model":
        """
        Create a new instance of the model with the same parameters.
        This does not copy the estimator, only the configuration.
        """
        return self.__class__(**self.dict(exclude={"estimator"}))

    def train(self, dataloader):
        """
        Just see the mtenn.py for reference.

        This method exists only to satisfy the abstract base class contract.

        Use openadmet.models.trainer.lightning.LightningTrainer for training.
        """
        raise NotImplementedError(
            "GAT training is handled by LightningTrainer. "
            "Use: trainer = LightningTrainer(); trainer.train(model, dataloader)"
        )

    def predict(self, dataloader, accelerator="gpu", devices=1) -> np.ndarray:
        """
        Make predictions on a dataloader using the core GAT model.

        Args:
            dataloader (torch.utils.data.DataLoader): Dataloader to predict on.
            accelerator (str, optional): Accelerator to use. Defaults to "gpu".
            devices (int, optional): Number of devices to use. Defaults to 1.

        Returns:
            np.ndarray: Predictions.
        """
        if not hasattr(self, "estimator") or self.estimator is None:
            raise RuntimeError(
                "Model has not been built yet. Call `build` before `predict`."
            )

        # Set model to evaluation mode
        self.estimator.eval()

        # Determine device
        if accelerator == "gpu" and torch.cuda.is_available():
            device = torch.device("cuda")
        elif accelerator == "mps" and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")

        self.estimator.to(device)

        predictions = []

        with torch.no_grad():
            for batch in dataloader:
                # Move batch to device
                batch = batch.to(device)

                # Forward pass through core model
                pred = self.estimator(batch)

                # Move predictions to CPU and store
                predictions.append(pred.cpu())

        # Concatenate all predictions
        y_pred = torch.cat(predictions).numpy()

        # Apply inverse scaling if scaler is available
        if self.scaler is not None:
            y_pred = self.scaler.inverse_transform(y_pred)

        # Ensure correct shape
        if y_pred.ndim == 1:
            y_pred = y_pred.reshape(-1, 1)

        return y_pred

    def get_model_summary(self):
        """
        Get model summary information
        """
        if not self.estimator:
            return "Model not built"

        total_params = sum(p.numel() for p in self.estimator.parameters())
        trainable_params = sum(
            p.numel() for p in self.estimator.parameters() if p.requires_grad
        )

        summary = {
            "model_type": "GATv2 (Graph Attention Network v2)",
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "num_layers": self.num_layers,
            "num_attention_heads": self.num_heads,
            "hidden_dimension": self.hidden_dim,
            "pooling_method": self.pooling,
            "dropout_rate": self.dropout,
        }

        return summary
