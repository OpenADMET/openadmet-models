import json
from typing import ClassVar, Optional, List, Any
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, global_mean_pool, global_max_pool, global_add_pool
from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader
import lightning as pl
from lightning.pytorch.callbacks import TQDMProgressBar

from loguru import logger
from pydantic import field_validator
import pytest
import yaml

from openadmet.models.architecture.model_base import TorchModelBase
from openadmet.models.architecture.model_base import models as model_registry
from openadmet.models.features.gat_featurizer import GATGraphFeaturizer


class GATv2Model(nn.Module):
    """
    Graph Attention Network v2 (GATv2) Model
    """
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 3,
        num_heads: int = 8,
        dropout: float = 0.2,
        pooling: str = "mean",
        output_dim: int = 1,
        edge_dim: Optional[int] = None,
        concat_heads: bool = True,
        add_self_loops: bool = True,
        share_weights: bool = False,
        bias: bool = True
    ):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        self.pooling = pooling
        self.output_dim = output_dim
        self.concat_heads = concat_heads

        # Input projection layer
        self.input_projection = nn.Linear(input_dim, hidden_dim)

        # GAT layers
        self.gat_layers = nn.ModuleList()

        for i in range(num_layers):
            # First and intermediate layers
            if i < num_layers - 1:
                in_channels = hidden_dim if i == 0 else (hidden_dim * num_heads if concat_heads else hidden_dim)
                out_channels = hidden_dim
                concat = concat_heads
            # Last layer
            else:
                in_channels = hidden_dim * num_heads if concat_heads else hidden_dim
                out_channels = hidden_dim
                concat = False  # Don't concatenate in the last layer

            self.gat_layers.append(
                GATv2Conv(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    heads=num_heads,
                    concat=concat,
                    dropout=dropout,
                    edge_dim=edge_dim,
                    add_self_loops=add_self_loops,
                    share_weights=share_weights,
                    bias=bias
                )
            )

        # Batch normalization layers
        self.batch_norms = nn.ModuleList()
        for i in range(num_layers):
            if i < num_layers - 1 and concat_heads:
                bn_dim = hidden_dim * num_heads
            else:
                bn_dim = hidden_dim
            self.batch_norms.append(nn.BatchNorm1d(bn_dim))

        # Output layers
        final_dim = hidden_dim
        self.output_layers = nn.Sequential(
            nn.Linear(final_dim, final_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(final_dim // 2, output_dim)
        )

        # Pooling function
        if pooling == "mean":
            self.pool = global_mean_pool
        elif pooling == "max":
            self.pool = global_max_pool
        elif pooling == "add":
            self.pool = global_add_pool
        else:
            raise ValueError(f"Unsupported pooling method: {pooling}")

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
        edge_attr = getattr(data, 'edge_attr', None)

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


class GATv2LightningWrapper(pl.LightningModule):
    """
    Lightning wrapper for GAT model
    """
    def __init__(
        self,
        model_config: dict,
        loss_fn_name: str = "mse",
        lr: float = 1e-3,
        weight_decay: float = 1e-5,
        scheduler_name: str = "cosine",
        warmup_epochs: int = 10
    ):
        super().__init__()
        self.save_hyperparameters(ignore=['loss_fn_name'])
        
        self.loss_functions = {
            "mse": nn.MSELoss(), "mae": nn.L1Loss(), "huber": nn.HuberLoss(),
            "bce": nn.BCEWithLogitsLoss(), "cross_entropy": nn.CrossEntropyLoss()
        }
        
        self.model = GATv2Model(**model_config)
        self.loss_fn = self._get_loss_function(loss_fn_name)
        self.lr = lr
        self.weight_decay = weight_decay
        self.scheduler_name = scheduler_name
        self.warmup_epochs = warmup_epochs

    def _get_loss_function(self, name: str):
        if name.lower() not in self.loss_functions:
            raise ValueError(f"Unsupported loss function: {name}. Supported: {list(self.loss_functions.keys())}")
        return self.loss_functions[name.lower()]
    
    def forward(self, data: Batch):
        """Forward pass"""
        return self.model(data)

    def training_step(self, batch: Batch, batch_idx: int):
        """Training step"""
        target = batch.y

        pred = self(batch)

        if pred.ndim > 1 and pred.shape[1] == 1:
            pred = pred.squeeze(-1)
        if target.ndim > 1 and target.shape[1] == 1:
            target = target.squeeze(-1)

        loss = self.loss_fn(pred, target)

        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=batch.num_graphs)

        return loss

    def validation_step(self, batch: Batch, batch_idx: int):
        """Validation step"""
        target = batch.y

        pred = self(batch)

        if pred.ndim > 1 and pred.shape[1] == 1:
            pred = pred.squeeze(-1)
        if target.ndim > 1 and target.shape[1] == 1:
            target = target.squeeze(-1)

        loss = self.loss_fn(pred, target)

        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=batch.num_graphs)

        return loss

    def predict_step(self, batch: Batch, batch_idx: int):
        """Prediction step"""
        data = batch
        pred = self(data)
        return pred

    def configure_optimizers(self):
        """Configure optimizers and learning rate schedulers"""
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay
        )

        if self.scheduler_name == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=self.trainer.max_epochs if self.trainer else 100
            )
            return [optimizer], [{"scheduler": scheduler, "interval": "epoch"}]
        elif self.scheduler_name == "reduce_on_plateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=0.5, patience=10
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val_loss",
                    "interval": "epoch",
                    "frequency": 1
                }
            }
        elif self.scheduler_name == "none":
            return optimizer
        else:
            logger.warning(f"Unsupported scheduler: {self.scheduler_name}, using AdamW without LR scheduler.")
            return optimizer


@model_registry.register("GATv2ModelWrapper")
class GATv2ModelWrapper(TorchModelBase):
    """
    GATv2 model wrapper inheriting from TorchModelBase
    """

    type: ClassVar[str] = "GATv2ModelWrapper"
    scaler: Optional[Any] = None
    
    # Model hyperparameters
    input_dim: Optional[int] = None
    hidden_dim: int = 64
    num_layers: int = 3
    num_heads: int = 8
    gat_dropout: float = 0.2
    pooling: str = "mean"
    output_dim: int = 1
    edge_dim: Optional[int] = None
    concat_heads: bool = True
    add_self_loops: bool = True
    share_weights: bool = False
    bias: bool = True

    # Training hyperparameters
    lr: float = 1e-3
    weight_decay: float = 1e-5
    scheduler: str = "cosine"
    warmup_epochs: int = 10
    loss_function: str = "mse"

    @field_validator("pooling")
    @classmethod
    def validate_pooling(cls, value):
        """Validate pooling method"""
        if value not in ["mean", "max", "add"]:
            raise ValueError("pooling must be one of 'mean', 'max', or 'add'")
        return value

    @field_validator("scheduler")
    @classmethod
    def validate_scheduler(cls, value):
        """Validate learning rate scheduler"""
        if value not in ["cosine", "reduce_on_plateau", "none"]:
            raise ValueError("scheduler must be one of 'cosine', 'reduce_on_plateau', or 'none'")
        return value

    @field_validator("loss_function")
    @classmethod
    def validate_loss_function(cls, value):
        """Validate loss function"""
        if value not in ["mse", "mae", "huber", "bce", "cross_entropy"]:
            raise ValueError("loss_function must be one of 'mse', 'mae', 'huber', 'bce', or 'cross_entropy'")
        return value

    @classmethod
    def from_params(cls, class_params: dict = None, model_params: dict = None):
        """
        Create model instance from parameters
        """
        
        if class_params:
            instance = cls(**class_params)
        else:
            instance = cls()
        
        instance.build()
        return instance
    
    def build(self, scaler=None, **kwargs):
        """
        Builds the GATv2 model and lightning wrapper.
        'input_dim' is a mandatory parameter.
        """
        self.scaler = scaler

        if self.input_dim is None:
            raise ValueError("'input_dim' must be provided to build the GATv2 model.")

        if not self.estimator:
            # Prepare model configuration
            model_config = {
                "input_dim": self.input_dim,
                "hidden_dim": self.hidden_dim,
                "num_layers": self.num_layers,
                "num_heads": self.num_heads,
                "dropout": self.gat_dropout,
                "pooling": self.pooling,
                "output_dim": self.output_dim,
                "edge_dim": self.edge_dim,
                "concat_heads": self.concat_heads,
                "add_self_loops": self.add_self_loops,
                "share_weights": self.share_weights,
                "bias": self.bias,
            }

            self.estimator = GATv2LightningWrapper(
                model_config=model_config,
                loss_fn_name=self.loss_function,
                lr=self.lr,
                weight_decay=self.weight_decay,
                scheduler_name=self.scheduler,
                warmup_epochs=self.warmup_epochs
            )

            logger.info(f"Built GATv2LightningWrapper with GATv2Model config: {model_config}")
        else:
            logger.warning("Model already exists, skipping build")

    def train(self, dataloader):
        """
        Train the model
        """
        raise NotImplementedError(
            "Training not implemented in model class, use a trainer"
        )

    def predict(self, dataloader, accelerator="gpu", devices=1) -> np.ndarray:
        """
        Make predictions on a dataloader.

        Args:
            dataloader (torch.utils.data.DataLoader): Dataloader to predict on.
            accelerator (str, optional): Accelerator to use. Defaults to "gpu".
            devices (int, optional): Number of devices to use. Defaults to 1.

        Returns:
            np.ndarray: Predictions.
        """
        if not hasattr(self, 'estimator') or self.estimator is None:
            raise RuntimeError("Model has not been built yet. Call 'build()' before 'predict'.")

        # Create a PyTorch Lightning Trainer for prediction
        progress_bar = TQDMProgressBar(refresh_rate=10)
        trainer = pl.Trainer(
            accelerator=accelerator, 
            devices=devices, 
            callbacks=[progress_bar],
            logger=False
        )

        # Make predictions
        preds = trainer.predict(self.estimator, dataloader)
        
        y_pred = torch.cat(preds).cpu().numpy()

        if self.scaler is not None:
            y_pred = self.scaler.inverse_transform(y_pred)

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
        trainable_params = sum(p.numel() for p in self.estimator.parameters() if p.requires_grad)

        summary = {
            "model_type": "GATv2 (Graph Attention Network v2)",
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "num_layers": self.num_layers,
            "num_attention_heads": self.num_heads,
            "hidden_dimension": self.hidden_dim,
            "pooling_method": self.pooling,
            "dropout_rate": self.gat_dropout
        }

        return summary


if __name__ == "__main__":
    import pandas as pd
    from sklearn.model_selection import train_test_split

    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        default_yaml_path = os.path.join(script_dir, '..', 'tests', 'test_data', 'basic_anvil_gat.yaml')
        # Since run_workflow_from_config is removed, this part will be cleaned up.
        # Placeholder for potential future main-guard use.
    except Exception as e:
        import traceback
        logger.error(f"An error occurred: {e}")
        logger.error(f"Full Traceback:\n{traceback.format_exc()}") 