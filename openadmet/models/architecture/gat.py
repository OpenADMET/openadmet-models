import json
from typing import ClassVar, Optional, List, Any
from pathlib import Path
import numpy as np
import os
# os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

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
        
        self.model = GATv2Model(**model_config)
        self.loss_fn = self._get_loss_function(loss_fn_name)
        self.lr = lr
        self.weight_decay = weight_decay
        self.scheduler_name = scheduler_name
        self.warmup_epochs = warmup_epochs
        
    def _get_loss_function(self, name: str):
        loss_functions = {
            "mse": nn.MSELoss(), "mae": nn.L1Loss(), "huber": nn.HuberLoss(),
            "bce": nn.BCEWithLogitsLoss(), "cross_entropy": nn.CrossEntropyLoss()
        }
        if name.lower() not in loss_functions:
            raise ValueError(f"Unsupported loss function: {name}. Supported: {list(loss_functions.keys())}")
        return loss_functions[name.lower()]
    
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
        
        # Handle case where target is None
        if target is None:
            logger.warning(f"Target is None in validation batch {batch_idx}, skipping")
            return None
        
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
        
        merged_params = {}
        if class_params:
            merged_params.update(class_params)
        
        instance = cls(**merged_params) 
        
        instance.build(passed_model_params=model_params) 
        return instance
    
    def build(self, scaler=None, passed_model_params: Optional[dict] = None, train_dataloader=None):
        """
        Build the model
        """
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
            if passed_model_params: 
                model_config.update(passed_model_params)

            # Try to infer input_dim from train_dataloader if not provided
            if model_config.get("input_dim") is None and train_dataloader is not None:
                try:
                    sample_batch = next(iter(train_dataloader))
                    if hasattr(sample_batch, 'x') and sample_batch.x is not None:
                        model_config["input_dim"] = sample_batch.x.shape[1]
                        logger.info(f"Inferred input_dim from data: {model_config['input_dim']}")
                    if hasattr(sample_batch, 'edge_attr') and sample_batch.edge_attr is not None:
                        model_config["edge_dim"] = sample_batch.edge_attr.shape[1]
                        logger.info(f"Inferred edge_dim from data: {model_config['edge_dim']}")
                except Exception as e:
                    logger.warning(f"Could not infer dimensions from dataloader: {e}")

            if model_config.get("input_dim") is None:
                raise ValueError("input_dim must be specified or inferrable from training data")

            self.estimator = GATv2LightningWrapper(
                model_config=model_config,
                loss_fn_name=self.loss_function,
                lr=self.lr,
                weight_decay=self.weight_decay,
                scheduler_name=self.scheduler,
                warmup_epochs=self.warmup_epochs
            )
            
            logger.info(f"Built GATv2LightningWrapper with GATv2Model config: {model_config}")
            logger.info(f"LightningWrapper params: lr={self.lr}, loss={self.loss_function}, scheduler={self.scheduler}")
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
        Use model for prediction
        
        Args:
            dataloader: PyTorch Geometric DataLoader
            accelerator: Accelerator type ("gpu", "cpu")
            devices: Number of devices
        
        Returns:
            Numpy array of predictions
        """
        if not self.estimator:
            raise AttributeError("Model not built or trained")
        
        with torch.inference_mode():
            trainer = pl.Trainer(
                logger=None,
                enable_progress_bar=False,
                accelerator=accelerator,
                devices=devices
            )
            preds = trainer.predict(self.estimator, dataloader)
        
        # Return predictions as 2D array (samples, 1) to match evaluator expectations
        preds_array = torch.cat(preds).cpu().numpy()
        if preds_array.ndim == 1:
            preds_array = preds_array.reshape(-1, 1)
        return preds_array
    
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


def test_gat_from_yaml():
    """Test GATv2ModelWrapper creation from YAML-like configuration"""
    yaml_config_model_params = {
        "input_dim": 128, 
        "hidden_dim": 64,
        "num_layers": 3,
        "num_heads": 8,
        "gat_dropout": 0.2,
        "lr": 0.001
    }
    
    gat_model_wrapper = GATv2ModelWrapper.from_params(class_params=yaml_config_model_params)
    
    assert gat_model_wrapper.type == "GATv2ModelWrapper"
    assert gat_model_wrapper.hidden_dim == 64
    assert gat_model_wrapper.num_heads == 8
    assert gat_model_wrapper.lr == 0.001
    assert gat_model_wrapper.estimator is not None 
    assert gat_model_wrapper.estimator.model.hidden_dim == 64


def test_gat_yaml_validation():
    """Test GATv2ModelWrapper YAML parameter validation"""
    with pytest.raises(ValueError): # Pydantic's ValidationError
        GATv2ModelWrapper(pooling="invalid_pooling_method")
    
    with pytest.raises(ValueError):
        GATv2ModelWrapper(scheduler="invalid_scheduler_name")

    with pytest.raises(ValueError):
        GATv2ModelWrapper(loss_function="invalid_loss_fn")


def load_gat_from_yaml_file(yaml_path: str, model_params: Optional[dict] = None):
    """Load GAT model from YAML file (Illustrative)"""
    with open(yaml_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Assuming YAML structure like:
    # procedure:
    #   model:
    #     type: GATv2ModelWrapper
    #     params:
    #       hidden_dim: 64
    #       ...
    
    if "procedure" not in config or "model" not in config["procedure"] or "params" not in config["procedure"]["model"]:
        raise ValueError("YAML file must contain procedure.model.params section")
        
    wrapper_params = config["procedure"]["model"].get("params", {})
    return GATv2ModelWrapper.from_params(class_params=wrapper_params, model_params=model_params)


def run_workflow_from_config(config_path: str):
    """
    Runs the entire training and evaluation workflow from a YAML configuration file.
    """
    # --- 1. Load Configuration ---
    logger.info(f"Loading workflow configuration from {config_path}")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    data_cfg = config['data']
    proc_cfg = config['procedure']
    
    # --- 2. Data Loading and Preprocessing ---
    config_dir = os.path.dirname(os.path.abspath(config_path))
    csv_file_path = os.path.join(config_dir, data_cfg['resource'])
    
    try:
        df = pd.read_csv(csv_file_path)
        df.dropna(subset=[data_cfg['input_col'], data_cfg['target_col']], inplace=True)
        all_smiles = df[data_cfg['input_col']].tolist()
        all_y = df[data_cfg['target_col']].astype(float).tolist()
        logger.info(f"Loaded {len(all_smiles)} samples from {csv_file_path}")
    except Exception as e:
        logger.error(f"Error loading or processing CSV file: {e}. Exiting.")
        return

    max_samples = data_cfg.get('max_samples')
    if max_samples and len(all_smiles) > max_samples:
        logger.info(f"Using the first {max_samples} samples for this run.")
        smiles_subset, y_subset = all_smiles[:max_samples], all_y[:max_samples]
    else:
        smiles_subset, y_subset = all_smiles, all_y

    if not smiles_subset:
        logger.error("No valid SMILES data to process. Exiting.")
        return

    # --- 3. Featurization ---
    feat_cfg = proc_cfg['feat']
    if feat_cfg['type'] == 'GATGraphFeaturizer':
        featurizer = GATGraphFeaturizer(**feat_cfg.get('params', {}))
        graph_data_list = featurizer.featurize(smiles_subset, y_subset)
    else:
        raise NotImplementedError(f"Featurizer '{feat_cfg['type']}' not implemented.")

    if not graph_data_list or len(graph_data_list) < 3:
        logger.error("Not enough data for train/validation/test split after featurization. Exiting.")
        return
        
    # --- 4. Data Splitting ---
    split_params = proc_cfg['split']['params']
    train_size = split_params['train_size']
    val_size = split_params['val_size']
    test_size = 1.0 - train_size - val_size

    if not (train_size > 0 and val_size > 0 and test_size > 0):
        raise ValueError("train_size, val_size, and test_size must all be positive.")

    train_val_size = train_size + val_size
    train_val_data, test_data = train_test_split(
        graph_data_list,
        train_size=train_val_size,
        random_state=split_params['random_state'],
        shuffle=split_params['shuffle']
    )
    val_split_ratio = val_size / train_val_size
    train_data, val_data = train_test_split(
        train_val_data,
        test_size=val_split_ratio,
        random_state=split_params['random_state'],
        shuffle=split_params['shuffle']
    )
    logger.info(f"Data split: {len(train_data)} train, {len(val_data)} validation, {len(test_data)} test samples.")

    # --- 5. Create DataLoaders ---
    train_params = proc_cfg['train']['params']
    batch_size = train_params.get('batch_size', 32)
    num_workers = train_params.get('num_workers', 0)

    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_dataloader = DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # --- 6. Model Preparation ---
    first_graph = train_data[0]
    determined_input_dim = first_graph.x.shape[1] if hasattr(first_graph, 'x') and first_graph.x is not None else 8
    determined_edge_dim = first_graph.edge_attr.shape[1] if hasattr(first_graph, 'edge_attr') and first_graph.edge_attr is not None else None
    
    dynamic_params = {"input_dim": determined_input_dim, "edge_dim": determined_edge_dim}
    gat_model_wrapper = load_gat_from_yaml_file(config_path, model_params=dynamic_params)

    # --- 7. Configure and Run Trainer ---
    trainer_params = {
        "max_epochs": train_params['max_epochs'],
        "accelerator": train_params['accelerator'],
        "devices": train_params['devices'],
        "logger": True,
        "enable_progress_bar": True,
        "callbacks": [TQDMProgressBar(refresh_rate=10)],
        "log_every_n_steps": max(1, len(train_dataloader) // 10 if train_dataloader and len(train_dataloader) > 10 else 1)
    }
    trainer = pl.Trainer(**trainer_params)

    logger.info(f"Starting training for {trainer_params['max_epochs']} epochs...")
    trainer.fit(
        model=gat_model_wrapper.estimator,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader
    )
    logger.info("Training completed.")

    # --- 8. Evaluation ---
    logger.info(f"Evaluating model on the test set ({len(test_data)} samples)...")
    test_results = trainer.validate(model=gat_model_wrapper.estimator, dataloaders=test_dataloader, verbose=False)
    if test_results and isinstance(test_results, list):
        test_loss = test_results[0].get('val_loss', float('nan'))
        logger.info(f"Loss on the test set: {test_loss:.4f}")
    
    return gat_model_wrapper, trainer, test_results

if __name__ == "__main__":
    import pandas as pd
    from sklearn.model_selection import train_test_split
    
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        default_yaml_path = os.path.join(script_dir, '..', 'tests', 'test_data', 'basic_anvil_gat.yaml')
        run_workflow_from_config(default_yaml_path)
    except Exception as e:
        import traceback
        logger.error(f"An error occurred during the workflow execution: {e}")
        logger.error(f"Full Traceback:\n{traceback.format_exc()}") 