import numpy as np
import torch
from lightning import pytorch as pl

from openadmet.models.architecture.model_base import TorchModelBase
from openadmet.models.architecture.model_base import models as model_registry

from collections import OrderedDict


class NeuralPairwiseRegressor(pl.LightningModule):
    def __init__(self,
                 input_size: int,
                 hidden_size: int,
                 num_layers: int,
                 activation: torch.nn.Module = torch.nn.ReLU,
                 lr: float = 1e-3,
                 n_targets: int = 1):
        super().__init__()
        _modules = OrderedDict()
        for i in range(num_layers):
            _modules[f"hidden_{i}"] = torch.nn.Linear(input_size if i == 0 else hidden_size, hidden_size)
            _modules[f"{activation.__name__.lower()}_{i}"] = activation()
        _modules["readout"] = torch.nn.Linear(hidden_size, n_targets)
        self.fnn = torch.nn.Sequential(_modules)
        self.lr = lr
        self.save_hyperparameters()

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), self.lr)

    def forward(self, x: torch.Tensor):
        return self.fnn(x)

    def _step(self, batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor], name: str):
        x_1, x_2, y = batch
        x = torch.cat((x_1, x_2), dim=1)
        y_hat = self(x)
        loss = torch.nn.functional.mse_loss(y_hat, y)
        self.log(f"{name}/loss", loss, prog_bar=True)
        return loss

    def training_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx):
        return self._step(batch, "training")

    def validation_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx):
        return self._step(batch, "validation")

    def test_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx):
        return self._step(batch, "testing")

    def predict_step(self, batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor]):
        x_1, x_2, _ = batch
        x = torch.cat((x_1, x_2), dim=1)
        return self(x)

@model_registry.register("NepareModel")
class NepareModel(TorchModelBase):

    # Model parameters
    input_dim: int = 128
    hidden_dim: int = 64
    num_layers: int = 3
    activation = torch.nn.ReLU
    n_targets: int = 1

    # Training parameters
    lr: float = 1e-3
    loss_function: str = "mse"
