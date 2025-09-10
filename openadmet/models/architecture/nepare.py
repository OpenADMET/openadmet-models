import numpy as np
import torch
from lightning import pytorch as pl
from pydantic import field_validator
from loguru import logger
from chemprop import models, nn
from typing import OrderedDict

from openadmet.models.architecture.model_base import models as model_registry
from openadmet.models.architecture.model_base import LightningModuleBase, LightningModelBase


from typing import ClassVar


_METRIC_TO_LOSS = {
    "mse": nn.metrics.MSE(),
    "mae": nn.metrics.MAE(),
    "rmse": nn.metrics.RMSE(),
}

class NeuralPairwiseRegressorModule(LightningModuleBase):
    def __init__(self, 
                 input_size, 
                 hidden_size, 
                 num_layers, 
                 activation = torch.nn.ReLU, 
                 lr: float = 1e-4, 
                 n_targets: int = 1,
                 monitor_metric: str = "val_loss"):
        super().__init__()
        input_size = input_size * 2
        _modules = OrderedDict()
        for i in range(num_layers):
            _modules[f"hidden_{i}"] = torch.nn.Linear(input_size if i == 0 else hidden_size, hidden_size)
            _modules[f"{activation.__name__.lower()}_{i}"] = activation()
        _modules["readout"] = torch.nn.Linear(hidden_size, n_targets)
        self.fnn = torch.nn.Sequential(_modules)
        self.lr = lr
        self.save_hyperparameters()

    def forward(self, x: torch.Tensor):
        return self.fnn(x)

    def training_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx):
        return self._step(batch, self.monitor_metric)

    def validation_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx):
        return self._step(batch, self.monitor_metric)

    def test_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx):
        return self._step(batch, self.monitor_metric)
    
    def _step(self, batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor], name: str):
        x_1, x_2, y = batch
        x = torch.cat((x_1, x_2), dim=1)
        y_hat = self(x)
        loss = torch.nn.functional.mse_loss(y_hat, y)
        self.log(f"{name}", loss, prog_bar=True)
        return loss

    def predict_step(self, batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor]):
        x_1, x_2, _ = batch
        x = torch.cat((x_1, x_2), dim=1)
        return self(x)

@model_registry.register("NeuralPairwiseRegressorModel")
class NeuralPairwiseRegressorModel(LightningModelBase):
    """
    NepareChemPropModel is a neural pairwise regression model based on ChemProp.
    It uses learned embeddings for pairwise features.
    """

    type: ClassVar[str] = "NeuralPairwiseRegressorModel"
    mod_params: dict = {}

    def train(self, dataloader):
        """
        Train the model
        """
        raise NotImplementedError(
            "Training not implemented in model class, use a trainer."
        )
    
    @classmethod
    def from_params(cls, class_params: dict = {}, mod_params: dict = {}):
        """
        Create a model from parameters
        """

        instance = cls(**class_params, mod_params=mod_params)
        instance.build()
        return instance

    def build(self, scaler=None):
        if not self.estimator:
            nepare = NeuralPairwiseRegressorModule(**self.mod_params)
            self.estimator = nepare
        else:
            logger.warning("Model already exists, skipping build")

        return self

    def make_new(self) -> "NeuralPairwiseRegressorModel":
        """
        Copy parameters to a new model instance without copying the estimator
        """
        return self.__class__(**self.mod_params, **self.dict(exclude={"estimator"}))

    def predict(self, dataloader, accelerator="gpu", devices=1) -> torch.Tensor:

        if not self.estimator:
            raise AttributeError("Model not built or trained.")
        
        with torch.inference_mode():
            trainer = pl.Trainer(
                logger=None,
                enable_progress_bar=False,
                accelerator=accelerator,
                devices=devices,
            )
            preds = trainer.predict(self.estimator, dataloader)
        return torch.cat(preds, dim=0).numpy()
