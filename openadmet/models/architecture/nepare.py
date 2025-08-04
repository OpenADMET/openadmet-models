import numpy as np
import torch
from lightning import pytorch as pl
from pydantic import field_validator
from loguru import logger

from openadmet.models.architecture.chemprop import ChemPropModel
from openadmet.models.architecture.model_base import models as model_registry

from nepare.nn import LearnedEmbeddingNeuralPairwiseRegressor

from typing import ClassVar, Optional, Any
from collections import OrderedDict

@model_registry.register("NepareChemPropModel")
class NepareChempropModelBase(ChemPropModel):
    """
    NepareChemPropModel is a neural pairwise regression model based on ChemProp.
    It uses learned embeddings for pairwise features.
    """
    
    type: ClassVar[str] = "NepareChemPropModel"
    
    def _step(self, batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor], name: str):
        x_1, x_2, y = batch
        x = torch.cat((x_1, x_2), dim=1)
        return super()._step((x, y), name)

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