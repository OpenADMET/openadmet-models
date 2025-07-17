import numpy as np
import torch
from lightning import pytorch as pl

from openadmet.models.architecture.model_base import TorchModelBase
from openadmet.models.architecture.model_base import models as model_registry

@model_registry.register("NePaReModel")
class NePaReModel(TorchModelBase):
    pass