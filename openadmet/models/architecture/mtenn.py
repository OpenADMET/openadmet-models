from lightning import pytorch as pl
import torch
from typing import ClassVar
from loguru import logger
from mtenn.config import SchNetModelConfig
from openadmet.models.architecture.model_base import TorchModelBase
from openadmet.models.architecture.model_base import models as model_registry


class MTENNLightningWrapper(pl.LightningModule):
    def __init__(self,
                 model_config: SchNetModelConfig,
                 loss_fn = torch.nn.MSELoss(),
                 lr = 1e-4):
         super().__init__()
         self.model = model_config.build()
         self.loss_fn = loss_fn
         self.lr = lr

    def forward(self, batch):
        data_batch, _ = batch
        preds = []
        for data in data_batch:
            for k,v in data.items():
                data[k] = v.to(self.device)
            pred, _ = self.model(data)
            preds.append(pred)
        return torch.stack(preds)

    def training_step(self, batch, batch_idx):
        data_batch, target_batch = batch
        batch_loss = 0
        
        for data, target in zip(data_batch, target_batch):
            for k, v in data.items():
                data[k] = v.to(self.device)
            pred,_ = self.model(data)
            loss = self.loss_fn(pred, target.unsqueeze(0).to(self.device))
            batch_loss += loss

        avg_loss = batch_loss / len(data_batch)
        self.log('train_loss', avg_loss)
        return avg_loss

    def predict_step(self, batch, batch_idx):
        preds = []
        data_batch, _  = batch
        device = next(self.model.parameters()).device
        for data in data_batch:
            for k, v in data.items():
                data[k] = v.to(self.device)
            pred, _ = self.model(data)
            preds.append(pred)

        return torch.stack(preds)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.model.parameters(), lr=self.lr)



@model_registry.register("MTENNSchNetModel")
class MTENNSchNetModel(TorchModelBase):
    """
    """

    type:ClassVar[str] = "MTENNSchNetModel"
    #metric_list : list = ['mae']
    model_params: dict = {}

    #from params?
    # train here?

    def build(self, scaler=None): #what is scaler?
        """
        Prepare the model
        """
        if not self.estimator:
           #i think scaler is chemprop specific, but if not i will add loop back in
            model_config = SchNetModelConfig(**self.model_params)
            self.estimator = MTENNLightningWrapper(model_config)
        else:
            logger.warning("Model already exists, skipping build.")

    def from_params(self, params):
        pass

    def train(self, dataloader):
        """
        Train the model
        """
        raise NotImplementedError(
           "Training not implemented in model class, use a trainer."
        )

    def predict(self, dataloader) -> torch.Tensor:
        """
        """
        if not self.estimator:
            raise AttributeError("Model not built or trained.")

        with torch.inference_mode():
            acc = "cpu"
            #acc = "auto"	#uncomment line and commet above for gpu usage; if auto doesnt work change string to gpu
            trainer = pl.Trainer(
                logger=None, enable_progress_bar=False, accelerator=acc, devices=1    #changed cpu from auto this can be made back in future workarounds
            )
            preds = trainer.predict(self.estimator, dataloader)
        return torch.cat(preds).numpy().ravel()
