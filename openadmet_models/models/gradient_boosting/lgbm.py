import lightgbm as lgb
from pydantic import BaseModel
from openadmet_models.models.model_base import ModelBase

class LGBMRegressorModel(ModelBase):
    """
    LightGBM regression model
    """

    

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict using the model
        """
        return self.model.predict(X)
    
