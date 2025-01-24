from pydantic import BaseModel
from abc import ABC, abstractmethod

class EvalBase(BaseModel, ABC):

    @abstractmethod
    def evaluate(self, y_true, y_pred):
        """
        Evaluate the model
        """
        pass





    