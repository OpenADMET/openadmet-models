from abc import abstractmethod

from class_registry import ClassRegistry, RegistryKeyError
from pydantic import BaseModel

evaluators = ClassRegistry(unique=True)


def get_eval_class(eval_type):
    try:
        eval_class = evaluators.get_class(eval_type)
    except RegistryKeyError:
        raise ValueError(f"Eval type {eval_type} not found in eval catalouge")

    return eval_class


class EvalBase(BaseModel):

    class Config:
        extra = "allow"

    @abstractmethod
    def evaluate(self, y_true=None, y_pred=None, model=None, X_train=None, y_train=None):
        """
        Evaluate the model
        """
        pass

    @abstractmethod
    def report(self):
        """
        Report the evaluation
        """
        pass


