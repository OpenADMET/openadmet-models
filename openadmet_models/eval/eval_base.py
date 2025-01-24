from pydantic import BaseModel
from abc import abstractmethod
from class_registry import ClassRegistry
from class_registry import RegistryKeyError


EVAL_CLASSES = ClassRegistry()


def get_eval_class(eval_type):
    try:
        eval_class = EVAL_CLASSES.get_class(eval_type)
    except RegistryKeyError:
        raise ValueError(f"Eval type {eval_type} not found in eval catalouge")
    
    return eval_class
    


class EvalBase(BaseModel):
    

    @abstractmethod
    def evaluate(self, y_true, y_pred):
        """
        Evaluate the model
        """
        pass






    