from typing import ClassVar

from class_registry import ClassRegistry, RegistryKeyError
from pydantic import field_validator

from openadmet.models.architecture.model_base import ModelBase

ensemblers = ClassRegistry(unique=True)


def get_ensemble_class(ensemble_type):
    try:
        ensemble_class = ensemblers.get_class(ensemble_type)
    except RegistryKeyError:
        raise ValueError(
            f"Ensemble type {ensemble_type} not found in ensemble catalogue,"
            f"available ensembles are {list(ensemblers.classes())}"
        )
    return ensemble_class


class EnsembleBase(ModelBase):
    """
    Base class for ensemble models.
    """

    type: ClassVar[str] = "EnsembleBase"
    models: list = []
    n_models: int = 1

    @field_validator("n_models")
    @classmethod
    def validate_n_models(cls, value):
        if value < 1:
            raise ValueError("Number of models must be greater than zero.")
        return value

    def build(self):
        """
        Not needed, as the committee will be built from provided models.

        """

        pass

    def from_params(self):
        """
        This method doesn't really make sense for this class, as it is instantiated from already-trained models
        or from the `train` method.
        """
        raise NotImplementedError
