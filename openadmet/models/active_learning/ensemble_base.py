from typing import ClassVar

from class_registry import ClassRegistry, RegistryKeyError

from openadmet.models.architecture.model_base import ModelBase

ensemblers = ClassRegistry(unique=True)


def get_ensemble_class(ensemble_type):
    try:
        feat_class = ensemblers.get_class(ensemble_type)
    except RegistryKeyError:
        raise ValueError(
            f"Ensemble type {ensemble_type} not found in ensemble catalogue,"
            f"available ensembles are {list(ensemblers.classes())}"
        )
    return feat_class


class EnsembleBase(ModelBase):
    """
    Base class for ensemble models.
    """

    type: ClassVar[str] = "EnsembleBase"
    models: list = []
    n_models: int = 0

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
