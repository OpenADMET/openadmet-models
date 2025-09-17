"""Base class for ensemble models."""

from typing import ClassVar

from class_registry import ClassRegistry, RegistryKeyError

from openadmet.models.architecture.model_base import ModelBase

ensemblers = ClassRegistry(unique=True)


def get_ensemble_class(ensemble_type):
    """Get the ensemble class."""
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

    Attributes
    ----------
    type : ClassVar[str]
        The type of the ensemble model.
    models : list
        The list of models in the ensemble.
    _calibration_model_save_name : ClassVar[str]
        The name of the calibration model save file.

    """

    type: ClassVar[str] = "EnsembleBase"
    models: list = []
    _calibration_model_save_name: ClassVar[str] = "calibration_model.pkl"

    @property
    def n_models(self):
        """Get the number of models in the ensemble."""
        return len(self.models)

    def build(self):
        """Is here as placeholder, as the committee will be built from provided models."""
        pass

    def from_params(self):
        """
        Is here as placeholder.

        This method doesn't really make sense for this class, as it is instantiated from already-trained models
        or from the `train` method.
        """
        raise NotImplementedError
