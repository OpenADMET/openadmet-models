from sklearn.utils.discovery import all_estimators

from openadmet_models.models.model_base import PickleableModelBase, models


def get_sklearn_estimators_as_dict(type_filter: str = None):
    """
    Get the sklearn estimators

    Parameters
    ----------
    type_filter: str
        Filter for the type of estimator to get, one of “classifier”, “regressor”, “cluster”, “transformer”
    """
    from sklearn.utils import all_estimators

    estimators = all_estimators(type_filter=type_filter)
    estimator_dict = {name: est for name, est in estimators}
    return estimator_dict
