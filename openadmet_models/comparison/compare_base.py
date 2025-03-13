from abc import ABC, abstractmethod

from class_registry import ClassRegistry, RegistryKeyError
from pydantic import BaseModel

comparisons = ClassRegistry(unique=True)


def get_comparison_class(compare_type):
    try:
        compare_class = comparisons.get_class(compare_type)
    except RegistryKeyError:
        raise ValueError(
            f"Comparison type {compare_type} not found in comparisons catalouge"
        )
    return compare_class


class ComparisonBase(BaseModel, ABC):

    @abstractmethod
    def compare(model_stats_fns: list[str], model_tags: list[str]):
        """
        Compare two model runs

        Parameters
        ----------
        model_stats_fns: list[str]
            List of paths to model stats (probably anvil_run/cross_validation_metrics.json)
        model_tags: list[str]
            List of names for user to identify models, must be in the
            same order as model_stats_fns

        """
        pass
