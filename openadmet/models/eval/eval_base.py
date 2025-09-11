from abc import abstractmethod
from typing import Callable

import numpy as np
from class_registry import ClassRegistry, RegistryKeyError
from pydantic import BaseModel
from scipy.stats import bootstrap

evaluators = ClassRegistry(unique=True)


def get_eval_class(eval_type):
    try:
        eval_class = evaluators.get_class(eval_type)
    except RegistryKeyError:
        raise ValueError(f"Eval type {eval_type} not found in eval catalouge")

    return eval_class


def mask_nans(y_true: np.ndarray, y_pred: np.ndarray):
    """
    Remove any pairs where either y_true or y_pred is NaN.
    """
    mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    return y_true[mask], y_pred[mask]


def mask_nans_std(y_true: np.ndarray, y_pred: np.ndarray, y_std: np.ndarray):
    """
    Remove any pairs where either y_true or y_pred is NaN.
    """
    mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    return y_true[mask], y_pred[mask], y_std[mask]


class EvalBase(BaseModel):
    class Config:
        extra = "allow"

    @abstractmethod
    def evaluate(
        self,
        y_true=None,
        y_pred=None,
        model=None,
        X_train=None,
        y_train=None,
        wandb_logger=None,
    ):
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

    def stat_and_bootstrap(
        self,
        metric_tag: str,
        y_pred: np.ndarray,
        y_true: np.ndarray,
        statistic: Callable,
        confidence_level: float = 0.95,
        is_scipy_statistic: bool = False,
    ):
        # calculate the metric and confidence intervals
        if is_scipy_statistic:
            metric = statistic(y_true, y_pred).statistic
            conf_interval = bootstrap(
                (y_true, y_pred),
                statistic=lambda y_true, y_pred: statistic(y_true, y_pred).statistic,
                method="basic",
                confidence_level=confidence_level,
                paired=True,
            ).confidence_interval

        else:
            metric = statistic(y_true, y_pred)
            conf_interval = bootstrap(
                (y_true, y_pred),
                statistic=statistic,
                method="basic",
                confidence_level=confidence_level,
                paired=True,
            ).confidence_interval

        return (
            metric,
            conf_interval.low,
            conf_interval.high,
        )
