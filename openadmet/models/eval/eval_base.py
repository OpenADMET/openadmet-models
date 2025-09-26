"""Base class and utilities for evaluation modules."""

from abc import abstractmethod
from typing import Callable

import numpy as np
from class_registry import ClassRegistry, RegistryKeyError
from pydantic import BaseModel
from scipy.stats import bootstrap

evaluators = ClassRegistry(unique=True)


def get_eval_class(eval_type):
    """
    Retrieve an evaluation class from the registry by type.

    Parameters
    ----------
    eval_type : str
        The evaluation type string.

    Returns
    -------
    type
        The evaluation class corresponding to the given type.

    Raises
    ------
    ValueError
        If the evaluation type is not found in the registry.

    """
    try:
        eval_class = evaluators.get_class(eval_type)
    except RegistryKeyError:
        raise ValueError(f"Eval type {eval_type} not found in eval catalouge")

    return eval_class


def mask_nans(y_true: np.ndarray, y_pred: np.ndarray):
    """
    Remove any pairs where either y_true or y_pred is NaN.

    Parameters
    ----------
    y_true : np.ndarray
        Array of true values.
    y_pred : np.ndarray
        Array of predicted values.

    Returns
    -------
    tuple of np.ndarray
        Filtered arrays (y_true, y_pred) with NaNs removed.

    """
    mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    return y_true[mask], y_pred[mask]


def mask_nans_std(y_true: np.ndarray, y_pred: np.ndarray, y_std: np.ndarray):
    """
    Remove any pairs where either y_true or y_pred is NaN.

    Parameters
    ----------
    y_true : np.ndarray
        Array of true values.
    y_pred : np.ndarray
        Array of predicted values.
    y_std : np.ndarray
        Array of standard deviations.

    Returns
    -------
    tuple of np.ndarray
        Filtered arrays (y_true, y_pred, y_std) with NaNs removed.

    """
    mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    return y_true[mask], y_pred[mask], y_std[mask]


class EvalBase(BaseModel):
    """Abstract base class for evaluation modules."""

    class Config:
        """Pydantic configuration for the EvalBase class."""

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
        Evaluate the model.

        Parameters
        ----------
        y_true : array-like, optional
            True values.
        y_pred : array-like, optional
            Predicted values.
        model : object, optional
            Model instance.
        X_train : array-like, optional
            Training features.
        y_train : array-like, optional
            Training targets.
        wandb_logger : object, optional
            Weights & Biases logger.

        Returns
        -------
        Any
            Evaluation results.

        """
        pass

    @abstractmethod
    def report(self):
        """
        Report the evaluation results.

        Returns
        -------
        Any
            Report output.

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
        """
        Calculate a metric and its bootstrap confidence interval.

        Parameters
        ----------
        metric_tag : str
            Name of the metric.
        y_pred : np.ndarray
            Predicted values.
        y_true : np.ndarray
            True values.
        statistic : Callable
            Function to compute the metric.
        confidence_level : float, optional
            Confidence level for the interval (default is 0.95).
        is_scipy_statistic : bool, optional
            Whether the statistic is a scipy.stats object (default is False).

        Returns
        -------
        tuple
            Tuple of (metric, lower confidence bound, upper confidence bound).

        """
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
