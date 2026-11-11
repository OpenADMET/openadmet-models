"""Posthoc binary metrics evaluation."""

import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from class_registry import RegistryKeyError
from pydantic import Field
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    confusion_matrix,
    precision_score,
    recall_score,
)

from openadmet.models.eval.eval_base import EvalBase, evaluators, get_t_true_and_t_pred


@evaluators.register("PosthocBinaryMetrics")
class PosthocBinaryMetrics(EvalBase):
    """
    Posthoc binary metrics.

    Intended to be used for regression-based models to calculate
    precision and recall metrics for user-input cutoffs.

    Not intended for binary models.

    Attributes
    ----------
    _evaluated : bool
        Whether the model has been evaluated.
    data : dict
        Dictionary of computed metrics.

    """

    _evaluated: bool = False
    data: dict = {}

    def evaluate(
        self,
        y_true: list = None,
        y_pred: list = None,
        cutoff: float = None,
        target_labels: list = None,
        **kwargs,
    ):
        """
        Evaluate the precision and recall metrics for the model with user-input cutoffs.

        Parameters
        ----------
        y_true : array-like
            True values or labels.
        y_pred : array-like
            Predicted values or labels.
        cutoff : float, optional
            Cutoff value to calculate precision and recall.
        target_labels : list of str, optional
            List of target names.
        kwargs : Dict
            Additional keyword arguments.

        Returns
        -------
        dict
            Dictionary of computed metrics.

        Raises
        ------
        ValueError
            If `y_true`, `y_pred`, or `cutoff` is not provided.

        """
        if y_true is None or y_pred is None:
            raise ValueError("Must provide y_true and y_pred")
        if cutoff is None:
            raise ValueError("Must provide cutoff")

        if isinstance(y_true, (pd.Series, pd.DataFrame)):
            y_true = y_true.to_numpy()

        # Ensure y_pred and y_true are 2D arrays for consistency
        if isinstance(y_pred, list):
            y_pred = np.array(y_pred)
        if isinstance(y_true, list):
            y_true = np.array(y_true)

        if y_pred.ndim == 1:
            y_pred = y_pred.reshape(-1, 1)
        if y_true.ndim == 1:
            y_true = y_true.reshape(-1, 1)

        n_tasks = y_true.shape[1]
        if not (n_tasks == y_pred.shape[1]):
            raise ValueError("y_true and y_pred must have the same number of tasks")
        if target_labels is None:
            target_labels = [f"task_{i}" for i in range(n_tasks)]

        self.data = {"cutoff": cutoff}

        for task_id in range(n_tasks):
            t_true, t_pred = get_t_true_and_t_pred(task_id, y_true, y_pred, None, None)
            t_label = target_labels[task_id]

            precision, recall = self.get_precision_recall(t_pred, t_true, cutoff)

            self.data[t_label] = {
                "precision": {"value": precision},
                "recall": {"value": recall},
            }

        self._evaluated = True
        return self.data

    def get_precision_recall(self, y_pred: list, y_true: list, cutoff: float):
        """
        Calculate precision and recall metrics for a given cutoff.

        Parameters
        ----------
        y_pred : array-like
            Predicted values.
        y_true : array-like
            True values.
        cutoff : float
            Cutoff value to calculate precision and recall.

        Returns
        -------
        tuple
            A tuple containing:
            - precision : float
                Precision value.
            - recall : float
                Recall value.

        """
        pred_class = [y > cutoff for y in y_pred]
        true_class = [y > cutoff for y in y_true]
        precision = precision_score(true_class, pred_class, zero_division=0)
        recall = recall_score(true_class, pred_class, zero_division=0)

        return (precision, recall)

    def stats_to_json(self, data_dict, output_dir):
        """
        Save the precision-recall metrics to a JSON file.

        Parameters
        ----------
        data_dict : dict
            Dictionary containing precision and recall metrics.
        output_dir : str
            Directory to save the JSON file.

        """
        with open(f"{output_dir}/posthoc_binary_eval.json", "w") as f:
            json.dump(data_dict, f, indent=2)

    def report(self, write=False, output_dir=None):
        """
        Report the evaluation results, optionally saving them to JSON.

        Parameters
        ----------
        write : bool, optional
            Whether to write the results to a JSON file. Default is False.
        output_dir : str, optional
            Directory to save the JSON file if write is True.

        Returns
        -------
        dict
            Dictionary of computed metrics.

        """
        if write and self.data:
            self.stats_to_json(self.data, output_dir)
        return self.data


@evaluators.register("PosthocBinaryPlots")
class PosthocBinaryPlots(EvalBase):
    """
    Generate and save posthoc binary plots such as confusion matrices and classification scatter plots.

    Attributes
    ----------
    plots : dict
        Dictionary of plot functions.
    dpi : int
        DPI for the plot.

    """

    plots: dict = {}
    dpi: int = Field(300, description="DPI for the plot")

    def evaluate(
        self,
        y_true: list = None,
        y_pred: list = None,
        cutoff: float = None,
        target_labels: list = None,
        **kwargs,
    ):
        """
        Generate posthoc binary plots.

        Parameters
        ----------
        y_true : array-like
            True values or labels.
        y_pred : array-like
            Predicted values or labels.
        cutoff : float, optional
            Cutoff value to binarize predictions and true values.
        target_labels : list of str, optional
            List of target names.
        kwargs : Dict
            Additional keyword arguments.

        Returns
        -------
        dict
            Dictionary of plot figures.

        Raises
        ------
        ValueError
            If `y_true`, `y_pred`, or `cutoff` is not provided.

        """
        if y_true is None or y_pred is None:
            raise ValueError("Must provide y_true and y_pred")
        if cutoff is None:
            raise ValueError("Must provide cutoff")

        if isinstance(y_true, (pd.Series, pd.DataFrame)):
            y_true = y_true.to_numpy()

        # Ensure y_pred and y_true are 2D arrays for consistency
        if isinstance(y_pred, list):
            y_pred = np.array(y_pred)
        if isinstance(y_true, list):
            y_true = np.array(y_true)

        if y_pred.ndim == 1:
            y_pred = y_pred.reshape(-1, 1)
        if y_true.ndim == 1:
            y_true = y_true.reshape(-1, 1)

        n_tasks = y_true.shape[1]
        if not (n_tasks == y_pred.shape[1]):
            raise ValueError("y_true and y_pred must have the same number of tasks")
        if target_labels is None:
            target_labels = [f"task_{i}" for i in range(n_tasks)]

        self.plots = {
            "confusion_matrix": self.plot_confusion_matrix,
            "classification_scatter": self.plot_posthoc_classification,
        }

        self.plot_data = {}

        for task_id in range(n_tasks):
            t_true, t_pred = get_t_true_and_t_pred(task_id, y_true, y_pred, None, None)
            t_label = target_labels[task_id]

            for plot_tag, plot_func in self.plots.items():
                self.plot_data[f"{t_label}_{plot_tag}"] = plot_func(
                    t_true, t_pred, cutoff
                )

        return self.plot_data

    @staticmethod
    def plot_confusion_matrix(y_true: list, y_pred: list, cutoff: float):
        """
        Plot the confusion matrix for a given cutoff.

        Parameters
        ----------
        y_true : list or array-like
            True values or labels.
        y_pred : list or array-like
            Predicted values or labels.
        cutoff : float
            Cutoff value to binarize predictions and true values.

        Returns
        -------
        matplotlib.figure.Figure
            The confusion matrix plot figure.

        """
        pred_class = [y > cutoff for y in y_pred]
        true_class = [y > cutoff for y in y_true]
        cm = confusion_matrix(true_class, pred_class)
        disp = ConfusionMatrixDisplay(cm)
        # Plotting to a new figure to avoid modifying global state or overlapping
        fig, ax = plt.subplots()
        disp.plot(ax=ax)
        return fig

    @staticmethod
    def plot_posthoc_classification(y_true: list, y_pred: list, cutoff: float):
        """
        Plot the post-hoc classification scatter plot with cutoff lines.

        Parameters
        ----------
        y_true : list or array-like
            True values or labels.
        y_pred : list or array-like
            Predicted values or labels.
        cutoff : float
            Cutoff value to draw threshold lines.

        Returns
        -------
        matplotlib.figure.Figure
            The classification scatter plot figure.

        """
        fig, ax = plt.subplots()
        ax.scatter(y_true, y_pred)
        ax.axvline(cutoff, color="r", linestyle="--")
        ax.axhline(cutoff, color="r", linestyle="--")
        ax.set_xlabel("True Value")
        ax.set_ylabel("Predicted Value")
        ax.set_title(f"Post-hoc classification with cutoff: {cutoff} ")
        return fig

    def report(self, write=False, output_dir=None):
        """
        Report the generated plots, optionally writing to disk.

        Parameters
        ----------
        write : bool, optional
            Whether to write the plots to disk.
        output_dir : str, optional
            Output directory for the plots.

        Returns
        -------
        dict
            Dictionary of plot figures.

        """
        if write:
            self.write_report(output_dir)
        return self.plot_data

    def write_report(self, output_dir):
        """
        Write the generated plots to PNG files.

        Parameters
        ----------
        output_dir : str
            Output directory for the plots.

        """
        for plot_tag, plot in self.plot_data.items():
            plot_path = output_dir / f"{plot_tag}.png"
            plot.savefig(plot_path, dpi=self.dpi)
