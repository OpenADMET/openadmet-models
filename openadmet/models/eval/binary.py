"""Posthoc binary metrics evaluation."""

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    confusion_matrix,
    precision_score,
    recall_score,
)

from openadmet.models.eval.eval_base import EvalBase, evaluators


@evaluators.register("PosthocBinaryMetrics")
class PosthocBinaryMetrics(EvalBase):
    """
    Posthoc binary metrics.

    Intended to be used for regression-based models to calculate
    precision and recall metrics for user-input cutoffs

    Not intended for binary models
    """

    def evaluate(
        self,
        y_true: list = None,
        y_pred: list = None,
        cutoff: float = None,
        report: bool = False,
        output_dir: str = None,
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
        report : bool, optional
            Whether to save JSON files of the resulting precision/recall metrics. Default is False.
        output_dir : str, optional
            Directory to save the output plots and report. Default is None.

        Raises
        ------
        ValueError
            If `y_true` or `y_pred` is not provided.

        Returns
        -------
        None

        """
        if y_true is None or y_pred is None:
            raise ValueError("Must provide y_true and y_pred")
        if cutoff is None:
            raise ValueError("Must provide cutoff")
        self.plot_confusion_matrix(y_true, y_pred, cutoff, output_dir)
        self.plot_posthoc_classification(y_true, y_pred, cutoff, output_dir)
        precision, recall = self.get_precision_recall(y_pred, y_true, cutoff)
        self.report(report, output_dir, precision=precision, recall=recall)

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
        precision = precision_score(true_class, pred_class)
        recall = recall_score(true_class, pred_class)

        return (precision, recall)

    def plot_confusion_matrix(
        self, y_true: list, y_pred: list, cutoff: float, output_dir: str = None
    ):
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
        output_dir : str, optional
            Directory to save the confusion matrix plot. If None, the plot is not saved.

        Returns
        -------
        None

        """
        pred_class = [y > cutoff for y in y_pred]
        true_class = [y > cutoff for y in y_true]
        cm = confusion_matrix(true_class, pred_class)
        disp = ConfusionMatrixDisplay(cm)
        disp.plot()
        if output_dir is not None:
            plt.savefig(f"{output_dir}/confusion_matrix.png", dpi=300)

    def plot_posthoc_classification(
        self, y_true: list, y_pred: list, cutoff: float, output_dir: str = None
    ):
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
        output_dir : str, optional
            Directory to save the classification plot. If None, the plot is not saved.

        Returns
        -------
        None

        """
        fig, ax = plt.subplots()
        plt.scatter(y_true, y_pred)
        plt.axvline(cutoff, color="r", linestyle="--")
        plt.axhline(cutoff, color="r", linestyle="--")
        plt.xlabel("True Value")
        plt.ylabel("Predicted Value")
        plt.title(f"Post-hoc classification with cutoff: {cutoff} ")
        if output_dir is not None:
            plt.savefig(f"{output_dir}/classification.png", dpi=300)

    def stats_to_json(self, data_df, output_dir):
        """
        Save the precision-recall DataFrame to a JSON file.

        Parameters
        ----------
        data_df : pandas.DataFrame
            DataFrame containing precision and recall metrics.
        output_dir : str
            Directory to save the JSON file.

        Returns
        -------
        None

        """
        data_df.to_json(f"{output_dir}/posthoc_binary_eval.json")

    def report(self, write=False, output_dir=None, precision=None, recall=None):
        """
        Report the evaluation results, optionally saving them to JSON.

        Parameters
        ----------
        write : bool, optional
            Whether to write the results to a JSON file. Default is False.
        output_dir : str, optional
            Directory to save the JSON file if write is True.
        precision : float or array-like, optional
            Precision value(s) to report.
        recall : float or array-like, optional
            Recall value(s) to report.

        Returns
        -------
        None

        """
        stats_df = pd.DataFrame({"precision": precision, "recall": recall}, index=[0])
        if write and stats_df is not None:
            self.stats_to_json(stats_df, output_dir)
