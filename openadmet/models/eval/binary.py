from sklearn.metrics import precision_score, recall_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np
import wandb
import pandas as pd

from openadmet.models.eval.eval_base import EvalBase, evaluators


@evaluators.register("PosthocBinaryMetrics")
class PosthocBinaryMetrics(EvalBase):

    """
    Intended to be used for regression-based models to calculate
    precision and recall metrics for user-input cutoffs

    Not intended for binary models
    """

    def evaluate(self, y_true:list = None, y_pred:list = None, cutoff:float = None, report:bool = False, output_dir:str = None):
        """
        Evaluate the precision and recall metrics for model with user-input cutoffs.

        Parameters:
        y_true (array-like, optional): True values.
        y_pred (array-like, optional): Predicted values.
        cutoffs (list, optional): List of cutoff values to calculate precision and recall.
        report (bool, optional): Whether to save jsons of resulting precision/recall metrics. Default is False.
        output_dir (str, optional): Directory to save the output plots and report. Default is None.

        Returns:
        None
        """

        if y_true is None or y_pred is None:
            raise ValueError("Must provide y_true and y_pred")
        self.plot_confusion_matrix(y_true, y_pred, cutoff, output_dir)
        self.plot_posthoc_classification(y_true, y_pred, cutoff, output_dir)
        precision, recall = self.get_precision_recall(y_pred, y_true, cutoff)
        self.report(report, output_dir, precision=precision, recall=recall)

    def get_precision_recall(self, y_pred:list, y_true:list, cutoff:float):
        """
        Calculate precision and recall metrics for given cutoffs.

        Parameters:
        y_pred (array-like): Predicted values.
        y_true (array-like): True values.
        cutoff (float): Cutoff to calculate precision and recall.

        Returns:
        Tuple: A tuple containing:
            - precision (float): Precision value.
            - recall (float): Recall value.
        """
        pred_class = [y > cutoff for y in y_pred]
        true_class = [y > cutoff for y in y_true]
        precision = precision_score(true_class, pred_class)
        recall = recall_score(true_class, pred_class)

        return(precision, recall)

    def plot_confusion_matrix(self, y_true:list, y_pred:list, cutoff:float, output_dir:str=None):
        """
        Plot the confusion matrix for a given cutoff
        """
        pred_class = [y > cutoff for y in y_pred]
        true_class = [y > cutoff for y in y_true]
        cm = confusion_matrix(true_class, pred_class)
        disp = ConfusionMatrixDisplay(cm)
        disp.plot()
        if output_dir is not None:
            plt.savefig(f"{output_dir}/confusion_matrix.pdf")

    def plot_posthoc_classification(self, y_true:list, y_pred:list, cutoff:float, output_dir:str=None):
        """
        Plot the classification of the model with a given cutoff
        """
        fig, ax = plt.subplots()
        plt.scatter(y_true, y_pred)
        plt.axvline(cutoff, color='r', linestyle='--')
        plt.axhline(cutoff, color='r', linestyle='--')
        plt.xlabel("True Value")
        plt.ylabel("Predicted Value")
        plt.title("Classification of the model")
        if output_dir is not None:
            plt.savefig(f"{output_dir}/classification.pdf")

    def stats_to_json(self, data_df, output_dir):
        """
        Convert the precision-recall dataframe to json
        """
        data_df.to_json(f"{output_dir}/posthoc_binary_eval.json")

    def report(self, write=False, output_dir=None, precision=None, recall=None):
        """
        Report the evaluation
        """
        stats_df = pd.DataFrame({"precision": precision, "recall": recall}, index=[0])
        if write and stats_df is not None:
            self.stats_to_json(stats_df, output_dir)
