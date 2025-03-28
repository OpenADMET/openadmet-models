from sklearn.metrics import precision_recall_curve, auc
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

    def evaluate(self, y_true:list = None, y_pred:list = None, cutoffs:list = None, report:bool = False, output_dir:str = None):
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

        prs_df, baseline = self.get_precision_recall(y_pred, y_true, cutoffs)
        self.plot_precision_recall_curve(prs_df, baseline, output_dir)
        self.plot_aupr(prs_df["AUPR"], cutoffs, output_dir)

        self.report(report, output_dir, prs_df)

    def get_precision_recall(self, y_pred:list, y_true:list, cutoffs:list):
        """
        Calculate precision and recall metrics for given cutoffs.

        Parameters:
        y_pred (array-like): Predicted values.
        y_true (array-like): True values.
        cutoffs (list): List of cutoff values to calculate precision and recall.

        Returns:
        tuple: A tuple containing:
            - prs_df (pd.DataFrame): DataFrame with precision, recall, cutoff, and AUPR values.
            - baseline (float): Baseline value for the precision-recall curve.
        """

        prs_df = {'Precision':[], 'Recall':[], 'Cutoff':[], 'AUPR':[]}
        for c in cutoffs:
            pred_class = [y > c for y in y_pred]
            true_class = [y > c for y in y_true]
            precision, recall, _ = precision_recall_curve(true_class, pred_class)
            prs_df['Precision'].append(precision)
            prs_df['Recall'].append(recall)
            prs_df['Cutoff'].append(c)
            prs_df['AUPR'].append(auc(precision, recall))

        prs_df = pd.DataFrame(prs_df)
        baseline = np.sum(true_class)/len(true_class)

        return(prs_df, baseline)

    def plot_precision_recall_curve(self, prs_df, baseline, output_dir):
        for index, row in prs_df.iterrows():
            recall = row["Recall"]
            precision = row["Precision"]
            plt.step(recall, precision, alpha=0.5, where='post')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.plot((0,1), (baseline, baseline), 'r--', alpha=0.3, label='baseline')
        if output_dir is not None:
            plt.savefig(f"{output_dir}/pr_curve.pdf")

    def plot_aupr(self, auprs, cutoffs, output_dir):
        plt.plot(cutoffs, auprs)
        plt.xlabel("Binary Cutoff")
        plt.ylabel("AUPR")
        plt.title("Area under the PR curve vs binary cutoff")
        if output_dir is not None:
            plt.savefig(f"{output_dir}/aupr.pdf")

    def stats_to_json(self, data_df, output_dir):
        """
        Convert the precision-recall dataframe to json
        """
        data_df.to_json(f"{output_dir}/posthoc_binary_eval.json")

    def report(self, write=False, output_dir=None, stats_dfs=None):
        """
        Report the evaluation
        """
        if write and stats_dfs is not None:
            self.stats_to_json(stats_dfs, output_dir)
