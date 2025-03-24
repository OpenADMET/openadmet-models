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
    precision and recall metrics for user-input
    """

    def evaluate(self, y_true=None, y_pred=None, cutoffs=None, report=False, output_dir=None, **kwargs):

        if y_true is None or y_pred is None:
            raise ValueError("Must provide y_true and y_pred")

        prs_df, baseline = self.get_precision_recall(y_pred, y_true, cutoffs)
        self.plot_precision_recall_curve(prs_df, baseline, output_dir)
        self.plot_aupr(prs_df["AUPR"], cutoffs, output_dir)

        self.report(report, output_dir, prs_df)

    def get_precision_recall(self, y_pred, y_true, cutoffs):
        prs_df = {'Precision':[], 'Recall':[], 'Cutoff':[], 'AUPR':[]}
        for c in cutoffs:
            pred_class = [y > c for y in y_pred]
            true_class = [y > c for y in y_true]
            precision, recall, _ = precision_recall_curve(true_class, pred_class)
            prs_df['Precision'].append(precision)
            prs_df['Recall'].append(recall)
            prs_df['Cutoff'].append(c)
            prs_df['AUPR'].append(auc(precision, recall))

        return(pd.DataFrame(prs_df), np.sum(true_class)/len(true_class))

    def plot_precision_recall_curve(self, prs_df, baseline, output_dir):
        for cutoff in prs_df["Cutoff"]:
            recall = list(prs_df[prs_df["Cutoff"] == cutoff]["Recall"])
            precision = list(prs_df[prs_df["Cutoff"] == cutoff]["Precision"])
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
        data_df.to_json(f"{output_dir}/posthoc_binary_eval.json")

    def report(self, write=False, output_dir=None, stats_dfs=None):
        """
        Report the evaluation
        """
        if write and stats_dfs is not None:
            self.stats_to_json(stats_dfs, output_dir)
        