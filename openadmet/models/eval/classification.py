import json
from typing import Callable

import numpy as np
import wandb
from pydantic import Field
from scipy.stats import bootstrap
from sklearn.metrics import (
    accuracy_score,
    auc,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from openadmet.models.eval.eval_base import EvalBase, evaluators


def pr_auc_score(y_true, y_pred):
    """
    Calculate the area under the precision-recall curve
    """
    precision, recall, _ = precision_recall_curve(y_true, y_pred)
    return auc(recall, precision)


@evaluators.register("ClassificationMetrics")
class ClassificationMetrics(EvalBase):
    bootstrap_confidence_level: float = Field(
        0.95, description="Confidence level for the bootstrap"
    )
    use_wandb: bool = Field(False, description="Whether to use wandb")
    _evaluated: bool = False

    # tuple of metric, whether it is a scipy statistic, and the name to use in the report
    _metrics: dict = {
        "accuracy": (accuracy_score, False, "Accuracy"),
        "precision": (precision_score, False, "Precision"),
        "recall": (recall_score, False, "Recall"),
        "f1": (f1_score, False, "F1 Score"),
        "roc_auc": (roc_auc_score, False, "ROC AUC"),
        "pr_auc": (pr_auc_score, False, "PR AUC"),
    }

    def evaluate(self, y_true=None, y_pred=None, use_wandb=False, tag=None, **kwargs):
        """
        Evaluate the classification model
        """
        if y_true is None or y_pred is None:
            raise ValueError("Must provide y_true and y_pred")

        self.data = {"tag": tag}

        if use_wandb:
            self.use_wandb = use_wandb

        for metric_tag, (metric, is_scipy, _) in self._metrics.items():
            value, lower_ci, upper_ci = self.stat_and_bootstrap(
                metric_tag,
                y_pred,
                y_true,
                metric,
                is_scipy_statistic=is_scipy,
                confidence_level=self.bootstrap_confidence_level,
            )

            metric_data = {}
            metric_data["value"] = value
            metric_data["lower_ci"] = lower_ci
            metric_data["upper_ci"] = upper_ci
            metric_data["confidence_level"] = self.bootstrap_confidence_level

            self.data[f"{metric_tag}"] = metric_data

        if self.use_wandb:
            # make a table for the metrics
            table = wandb.Table(
                columns=["Metric", "Value", "Lower CI", "Upper CI", "Confidence Level"]
            )
            for metric in self.metric_names:
                table.add_data(
                    metric,
                    self.data[metric]["value"],
                    self.data[metric]["lower_ci"],
                    self.data[metric]["upper_ci"],
                    self.data[metric]["confidence_level"],
                )
            wandb.log({"metrics": table})

            for metric in self.metric_names:
                wandb.log({metric: self.data[metric]["value"]})

        self._evaluated = True
        return self.data

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
        Calculate the metric and confidence intervals
        """
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

    @property
    def metric_names(self):
        """
        Return the metric names
        """
        return list(self._metrics.keys())

    def report(self, write=False, output_dir=None):
        """
        Report the evaluation
        """
        if write:
            self.write_report(output_dir)
        return self.data

    def write_report(self, output_dir):
        """
        Write the evaluation report
        """
        # write to JSON
        json_path = output_dir / "classification_metrics.json"
        with open(json_path, "w") as f:
            json.dump(self.data, f, indent=2)

        # also log the json to wandb
        if self.use_wandb:
            artifact = wandb.Artifact(name="metrics_json", type="metric_json")
            # Add a file to the artifact
            artifact.add_file(json_path)
            # Log the artifact
            wandb.log_artifact(artifact)

    def make_stat_caption(self):
        """
        Make a caption for the statistics
        """
        if not self._evaluated:
            raise ValueError("Must evaluate before making a caption")
        stat_caption = ""
        for metric in self.metric_names:
            value = self.data[metric]["value"]
            lower_ci = self.data[metric]["lower_ci"]
            upper_ci = self.data[metric]["upper_ci"]
            confidence_level = self.data[metric]["confidence_level"]
            stat_caption += f"{self._metrics[metric][2]}: {value:.2f}$_{{{lower_ci:.2f}}}^{{{upper_ci:.2f}}}$\n"
        stat_caption += f"Confidence level: {confidence_level}"
        return stat_caption


@evaluators.register("ClassificationPlots")
class ClassificationPlots(EvalBase):
    axes_labels: list[str] = Field(
        ["Measured", "Predicted"], description="Labels for the axes"
    )
    title: str = Field("Pred vs ", description="Title for the plot")
    plots: dict = {}
    use_wandb: bool = Field(False, description="Whether to use wandb")
    dpi: int = Field(300, description="DPI for the plot")

    def evaluate(self, y_true=None, y_pred=None, use_wandb=False, **kwargs):
        """
        Evaluate the classification model
        """
        if use_wandb:
            self.use_wandb = use_wandb

        if y_true is None or y_pred is None:
            raise ValueError("Must provide y_true and y_pred")

        self.plots = {
            "auroc": self.auroc,
            "aupr": self.aupr,
        }

        self.plot_data = {}

        # if self.do_stats:
        #     rm = ClassificationMetrics()
        #     rm.evaluate(y_true, y_pred)
        #     stat_caption = rm.make_stat_caption()

        # create the plots
        for plot_tag, plot in self.plots.items():
            self.plot_data[plot_tag] = plot(
                y_true,
                y_pred,
                xlabel=self.axes_labels[0],
                ylabel=self.axes_labels[1],
                title=self.title,
                stat_caption=stat_caption,
                pXC50=self.pXC50,
                min_val=self.min_val,
                max_val=self.max_val,
            )

    @staticmethod
    def auroc(
        y_true,
        y_pred,
        xlabel="Measured",
        ylabel="Predicted",
        title="",
        stat_caption="",
    ):
        fpr, tpr, _ = roc_curve(y_true, y_pred)
        pass

    @staticmethod
    def aupr(
        y_true,
        y_pred,
        xlabel="Measured",
        ylabel="Predicted",
        title="",
        stat_caption="",
    ):
        precision, recall, _ = precision_recall_curve(y_true, y_pred)
        pass

    def report(self, write=False, output_dir=None):
        """
        Report the evaluation
        """
        if write:
            self.write_report(output_dir)
        return self.plot_data

    def write_report(self, output_dir):
        """
        Write the evaluation report
        """

        for plot_tag, plot in self.plot_data.items():
            plot_path = output_dir / f"{plot_tag}.png"
            plot.savefig(plot_path, dpi=self.dpi)
            if self.use_wandb:
                wandb.log({plot_tag: wandb.Image(str(plot_path))})
