"""Classification metrics and plots for model evaluation."""

import json

import matplotlib.pyplot as plt
import numpy as np
import wandb
from pydantic import Field
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
    Calculate the area under the precision-recall curve.

    Parameters
    ----------
    y_true : array-like
        True binary labels or binary label indicators.
    y_pred : array-like
        Target scores, probability estimates of the positive class.

    Returns
    -------
    float
        Area under the precision-recall curve.

    """
    precision, recall, _ = precision_recall_curve(y_true, y_pred)
    return auc(recall, precision)


@evaluators.register("ClassificationMetrics")
class ClassificationMetrics(EvalBase):
    """
    Compute and report classification metrics such as accuracy, precision, recall, F1, ROC AUC, and PR AUC.

    Attributes
    ----------
    bootstrap_confidence_level : float
        Confidence level for the bootstrap.
    use_wandb : bool
        Whether to log metrics to Weights & Biases.
    _evaluated : bool
        Whether the evaluation has been performed.
    _metrics : dict
        Dictionary of metrics to compute, with metric functions and properties.

    """

    bootstrap_confidence_level: float = Field(
        0.95, description="Confidence level for the bootstrap"
    )
    use_wandb: bool = Field(False, description="Whether to use wandb")
    _evaluated: bool = False

    _metrics: dict = {
        "accuracy": (accuracy_score, False, True, "Accuracy"),
        "precision": (precision_score, False, True, "Precision"),
        "recall": (recall_score, False, True, "Recall"),
        "f1": (f1_score, False, True, "F1 Score"),
        "roc_auc": (roc_auc_score, False, False, "ROC AUC"),
        "pr_auc": (pr_auc_score, False, False, "PR AUC"),
    }

    def evaluate(self, y_true=None, y_pred=None, use_wandb=False, tag=None, **kwargs):
        """
        Evaluate the classification model and compute metrics.

        Parameters
        ----------
        y_true : array-like
            True labels.
        y_pred : array-like
            Predicted probabilities or class labels.
        use_wandb : bool, optional
            Whether to log metrics to Weights & Biases.
        tag : str, optional
            Tag for the evaluation run.
        kwargs: Dict
            Additional keyword arguments

        Returns
        -------
        dict
            Dictionary of computed metrics and confidence intervals.

        """
        if y_true is None or y_pred is None:
            raise ValueError("Must provide y_true and y_pred")

        # Cast as numpy arrays
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)

        self.data = {"tag": tag}

        if use_wandb:
            self.use_wandb = use_wandb

        for metric_tag, (metric, is_scipy, is_class_pred, _) in self._metrics.items():
            # Binary case
            if (y_true.ndim == 1) or (y_true.ndim == 2 and y_true.shape[1] == 1):
                # Cast to class predictions before calculating the metric
                if is_class_pred is True:
                    _y_pred = np.argmax(y_pred, axis=1).ravel()
                    _y_true = y_true.ravel()

                # Compare probabilities with labels
                else:
                    _y_pred = y_pred[:, 1].ravel()
                    _y_true = y_true.ravel()

            # Multiclass case
            else:
                # Cast to class predictions before calculating the metric
                if is_class_pred is True:
                    _y_pred = np.argmax(y_pred, axis=1).ravel()
                    _y_true = np.argmax(y_true, axis=1).ravel()

                # Micro-averaged one-versus-rest
                else:
                    _y_pred = y_pred.ravel()
                    _y_true = y_true.ravel()

            value, lower_ci, upper_ci = self.stat_and_bootstrap(
                metric_tag,
                _y_pred,
                _y_true,
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

    @property
    def metric_names(self):
        """
        Return the metric names.

        Returns
        -------
        list of str
            List of metric names.

        """
        return list(self._metrics.keys())

    def report(self, write=False, output_dir=None):
        """
        Report the evaluation results, optionally writing to disk.

        Parameters
        ----------
        write : bool, optional
            Whether to write the report to disk.
        output_dir : str, optional
            Output directory for the report.

        Returns
        -------
        dict
            Dictionary of computed metrics.

        """
        if write:
            self.write_report(output_dir)
        return self.data

    def write_report(self, output_dir):
        """
        Write the evaluation report to a JSON file and optionally log to wandb.

        Parameters
        ----------
        output_dir : str
            Output directory for the report.

        Returns
        -------
        None

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


@evaluators.register("ClassificationPlots")
class ClassificationPlots(EvalBase):
    """
    Generate and save classification plots such as ROC and PR curves.

    Attributes
    ----------
    plots : dict
        Dictionary of plot functions.
    use_wandb : bool
        Whether to log plots to Weights & Biases.
    dpi : int
        DPI for the plots.

    """

    plots: dict = {}
    use_wandb: bool = Field(False, description="Whether to use wandb")
    dpi: int = Field(300, description="DPI for the plot")

    def evaluate(self, y_true=None, y_pred=None, use_wandb=False, **kwargs):
        """
        Generate classification plots.

        Parameters
        ----------
        y_true : array-like
            True labels.
        y_pred : array-like
            Predicted probabilities or class labels.
        use_wandb : bool, optional
            Whether to log plots to Weights & Biases.
        kwargs : Dict
            Additional keyword arguments

        Returns
        -------
        None

        """
        # Cast as numpy arrays
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)

        if use_wandb:
            self.use_wandb = use_wandb

        if y_true is None or y_pred is None:
            raise ValueError("Must provide y_true and y_pred")

        self.plots = {
            "roc_curve": self.roc_curve,
            "pr_curve": self.pr_curve,
        }

        self.plot_data = {}

        # Create the plots
        for plot_tag, plot in self.plots.items():
            self.plot_data[plot_tag] = plot(
                y_true,
                y_pred,
            )

    def roc_curve(
        self,
        y_true,
        y_pred,
        xlabel="False Positive Rate",
        ylabel="True Positive Rate",
        title="Receiver Operating Characteristic Curve",
    ):
        """
        Plot the ROC curve.

        Parameters
        ----------
        y_true : array-like
            True labels.
        y_pred : array-like
            Predicted probabilities or class labels.
        xlabel : str, optional
            Label for the x-axis.
        ylabel : str, optional
            Label for the y-axis.
        title : str, optional
            Title for the plot.

        Returns
        -------
        matplotlib.figure.Figure
            The ROC curve figure.

        """
        # Binary
        if (y_true.ndim == 1) or (y_true.ndim == 2 and y_true.shape[1] == 1):
            fpr, tpr, _ = roc_curve(y_true.ravel(), y_pred[:, 1].ravel())

        # Micro-averaged one-versus-rest
        else:
            fpr, tpr, _ = roc_curve(y_true.ravel(), y_pred.ravel())

        fig, ax = plt.subplots(dpi=self.dpi)
        ax.set_title(title, fontsize=10)

        ax.plot(fpr, tpr)
        ax.plot([0, 1], [0, 1], linestyle="--", color="black")

        ax.set_aspect("equal", "box")
        ax.set_xlabel(xlabel, fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)

        return fig

    def pr_curve(
        self,
        y_true,
        y_pred,
        xlabel="Recall",
        ylabel="Precision",
        title="Precision-Recall Curve",
    ):
        """
        Plot the precision-recall curve.

        Parameters
        ----------
        y_true : array-like
            True labels.
        y_pred : array-like
            Predicted probabilities or class labels.
        xlabel : str, optional
            Label for the x-axis.
        ylabel : str, optional
            Label for the y-axis.
        title : str, optional
            Title for the plot.

        Returns
        -------
        matplotlib.figure.Figure
            The precision-recall curve figure.

        """
        # Binary
        if (y_true.ndim == 1) or (y_true.ndim == 2 and y_true.shape[1] == 1):
            precision, recall, _ = precision_recall_curve(
                y_true.ravel(), y_pred[:, 1].ravel()
            )

        # Micro-averaged one-versus-rest
        else:
            precision, recall, _ = precision_recall_curve(
                y_true.ravel(), y_pred.ravel()
            )

        fig, ax = plt.subplots(dpi=self.dpi)
        ax.set_title(title, fontsize=10)

        ax.plot(recall, precision)
        ax.plot([0, 1], [1, 1], linestyle="--", color="black")
        ax.plot([1, 1], [0, 1], linestyle="--", color="black")

        ax.set_aspect("equal", "box")
        ax.set_xlabel(xlabel, fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)

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
        Write the generated plots to PNG files and optionally log to wandb.

        Parameters
        ----------
        output_dir : str
            Output directory for the plots.

        Returns
        -------
        None

        """
        for plot_tag, plot in self.plot_data.items():
            plot_path = output_dir / f"{plot_tag}.png"
            plot.savefig(plot_path, dpi=self.dpi)
            if self.use_wandb:
                wandb.log({plot_tag: wandb.Image(str(plot_path))})
