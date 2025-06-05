import json
from functools import partial

import numpy as np
import pandas as pd
import seaborn as sns
import wandb
from matplotlib import pyplot as plt
from pydantic import Field
from scipy.stats import kendalltau, spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from openadmet.models.eval.eval_base import EvalBase, evaluators, mask_nans

# create partial functions for the scipy stats
nan_omit_ktau = partial(kendalltau, nan_policy="omit")
nan_omit_spearmanr = partial(spearmanr, nan_policy="omit")


@evaluators.register("RegressionMetrics")
class RegressionMetrics(EvalBase):
    bootstrap_confidence_level: float = Field(
        0.95, description="Confidence level for the bootstrap"
    )
    use_wandb: bool = Field(False, description="Whether to use wandb")
    _evaluated: bool = False

    # tuple of metric, whether it is a scipy statistic, and the name to use in the report
    _metrics: dict = {
        "mse": (mean_squared_error, False, "MSE"),
        "mae": (mean_absolute_error, False, "MAE"),
        "r2": (r2_score, False, "$R^2$"),
        "ktau": (nan_omit_ktau, True, "Kendall's $\\tau$"),
        "spearmanr": (nan_omit_spearmanr, True, "Spearman's $\\rho$"),
    }

    def evaluate(
        self,
        y_true=None,
        y_pred=None,
        use_wandb=False,
        tag=None,
        target_labels=None,
        **kwargs,
    ):
        """
        Evaluate the regression model
        """
        if y_true is None or y_pred is None:
            raise ValueError("Must provide y_true and y_pred")

        n_tasks = y_true.shape[1]
        if not (n_tasks == y_pred.shape[1]):
            raise ValueError("y_true and y_pred must have the same number of tasks")
        if target_labels is None:
            target_labels = [f"task_{i}" for i in range(n_tasks)]

        self.data = {"tag": tag}

        if use_wandb:
            self.use_wandb = use_wandb

        if isinstance(y_true, pd.DataFrame):
            y_true = y_true.to_numpy()

        for task_id in range(n_tasks):
            t_true = y_true[:, task_id]
            t_pred = y_pred[:, task_id]
            # remove Nan values
            t_true, t_pred = mask_nans(t_true, t_pred)
            t_label = target_labels[task_id]

            self.data[t_label] = {}

            for metric_tag, (metric, is_scipy, _) in self._metrics.items():
                value, lower_ci, upper_ci = self.stat_and_bootstrap(
                    metric_tag,
                    t_pred,
                    t_true,
                    metric,
                    is_scipy_statistic=is_scipy,
                    confidence_level=self.bootstrap_confidence_level,
                )

                self.data[t_label][metric_tag] = {
                    "value": value,
                    "lower_ci": lower_ci,
                    "upper_ci": upper_ci,
                    "confidence_level": self.bootstrap_confidence_level,
                }

        if self.use_wandb:
            for t_label in target_labels:
                # make a table for the metrics
                table = wandb.Table(
                    columns=[
                        "Metric",
                        "Value",
                        "Lower CI",
                        "Upper CI",
                        "Confidence Level",
                    ]
                )
                for metric_tag in self.metric_names:
                    metric = self.data[t_label][metric_tag]
                    table.add_data(
                        metric_tag,
                        metric["value"],
                        metric["lower_ci"],
                        metric["upper_ci"],
                        metric["confidence_level"],
                    )
                wandb.log({f"metrics_{t_label}": table})

        self._evaluated = True
        return self.data

    @property
    def metric_names(self):
        """
        Return the metric names
        """
        return list(self._metrics.keys())

    @property
    def task_names(self):
        """
        Return the task names
        """
        return list(self.data.keys())

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
        json_path = output_dir / "regression_metrics.json"
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

        for task_name in self.task_names:
            if task_name == "tag":
                continue

            stat_caption += f"## {task_name} ##\n"
            for metric in self.metric_names:
                value = self.data[task_name][metric]["value"]
                lower_ci = self.data[task_name][metric]["lower_ci"]
                upper_ci = self.data[task_name][metric]["upper_ci"]
                confidence_level = self.data[task_name][metric]["confidence_level"]
                stat_caption += f"{self._metrics[metric][2]}: {value:.2f}$_{{{lower_ci:.2f}}}^{{{upper_ci:.2f}}}$\n"
            stat_caption += "\n"
        stat_caption += f"Confidence level: {confidence_level} \n"
        return stat_caption


@evaluators.register("RegressionPlots")
class RegressionPlots(EvalBase):
    axes_labels: list[str] = Field(
        ["Measured", "Predicted"], description="Labels for the axes"
    )
    title: str = Field("Pred vs ", description="Title for the plot")
    do_stats: bool = Field(True, description="Whether to do stats for the plot")
    pXC50: bool = Field(
        False,
        description="Whether to plot for pXC50, highlighting 0.5 and 1.0 log range unit",
    )
    plots: dict = {}
    min_val: float = Field(None, description="Minimum value for the axes")
    max_val: float = Field(None, description="Maximum value for the axes")
    use_wandb: bool = Field(False, description="Whether to use wandb")
    dpi: int = Field(300, description="DPI for the plot")

    def evaluate(
        self, y_true=None, y_pred=None, use_wandb=False, target_labels=None, **kwargs
    ):
        """
        Evaluate the regression model
        """
        if use_wandb:
            self.use_wandb = use_wandb

        if y_true is None or y_pred is None:
            raise ValueError("Must provide y_true and y_pred")

        n_tasks = y_true.shape[1]
        if not (n_tasks == y_pred.shape[1]):
            raise ValueError("y_true and y_pred must have the same number of tasks")
        if target_labels is None:
            target_labels = [f"task_{i}" for i in range(n_tasks)]

        self.plots = {
            "regplot": self.regplot,
        }

        self.plot_data = {}

        if isinstance(y_true, pd.DataFrame):
            y_true = y_true.to_numpy()

        for task_id in range(n_tasks):
            t_true = y_true[:, task_id]
            t_pred = y_pred[:, task_id]
            # remove Nan values
            t_true, t_pred = mask_nans(t_true, t_pred)
            t_label = target_labels[task_id]

            if self.do_stats:
                rm = RegressionMetrics()
                rm.evaluate(
                    t_true.reshape(-1, 1),
                    t_pred.reshape(-1, 1),
                    target_labels=[t_label],
                )
                stat_caption = rm.make_stat_caption()
            else:
                stat_caption = ""

            # create the plots
            for plot_tag, plot in self.plots.items():
                self.plot_data[t_label] = plot(
                    t_true,
                    t_pred,
                    xlabel=self.axes_labels[0],
                    ylabel=self.axes_labels[1],
                    title=f"{self.title}\nTask: {t_label}",
                    stat_caption=stat_caption,
                    pXC50=self.pXC50,
                    min_val=self.min_val,
                    max_val=self.max_val,
                )

    @staticmethod
    def regplot(
        y_true,
        y_pred,
        xlabel="Measured",
        ylabel="Predicted",
        title="",
        stat_caption="",
        confidence_level=0.95,
        pXC50=False,
        min_val=None,
        max_val=None,
    ):
        """
        Create a regression plot
        """
        fig, ax = plt.subplots()
        ax.set_title(title, fontsize=10)
        if min_val is None:
            min_val = min(np.min(y_true), np.min(y_pred))
            min_ax = min_val - 1
        else:
            min_ax = min_val
        if max_val is None:
            max_val = max(np.max(y_true), np.max(y_pred))
            max_ax = max_val + 1
        else:
            max_ax = max_val
        # set the limits to be the same for both axes
        _ = sns.regplot(x=y_true, y=y_pred, ax=ax, ci=confidence_level * 100)
        # slope, intercept, r, p, sterr = scipy.stats.linregress(
        #     x=p.get_lines()[0].get_xdata(), y=p.get_lines()[0].get_ydata()
        # )
        ax.set_aspect("equal", "box")

        ax.set_xlim(min_ax, max_ax)
        ax.set_ylim(min_ax, max_ax)
        # plot y = x line in dashed grey
        ax.plot([min_ax, max_ax], [min_ax, max_ax], linestyle="--", color="black")

        # if pXC50 measure then plot the 0.5 and 1.0 log range unit
        if pXC50:
            ax.fill_between(
                [min_ax, max_ax],
                [min_ax - 0.5, max_ax - 0.5],
                [min_ax + 0.5, max_ax + 0.5],
                color="gray",
                alpha=0.2,
            )
            ax.fill_between(
                [min_ax, max_ax],
                [min_ax - 1, max_ax - 1],
                [min_ax + 1, max_ax + 1],
                color="gray",
                alpha=0.2,
            )
        ax.set_xlabel(xlabel, fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.text(0.05, 0.7, stat_caption, transform=ax.transAxes, fontsize=5)
        fig.tight_layout()

        return fig

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
            plot_path = output_dir / f"regplot_{plot_tag}.png"
            plot.savefig(plot_path, dpi=self.dpi)
            if self.use_wandb:
                wandb.log({plot_tag: wandb.Image(str(plot_path))})
