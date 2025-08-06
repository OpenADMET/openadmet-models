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
from openadmet.models.eval.utils import _make_stat_caption, _make_stat_dict

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

        if isinstance(y_true, (pd.Series, pd.DataFrame)):
            y_true = y_true.to_numpy()

        # Ensure y_pred and y_true are 2D arrays for consistency
        if y_pred.ndim == 1:
            y_pred = y_pred.reshape(-1, 1)
        if y_true.ndim == 1:
            y_true = y_true.reshape(-1, 1)

        n_tasks = y_true.shape[1]
        if not (n_tasks == y_pred.shape[1]):
            raise ValueError("y_true and y_pred must have the same number of tasks")
        if target_labels is None:
            target_labels = [f"task_{i}" for i in range(n_tasks)]

        self.data = {"tag": tag}

        if use_wandb:
            self.use_wandb = use_wandb

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

    def get_stat_caption(self, t_label):
        if not self._evaluated:
            raise ValueError(":( You must evaluate the model before the statistics caption can be made.")
        return _make_stat_caption(data=self.data,
                                  task_name=t_label,
                                  metric_names=self.metric_names,
                                  metrics=self._metrics,
                                  confidence_level=self.bootstrap_confidence_level,
                                  cv=False)

    def get_stat_dict(self, t_label):
        if not self._evaluated:
            raise ValueError("R'uh-r'oh! You must evaluate the model before the statistics dict can be made.")
        return _make_stat_dict(data=self.data,
                               task_name=t_label,
                               metric_names=self.metric_names,
                               metrics=self._metrics,
                               confidence_level=self.bootstrap_confidence_level,
                               cv=False)



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

        if isinstance(y_true, (pd.Series, pd.DataFrame)):
            y_true = y_true.to_numpy()

        # Ensure y_pred and y_true are 2D arrays for consistency
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
            "regplot": self.regplot,
            "ciplot": self.ciplot
        }

        self.plot_data = {}

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
                stat_dict = rm.get_stat_dict(t_label=t_label)
            else:
                stat_dict = {}

            # create the plots
            for plot_tag, plot in self.plots.items():
                if "ciplot" in plot_tag:
                    self.plot_data[f"{t_label}_{plot_tag}"] = plot(stat_dict=stat_dict)
                elif "regplot" in plot_tag:
                    self.plot_data[f"{t_label}_{plot_tag}"] = plot(
                        t_true,
                        t_pred,
                        xlabel=self.axes_labels[0],
                        ylabel=self.axes_labels[1],
                        title=f"{self.title}\nTask: {t_label}",
                        stat_dict=stat_dict,
                        pXC50=self.pXC50,
                        min_val=self.min_val,
                        max_val=self.max_val,
                    )
        return self.plot_data

    @staticmethod
    def regplot(
        y_true,
        y_pred,
        xlabel="Measured",
        ylabel="Predicted",
        title="",
        stat_dict={},
        confidence_level=0.95,
        pXC50=False,
        min_val=None,
        max_val=None,
    ):
        """
        Create a regression plot
        """
        title_font = 20
        ax_font = 18
        tick_font = 16
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

        g = sns.jointplot(
            x=np.ravel(y_true),
            y=np.ravel(y_pred),
            kind="reg",
            joint_kws={"ci": confidence_level * 100},
            scatter_kws={"alpha":0.3},
            color="teal",
            height=10)

        g.figure.suptitle(title, fontsize=title_font)
        g.ax_joint.set_aspect("equal", "box")
        g.ax_joint.set_xlim(min_ax, max_ax)
        g.ax_joint.set_ylim(min_ax, max_ax)
        g.ax_joint.tick_params(axis='both', labelsize=tick_font)
        # plot y = x line in dashed grey
        g.ax_joint.plot([min_ax, max_ax], [min_ax, max_ax], linestyle="--", color="black")

        # if pXC50 measure then plot the 0.5 and 1.0 log range unit
        if pXC50:
            g.ax_joint.fill_between(
                [min_ax, max_ax],
                [min_ax - 0.5, max_ax - 0.5],
                [min_ax + 0.5, max_ax + 0.5],
                color="gray",
                alpha=0.2,
            )
            g.ax_joint.fill_between(
                [min_ax, max_ax],
                [min_ax - 1, max_ax - 1],
                [min_ax + 1, max_ax + 1],
                color="gray",
                alpha=0.2,
            )
        g.ax_joint.set_xlabel(xlabel, fontsize=ax_font)
        g.ax_joint.set_ylabel(ylabel, fontsize=ax_font)

        # From the stat_dict, parse out the performance metric values and their labels to put into a table to print on the regression plot
        if stat_dict:
            conf_level = stat_dict.get("conf_level", None)
            metric_names = stat_dict.get("metrics", [])
            values = stat_dict.get("means", [])
            lower_bounds = stat_dict.get("lower_ci", [])
            upper_bounds = stat_dict.get("upper_ci", [])

            table_data = []
            # Format the metric values for readability
            for name, val, low, high in zip(metric_names, values, lower_bounds, upper_bounds):
                if None not in (val, low, high):
                    val_str = f"{val:.2f} [{low:.2f}, {high:.2f}]"
                else:
                    val_str = "N/A"
                table_data.append([name, val_str])
            # Create the table
            table = g.ax_joint.table(
                cellText=table_data,
                colLabels=["Metric", f"Value ± {int(conf_level*100)}% CI"],
                colWidths=[0.2, 0.3],
                loc="upper left",
                cellLoc="left",
            )

            table.scale(1, 1.8)
            for key, cell in table.get_celld().items():
                cell.set_fontsize(ax_font)
            # Right align the metric values
            for i in range(1, len(table_data) + 1):
                table[i, 1].get_text().set_horizontalalignment("right")

        g.ax_joint.set_box_aspect(1)
        g.figure.tight_layout()
        return g

    @staticmethod
    def ciplot(
        stat_dict={}
    ):

        metrics = stat_dict["metrics"]
        means = stat_dict["means"]
        lower_ci = stat_dict["lower_ci"]
        upper_ci = stat_dict["upper_ci"]
        conf_level = stat_dict["conf_level"]
        task_name = stat_dict["task_name"]

        title_font = 16
        tick_font = 12
        ax_font = 14

        padding = 0.2
        y_limits = {
            "MSE": (0-padding, 1+padding),
            "MAE": (0-padding, 1+padding),
            "$R^2$": (-1-padding, 1+padding),
            "Kendall's $\\tau$": (-1-padding, 1+padding),
            "Spearman's $\\rho$": (-1-padding, 1+padding)
        }

        n_metrics = len(metrics)
        fig, axes = plt.subplots(1, n_metrics, figsize=(8, n_metrics), sharex=False)

        if n_metrics == 1:
            axes = [axes]  # Ensure it's iterable

        for i, ax in enumerate(axes):
            if i == 0:
                ax.set_ylabel("Performance Metric Value", fontsize=ax_font)
            metric = metrics[i]
            y = means[i]
            yerr = [[y - lower_ci[i]], [upper_ci[i] - y]]
            ax.errorbar([metric], [y], yerr=yerr, fmt='o', capsize=8, color='green')
            ax.tick_params(axis='both', labelsize=tick_font)
            ax.yaxis.grid(True, linestyle='--', color='lightgray', alpha=0.6)
            ax.set_xlim(-0.5, 0.5)

            # Set fixed y-limits
            if metric in y_limits:
                ax.set_ylim(y_limits[metric])

        fig.suptitle(f"Evaluation of {task_name} with {int(conf_level * 100)}% Confidence Intervals", fontsize=title_font)
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
            plot_path = output_dir / f"{plot_tag}.png"
            plot.savefig(plot_path, dpi=self.dpi)
            if self.use_wandb:
                wandb.log({plot_tag: wandb.Image(str(plot_path))})
