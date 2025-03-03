import json
from functools import partial
from typing import Callable

import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from pydantic import Field
from scipy.stats import bootstrap, kendalltau, spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from openadmet_models.eval.eval_base import EvalBase, evaluators


def stat_and_bootstrap(
    metric_tag: str,
    y_pred: np.ndarray,
    y_true: np.ndarray,
    statistic: Callable,
    confidence_level: float = 0.95,
    is_scipy_statistic: bool = False,
):
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


# create partial functions for the scipy stats
nan_omit_ktau = partial(kendalltau, nan_policy="omit")
nan_omit_spearmanr = partial(spearmanr, nan_policy="omit")


@evaluators.register("RegressionMetrics")
class RegressionMetrics(EvalBase):
    bootstrap_confidence_level: float = Field(
        0.95, description="Confidence level for the bootstrap"
    )
    _evaluated: bool = False

    # tuple of metric, whether it is a scipy statistic, and the name to use in the report
    _metrics: dict = {
        "mse": (mean_squared_error, False, "MSE"),
        "mae": (mean_absolute_error, False, "MAE"),
        "r2": (r2_score, False, "$R^2$"),
        "ktau": (nan_omit_ktau, True, "Kendall's $\\tau$"),
        "spearmanr": (nan_omit_spearmanr, True, "Spearman's $\\rho$"),
    }

    def evaluate(self, y_true=None, y_pred=None, **kwargs):
        """
        Evaluate the regression model
        """
        if y_true is None or y_pred is None:
            raise ValueError("Must provide y_true and y_pred")

        self.data = {}

        for metric_tag, (metric, is_scipy, _) in self._metrics.items():
            value, lower_ci, upper_ci = stat_and_bootstrap(
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

        self._evaluated = True

        return self.data

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
        with open(output_dir / "regression_metrics.json", "w") as f:
            json.dump(self.data, f, indent=2)

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

    def evaluate(self, y_true=None, y_pred=None, **kwargs):
        """
        Evaluate the regression model
        """
        if y_true is None or y_pred is None:
            raise ValueError("Must provide y_true and y_pred")

        self.plots = {
            "regplot": self.regplot,
        }

        self.plot_data = {}

        if self.do_stats:
            rm = RegressionMetrics()
            rm.evaluate(y_true, y_pred)
            stat_caption = rm.make_stat_caption()

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
            min_val = min(y_true.min(), y_pred.min())
            min_ax = min_val - 1
        else:
            min_ax = min_val
        if max_val is None:
            max_val = max(y_true.max(), y_pred.max())
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
        ax.text(0.05, 0.7, stat_caption, transform=ax.transAxes, fontsize=6)

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
        # write each plot to a file
        for plot_tag, plot in self.plot_data.items():
            plot.savefig(output_dir / f"{plot_tag}.png", dpi=900)
