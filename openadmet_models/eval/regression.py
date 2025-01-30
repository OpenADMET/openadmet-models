from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from typing import Callable
from matplotlib import pyplot as plt
from openadmet_models.eval.eval_base import EvalBase, evaluators
import numpy as np


def stat_and_bootstrap(metric_tag: str, y_pred: np.ndarray, y_true: np.ndarray,  statistic: Callable, n_resamples: int=10000, confidence_level: float=0.95):

    # calculate the metric and confidence intervals
    metric = statistic(y_true, y_pred)
    conf_interval = bootstrap(
        (y_true, y_pred),
        statistic=statistic,
        method="basic",
        confidence_level=confidence_level,
        paired=True,
        n_bootstraps=n_bootstraps,
    ).confidence_interval

    return metric, conf_interval.low, conf_interval.high,



@evaluators.register("RegressionMetrics")
class RegressionMetrics(EvalBase):
    metrics: dict = {}

    def evaluate(self, y_true, y_pred):
        """
        Evaluate the regression model
        """

        self.metrics = {
            "mse": mean_squared_error,
            "mae": mean_absolute_error,
            "r2": r2_score,
        }

        self.data = {}

        for metric_tag, metric in self.metrics.items():
            value, lower_ci, upper_ci = do_stat_and_bootstrap(
                metric_tag, y_pred, y_true, metric
            )

            self.data[f"{metric_tag}_{value}"] = value
            self.data[f"{metric_tag}_lower_ci"] = lower_ci
            self.data[f"{metric_tag}_upper_ci"] = upper_ci

        

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
            json.dump(self.data, f)


class RegressionPlots(EvalBase):
    plots: dict = {}

    def evaluate(self, y_true, y_pred):
        """
        Evaluate the regression model
        """
        
        self.plots = {
            "regplot": self.regplot,
        }


        self.data = {}
        # create the plots
        for plot_tag, plot in self.plots.items():
            self.data[plot_tag] = plot(y_true, y_pred)


    @staticmethod
    def regplot(y_true, y_pred):
        """
        Create a regression plot
        """
        fig, ax = plt.subplots()
        ax.set_title(f"Test set performance:\n {model_name}", fontsize=6)
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        # set the limits to be the same for both axes
        p = sns.regplot(x=y_true, y=y_pred, ax=ax, ci=None)
        slope, intercept, r, p, sterr = scipy.stats.linregress(
            x=p.get_lines()[0].get_xdata(), y=p.get_lines()[0].get_ydata()
        )
        ax.set_aspect("equal", "box")
        min_ax = min_val - 1
        max_ax = max_val + 1

        ax.set_xlim(min_ax, max_ax)
        ax.set_ylim(min_ax, max_ax)
        # plot y = x line in dashed grey
        ax.plot([min_ax, max_ax], [min_ax, max_ax], linestyle="--", color="black")
        return fig

    
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
        # write each plot to a file
        for plot_tag, plot in self.data.items():
            plot.savefig(output_dir / f"{plot_tag}.png")
        