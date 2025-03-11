import json

import numpy as np
from loguru import logger
from pydantic import Field
from scipy.stats import norm
from sklearn.metrics import (
    make_scorer,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import RepeatedKFold, cross_validate

from openadmet_models.eval.eval_base import EvalBase, evaluators
from openadmet_models.eval.regression import (
    RegressionPlots,
    nan_omit_ktau,
    nan_omit_spearmanr,
)


def wrap_ktau(y_true, y_pred):
    return nan_omit_ktau(y_true, y_pred).statistic


def wrap_spearmanr(y_true, y_pred):
    return nan_omit_spearmanr(y_true, y_pred).correlation


@evaluators.register("SKLearnRepeatedKFoldCrossValidation")
class SKLearnRepeatedKFoldCrossValidation(EvalBase):
    n_splits: int = 5
    n_repeats: int = 5
    random_state: int = 42
    _evaluated: bool = False
    axes_labels: list[str] = Field(
        ["Measured", "Predicted"], description="Labels for the axes"
    )
    title: str = Field("Pred vs ", description="Title for the plot")
    pXC50: bool = Field(
        False,
        description="Whether to plot for pXC50, highlighting 0.5 and 1.0 log range unit",
    )
    confidence_level: float = Field(
        0.95, description="Confidence level for the confidence interval"
    )

    _metrics: dict = {
        "mse": (make_scorer(mean_squared_error), False, "MSE"),
        "mae": (make_scorer(mean_absolute_error), False, "MAE"),
        "r2": (make_scorer(r2_score), False, "$R^2$"),
        "ktau": (make_scorer(wrap_ktau), True, "Kendall's $\\tau$"),
        "spearmanr": (make_scorer(wrap_spearmanr), True, "Spearman's $\\rho$"),
    }

    def evaluate(
        self, model=None, X_train=None, y_train=None, y_pred=None, y_true=None, **kwargs
    ):
        """
        Evaluate the regression model
        """
        if (
            model is None
            or X_train is None
            or y_train is None
            or y_pred is None
            or y_true is None
        ):
            raise ValueError(
                "model, X_train, y_train, y_pred, and y_true must be provided"
            )

        # store the metric names and callables in dict suitable for sklearn cross_validate
        self.sklearn_metrics = {k: v[0] for k, v in self._metrics.items()}

        logger.info("Starting cross-validation")

        # run CV
        cv = RepeatedKFold(
            n_splits=self.n_splits,
            n_repeats=self.n_repeats,
            random_state=self.random_state,
        )

        estimator = model.model
        # evaluate the model, storing the results
        # we do one job here to avoid issues with double parallelization
        # we prefer to parallelize model training over cross-validation
        scores = cross_validate(
            estimator, X_train, y_train, cv=cv, n_jobs=1, scoring=self.sklearn_metrics
        )

        logger.info("Cross-validation complete")

        # remove the 'test_' prefix from the keys
        # also convert the numpy arrays to lists so they can be serialized to JSON
        clean_scores = {}
        for k, v in scores.items():
            clean_scores[k.replace("test_", "")] = v

        # exclude fit_time and score_time
        exclude = ["fit_time", "score_time"]

        self.data = {"shape": [self.n_splits, self.n_repeats]}
        for k, v in clean_scores.items() if k not in exclude else {}:
            # calculate the confidence interval, assuming normal distribution
            # TODO: check best practice???
            mean = v.mean()
            sigma = v.std(ddof=1)
            lower_ci, upper_ci = norm.interval(
                self.confidence_level, loc=mean, scale=sigma
            )
            metric_data = {}
            metric_data["value"] = v.tolist()
            metric_data["mean"] = np.mean(v)
            metric_data["lower_ci"] = lower_ci
            metric_data["upper_ci"] = upper_ci
            metric_data["confidence_level"] = self.confidence_level
            self.data[k] = metric_data

        self._evaluated = True

        self.plots = {
            "cross_validation_regplot": RegressionPlots.regplot,
        }

        self.plot_data = {}

        stat_caption = self.make_stat_caption()

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

        return self.data

    @property
    def metric_names(self):
        """
        Return the metric names
        """
        return list(self._metrics.keys())

    def make_stat_caption(self):
        """
        Make a caption for the statistics
        """
        if not self._evaluated:
            raise ValueError("Must evaluate before making a caption")
        stat_caption = ""
        for metric in self.metric_names:
            value = self.data[metric]["mean"]
            lower_ci = self.data[metric]["lower_ci"]
            upper_ci = self.data[metric]["upper_ci"]
            stat_caption += f"{self._metrics[metric][2]}: {value:.2f}$_{{{lower_ci:.2f}}}^{{{upper_ci:.2f}}}$\n"
        stat_caption += f"Confidence level: {self.confidence_level}"
        return stat_caption

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
        with open(output_dir / "cross_validation_metrics.json", "w") as f:
            json.dump(self.data, f, indent=2)

        # write each plot to a file
        for plot_tag, plot in self.plot_data.items():
            plot.savefig(output_dir / f"{plot_tag}.png", dpi=900)
