"""Cross-validation evaluators for regression models."""

import json
from collections import defaultdict

import pandas as pd
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

from openadmet.models.eval.eval_base import EvalBase, evaluators
from openadmet.models.eval.regression import (
    RegressionPlots,
    mask_nans,
    nan_omit_ktau,
    nan_omit_spearmanr,
)
from openadmet.models.trainer.lightning import LightningTrainer
from openadmet.models.eval.utils import _make_stat_caption, _make_stat_dict


def wrap_ktau(y_true, y_pred):
    """Wrap ktau nan omission."""
    return nan_omit_ktau(y_true, y_pred).statistic


def wrap_spearmanr(y_true, y_pred):
    """Wrap spearmanR nan omission."""
    return nan_omit_spearmanr(y_true, y_pred).correlation


class CrossValidationBase(EvalBase):
    """
    Base class for cross-validation evaluators.

    Attributes
    ----------
    _evaluated : bool
        Whether the evaluator has been run.
    axes_labels : list[str]
        Labels for the axes in plots.
    title : str
        Title for the plots.
    pXC50 : bool
        Whether to plot for pXC50, highlighting 0.5 and 1.0 log range unit.
    confidence_level : float
        Confidence level for the confidence interval.
    _metrics : dict
        Dictionary of metrics to evaluate.
    min_val : float
        Minimum value for the axes.
    max_val : float
        Maximum value for the axes.

    """

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
    min_val: float = Field(None, description="Minimum value for the axes")
    max_val: float = Field(None, description="Maximum value for the axes")

    @property
    def metric_names(self):
        """
        Get the list of metric names.

        Returns
        -------
        list of str
            List of metric names.

        """
        return list(self._metrics.keys())


@evaluators.register("SKLearnRepeatedKFoldCrossValidation")
class SKLearnRepeatedKFoldCrossValidation(CrossValidationBase):
    """
    Cross-validation evaluator for sklearn models (single-task regression).

    Attributes
    ----------
    n_splits : int
        Number of splits for cross-validation.
    n_repeats : int
        Number of repeats for cross-validation.
    random_state : int
        Random state for reproducibility.

    """

    n_splits: int = Field(5, description="Number of splits for cross-validation")
    n_repeats: int = Field(1, description="Number of repeats for cross-validation")
    random_state: int = Field(42, description="Random state for reproducibility")

    def evaluate(
        self,
        model=None,
        X_train=None,
        y_train=None,
        y_pred=None,
        y_true=None,
        tag=None,
        target_labels=None,
        **kwargs,
    ):
        """
        Evaluate the regression model using repeated K-fold cross-validation.

        Parameters
        ----------
        model : sklearn-like estimator
            The regression model to evaluate.
        X_train : array-like
            Training features.
        y_train : array-like
            Training targets.
        y_pred : array-like
            Predicted values (not used in cross-validation, but required for interface).
        y_true : array-like
            True values (not used in cross-validation, but required for interface).
        tag : str, optional
            Tag for the evaluation run.
        target_labels : list of str, optional
            List of target names.
        kwargs : Dict
            Additional keyword arguments.

        Returns
        -------
        dict
            Dictionary containing cross-validation metrics and confidence intervals.

        """
        if (
            model is None
            or X_train is None
            or y_train is None
            or y_pred is None
            or y_true is None
        ):
            raise ValueError(
                "model, X_train, y_train, y_pred, y_true, must be provided"
            )

        if isinstance(y_true, (pd.Series, pd.DataFrame)):
            y_true = y_true.to_numpy()

        # store the metric names and callables in dict suitable for sklearn cross_validate
        self.sklearn_metrics = {k: v[0] for k, v in self._metrics.items()}

        logger.info("Starting cross-validation")

        n_tasks = 1
        if target_labels is None:
            target_labels = [f"task_{i}" for i in range(n_tasks)]

        if len(target_labels) != n_tasks:
            raise ValueError(
                f"Number of target labels ({len(target_labels)}) must match number of tasks ({n_tasks})"
            )

        # run CV
        cv = RepeatedKFold(
            n_splits=self.n_splits,
            n_repeats=self.n_repeats,
            random_state=self.random_state,
        )

        estimator = model.estimator
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

        self.data = {"shape": [self.n_splits, self.n_repeats], "tag": tag}

        for task_id in range(n_tasks):
            t_label = target_labels[task_id]
            self.data[t_label] = {}
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
                self.data[t_label][k] = metric_data

        self._evaluated = True

        self.plots = {
            "cross_validation_regplot": RegressionPlots.regplot,
            "cross_validation_ciplot": RegressionPlots.ciplot,
        }

        self.plot_data = {}

        stat_dict = self.get_stat_dict(t_label=t_label)

        # create the plots
        for plot_tag, plot in self.plots.items():
            if "ciplot" in plot_tag:
                self.plot_data[plot_tag] = plot(stat_dict=stat_dict)
            elif "regplot" in plot_tag:
                self.plot_data[plot_tag] = plot(
                    y_true,
                    y_pred,
                    xlabel=self.axes_labels[0],
                    ylabel=self.axes_labels[1],
                    title=f"{self.title}\nTask: {t_label}",
                    stat_dict=stat_dict,
                    pXC50=self.pXC50,
                    min_val=self.min_val,
                    max_val=self.max_val,
                )

        return self.data

    def get_stat_caption(self, t_label):
        """
        Get a formatted statistics caption for a given task.

        Parameters
        ----------
        t_label : str
            Task label.

        Returns
        -------
        str
            Caption string with statistics.

        """
        if not self._evaluated:
            raise ValueError(
                ":( You must evaluate the model before the statistics caption can be made."
            )
        return _make_stat_caption(
            data=self.data,
            task_name=t_label,
            metric_names=self.metric_names,
            metrics=self._metrics,
            confidence_level=self.confidence_level,
            cv=True,
        )

    def get_stat_dict(self, t_label):
        """
        Get a statistics dictionary for a given task.

        Parameters
        ----------
        t_label : str
            Task label.

        Returns
        -------
        dict
            Dictionary of statistics for the task.

        """
        if not self._evaluated:
            raise ValueError(
                "R'uh-r'oh! You must evaluate the model before the statistics dict can be made."
            )
        return _make_stat_dict(
            data=self.data,
            task_name=t_label,
            metric_names=self.metric_names,
            metrics=self._metrics,
            confidence_level=self.confidence_level,
            cv=True,
        )

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
        Write the evaluation report and plots to disk.

        Parameters
        ----------
        output_dir : str
            Output directory for the report and plots.

        Returns
        -------
        None

        """
        # write to JSON
        with open(output_dir / "cross_validation_metrics.json", "w") as f:
            json.dump(self.data, f, indent=2)

        # write each plot to a file
        for plot_tag, plot in self.plot_data.items():
            plot.savefig(output_dir / f"{plot_tag}.png", bbox_inches="tight", dpi=900)


@evaluators.register("PytorchLightningRepeatedKFoldCrossValidation")
class PytorchLightningRepeatedKFoldCrossValidation(CrossValidationBase):
    """
    Cross-validation evaluator for PyTorch Lightning models.

    Attributes
    ----------
    n_splits : int
        Number of splits for cross-validation.
    n_repeats : int
        Number of repeats for cross-validation.
    random_state : int
        Random state for reproducibility.
    _evaluated : bool
        Whether the evaluator has been run.
    axes_labels : list[str]
        Labels for the axes in plots.
    title : str
        Title for the plots.
    pXC50 : bool
        Whether to plot for pXC50, highlighting 0.5 and 1.0 log range unit.
    confidence_level : float
        Confidence level for the confidence interval.
    _metrics : dict
        Dictionary of metrics to evaluate.
    min_val : float
        Minimum value for the axes.
    max_val : float
        Maximum value for the axes.
    use_wandb : bool
        Whether to use wandb for logging.

    """

    n_splits: int = Field(5, description="Number of splits for cross-validation")
    n_repeats: int = Field(1, description="Number of repeats for cross-validation")
    random_state: int = Field(42, description="Random state for reproducibility")
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
        "mse": (mean_squared_error, False, "MSE"),
        "mae": (mean_absolute_error, False, "MAE"),
        "r2": (r2_score, False, "$R^2$"),
        "ktau": (wrap_ktau, True, "Kendall's $\\tau$"),
        "spearmanr": (wrap_spearmanr, True, "Spearman's $\\rho$"),
    }

    min_val: float = Field(None, description="Minimum value for the axes")
    max_val: float = Field(None, description="Maximum value for the axes")
    use_wandb: bool = Field(False, description="Whether to use wandb")

    def evaluate(
        self,
        model=None,
        X_train=None,
        y_true=None,
        y_pred=None,
        y_train=None,
        X_train_raw=None,
        y_train_raw=None,
        featurizer=None,
        trainer=None,
        tag=None,
        use_wandb=False,
        target_labels=None,
        **kwargs,
    ):
        """
        Evaluate the regression model using repeated K-fold cross-validation with PyTorch Lightning.

        Parameters
        ----------
        model : LightningModelBase
            The PyTorch Lightning model to evaluate.
        X_train : array-like
            Training features.
        y_true : array-like
            True values for the full dataset.
        y_pred : array-like
            Predicted values for the full dataset.
        y_train : array-like
            Training targets.
        X_train_raw : array-like
            Raw training features (before featurization).
        y_train_raw : array-like
            Raw training targets (before featurization).
        featurizer : object
            Featurizer instance for data preprocessing.
        trainer : LightningTrainer
            Trainer instance for model training.
        tag : str, optional
            Tag for the evaluation run.
        use_wandb : bool, optional
            Whether to use Weights & Biases logging.
        target_labels : list of str, optional
            List of target names.
        kwargs : Dict
            Additional keyword arguments.

        Returns
        -------
        dict
            Dictionary containing cross-validation metrics and confidence intervals.

        """
        logger.info("Starting cross-validation")
        if (
            model is None
            or X_train is None
            or y_train is None
            or y_pred is None
            or y_true is None
            or tag is None
            or X_train_raw is None
            or y_train_raw is None
            or featurizer is None
            or trainer is None
        ):
            raise ValueError(
                "model, X_train, y_train, y_pred, y_true, and tag must be provided"
            )

        if isinstance(y_true, (pd.Series, pd.DataFrame)):
            y_true = y_true.to_numpy()

        self.data = {"tag": tag}

        if use_wandb:
            self.use_wandb = use_wandb

        # store the metric names and callables in dict suitable for sklearn cross_validate
        self.sklearn_metrics = {k: v[0] for k, v in self._metrics.items()}

        # run CV
        print(self.n_splits, self.n_repeats, self.random_state)
        cv = RepeatedKFold(
            n_splits=self.n_splits,
            n_repeats=self.n_repeats,
            random_state=self.random_state,
        )

        self.data = {
            "shape": [self.n_splits, self.n_repeats],
            "tag": tag,
        }

        self._metric_data = {}

        # cast to numpy arrays
        X_train_raw = X_train_raw.to_numpy()
        y_train_raw = y_train_raw.to_numpy()

        # prepare containers for metrics
        n_tasks = y_train_raw.shape[1]
        if target_labels is None:
            target_labels = [f"task_{i}" for i in range(n_tasks)]

        for task_id in range(n_tasks):
            t_label = target_labels[task_id]
            self._metric_data[t_label] = defaultdict(list)

        for fold, (fold_train_ids, fold_val_ids) in enumerate(
            cv.split(X=X_train_raw, y=y_train_raw)
        ):
            logger.info(f"Fold {fold}")

            X_train = X_train_raw[fold_train_ids]
            y_train = y_train_raw[fold_train_ids]
            X_val = X_train_raw[fold_val_ids]
            y_val = y_train_raw[fold_val_ids]

            # print shapes of matrices
            logger.debug(f"X_train shape: {X_train.shape}")
            logger.debug(f"y_train shape: {y_train.shape}")
            logger.debug(f"X_val shape: {X_val.shape}")
            logger.debug(f"y_val shape: {y_val.shape}")

            # Create a new featurizer and model for each fold
            fold_featurizer = featurizer.make_new()

            fold_train_dataloader, _, fold_train_scaler, _ = fold_featurizer.featurize(
                X_train, y_train
            )

            fold_val_dataloader, _, _, _ = fold_featurizer.featurize(X_val, y_val)
            fold_model = model.make_new()
            fold_model.build(scaler=fold_train_scaler)

            fold_trainer = LightningTrainer(
                max_epochs=trainer.max_epochs,
                accelerator=trainer.accelerator,
                devices=trainer.devices,
                use_wandb=False,
                output_dir=trainer.output_dir / "cv" / f"fold_{str(fold)}",
                wandb_project=trainer.wandb_project,
            )

            # Pass model to trainer
            fold_trainer.model = fold_model
            fold_trainer.build()

            # Pass the dataloaders to the trainer
            fold_model = fold_trainer.train(fold_train_dataloader, fold_val_dataloader)
            # evaluate the model
            y_pred_fold = fold_model.predict(
                fold_val_dataloader,
                accelerator=trainer.accelerator,
                devices=trainer.devices,
            )

            # calculate the mean and confidence interval for each metric
            # loop over tasks and calculate the statistics
            if not (n_tasks == y_pred_fold.shape[1]):
                raise ValueError("y_true and y_pred must have the same number of tasks")

            for task_id in range(n_tasks):
                t_true = y_val[:, task_id]
                t_pred = y_pred_fold[:, task_id]
                # remove Nan values
                t_true, t_pred = mask_nans(t_true, t_pred)
                t_label = target_labels[task_id]

                for metric_name, metric_data in self._metrics.items():
                    metric_func, is_scipy_metric, _ = metric_data
                    value = metric_func(t_true, t_pred)
                    self._metric_data[t_label][metric_name].append(value)

        logger.info(f"Fold {fold} complete")

        # now we have the metric data for each task, calculate the mean and confidence interval
        for t_label in target_labels:
            task_data = self._metric_data[t_label]
            self.data[t_label] = {}
            for k, v in task_data.items():
                # calculate the confidence interval, assuming normal distribution
                # TODO: check best practice???
                v = np.array(v)
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
                self.data[t_label][k] = metric_data

        self._evaluated = True

        self.plots = {
            "cross_validation_regplot": RegressionPlots.regplot,
            "cross_validation_ciplot": RegressionPlots.ciplot,
        }

        self.plot_data = {}

        # now the plots
        for task_id in range(n_tasks):
            t_true = y_true[:, task_id]
            t_pred = y_pred[:, task_id]
            # remove Nan values
            t_true, t_pred = mask_nans(t_true, t_pred)
            t_label = target_labels[task_id]

            stat_dict = self.get_stat_dict(t_label=t_label)

            # create the plots
            for plot_tag, plot in self.plots.items():
                plot_tag_task = f"{plot_tag}_{t_label}"
                if "ciplot" in plot_tag_task:
                    self.plot_data[plot_tag_task] = plot(stat_dict=stat_dict)
                elif "regplot" in plot_tag_task:
                    self.plot_data[plot_tag_task] = plot(
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

        return self.data

    @property
    def task_names(self):
        """
        Get the task names after evaluation.

        Returns
        -------
        list of str
            List of task names.

        """
        if not self._evaluated:
            raise ValueError("Must evaluate before getting task names")
        return list(self.data.keys())

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
        Write the evaluation report and plots to disk.

        Parameters
        ----------
        output_dir : str
            Output directory for the report and plots.

        Returns
        -------
        None

        """
        # write to JSON
        with open(output_dir / "cross_validation_metrics.json", "w") as f:
            json.dump(self.data, f, indent=2)

        # write each plot to a file
        for plot_tag, plot in self.plot_data.items():
            plot.savefig(output_dir / f"{plot_tag}.png", bbox_inches="tight", dpi=900)

    def get_stat_caption(self, t_label):
        """
        Get a formatted statistics caption for a given task.

        Parameters
        ----------
        t_label : str
            Task label.

        Returns
        -------
        str
            Caption string with statistics.

        """
        return _make_stat_caption(
            data=self.data,
            task_name=t_label,
            metric_names=self.metric_names,
            metrics=self._metrics,
            confidence_level=self.confidence_level,
            cv=True,
        )

    def get_stat_dict(self, t_label):
        """
        Get a statistics dictionary for a given task.

        Parameters
        ----------
        t_label : str
            Task label.

        Returns
        -------
        dict
            Dictionary of statistics for the task.

        """
        return _make_stat_dict(
            data=self.data,
            task_name=t_label,
            metric_names=self.metric_names,
            metrics=self._metrics,
            confidence_level=self.confidence_level,
            cv=True,
        )
