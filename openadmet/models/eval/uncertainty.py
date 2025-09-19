"""Evaluators for uncertainty quantification metrics and plots."""

import json

import matplotlib.pyplot as plt
import pandas as pd
import uncertainty_toolbox as uct
import wandb
from pydantic import Field

from openadmet.models.eval.eval_base import EvalBase, evaluators, mask_nans_std


@evaluators.register("UncertaintyMetrics")
class UncertaintyMetrics(EvalBase):
    """
    Evaluator for uncertainty metrics using uncertainty_toolbox.

    Attributes
    ----------
    use_wandb : bool
        Whether to use wandb for logging.
    _data : dict
        Stores computed metrics for each task.
    _metrics : dict
        Mapping of metric keys to human-readable names.

    """

    use_wandb: bool = Field(False, description="Whether to use wandb")
    _data: dict = {}
    _metrics: dict = {
        "mae": "MAE",
        "rmse": "RMSE",
        "mdae": "MDAE",
        "marpd": "MARPD",
        "r2": "$R^2$",
        "corr": "Correlation",
        "rms_cal": "Root-mean-squared Calibration Error",
        "ma_cal": "Mean-absolute Calibration Error",
        "miscal_area": "Miscalibration Area",
        "sharp": "Sharpness",
        "nll": "Negative-log-likelihood",
        "crps": "CRPS",
        "check": "Check Score",
        "interval": "Interval Score",
        "rms_adv_group_cal": "Root-mean-squared Adversarial Group Calibration Error",
        "ma_adv_group_cal": "Mean-absolute Adversarial Group Calibration Error",
    }

    @property
    def metric_names(self):
        """
        Get the list of metric keys.

        Returns
        -------
        list of str
            List of metric keys.

        """
        return list(self._metrics.keys())

    @property
    def task_names(self):
        """
        Get the list of evaluated task names.

        Returns
        -------
        list of str
            List of task names.

        """
        return list(self._data.keys())

    def evaluate(
        self,
        y_true,
        y_pred,
        y_std,
        target_labels=None,
        bins=100,
        resolution=99,
        scaled=True,
        **kwargs,
    ):
        """
        Evaluate uncertainty metrics for each task.

        Parameters
        ----------
        y_true : array-like
            Ground truth values.
        y_pred : array-like
            Predicted mean values.
        y_std : array-like
            Predicted standard deviations.
        target_labels : list of str, optional
            List of target labels for each task.
        bins : int, default=100
            Number of bins for calibration metrics.
        resolution : int, default=99
            Resolution for scoring rule metrics.
        scaled : bool, default=True
            Whether to scale scoring rule metrics.
        **kwargs
            Additional keyword arguments.

        Raises
        ------
        ValueError
            If required inputs are missing or shapes are inconsistent.

        """
        # Check inputs
        if y_true is None or y_pred is None or y_std is None:
            raise ValueError("Must provide `y_true`, `y_pred`, and `y_std`")

        # Convert to numpy array if needed
        if isinstance(y_true, (pd.Series, pd.DataFrame)):
            y_true = y_true.to_numpy()

        # Ensure 2D arrays for consistency
        if y_pred.ndim == 1:
            y_pred = y_pred.reshape(-1, 1)
        if y_true.ndim == 1:
            y_true = y_true.reshape(-1, 1)
        if y_std.ndim == 1:
            y_std = y_std.reshape(-1, 1)

        # Verify number of tasks
        n_tasks = y_true.shape[1]
        if (n_tasks != y_pred.shape[1]) or (n_tasks != y_std.shape[1]):
            raise ValueError(
                "`y_true`, `y_pred`, and `y_std` must have the same number of tasks"
            )

        # Construct target labels if not provided
        if target_labels is None:
            target_labels = [f"task_{i}" for i in range(n_tasks)]

        # Enumerate targets
        for task_id, task_label in enumerate(target_labels):
            # Task target values
            t_true = y_true[:, task_id].flatten()
            t_pred = y_pred[:, task_id].flatten()
            t_std = y_std[:, task_id].flatten()

            # Mask nans
            t_true, t_pred, t_std = mask_nans_std(t_true, t_pred, t_std)

            # Initialize task data
            self._data[task_label] = {}

            # Accuracy
            accuracy_metrics = uct.metrics.get_all_accuracy_metrics(
                t_pred, t_true, False
            )

            # Calibration
            calibration_metrics = uct.metrics.get_all_average_calibration(
                t_pred, t_std, t_true, bins, False
            )

            # # Adversarial Group Calibration
            # adv_group_cali_metrics = uct.metrics.get_all_adversarial_group_calibration(
            #     t_pred,
            #     t_std,
            #     t_true,
            #     bins,
            #     False,
            # )

            # Sharpness
            sharpness_metrics = uct.metrics.get_all_sharpness_metrics(t_std, False)

            # Proper Scoring Rules
            scoring_rule_metrics = uct.metrics.get_all_scoring_rule_metrics(
                t_pred,
                t_std,
                t_true,
                resolution,
                scaled,
                False,
            )

            # Store metrics
            for metric_dict in [
                accuracy_metrics,
                calibration_metrics,
                sharpness_metrics,
                scoring_rule_metrics,
                # adv_group_cali_metrics,
            ]:
                self._data[task_label].update(metric_dict)

    def report(self, write=False, output_dir=None):
        """
        Report the evaluation results.

        Parameters
        ----------
        write : bool, default=False
            Whether to write the report to disk.
        output_dir : Path or str, optional
            Directory to write the report to.

        Returns
        -------
        dict
            Dictionary of computed metrics.

        """
        if write:
            self.write_report(output_dir)

        return self._data

    def write_report(self, output_dir):
        """
        Write the evaluation report to disk and optionally log to wandb.

        Parameters
        ----------
        output_dir : Path or str
            Directory to write the report to.

        """
        # Write to JSON
        json_path = output_dir / "uncertainty_calibration_metrics.json"
        with open(json_path, "w") as f:
            json.dump(self._data, f, indent=2)

        # Also log the JSON to wandb
        if self.use_wandb:
            artifact = wandb.Artifact(
                name="uncertainty_calibration_json", type="metric_json"
            )
            # Add a file to the artifact
            artifact.add_file(json_path)
            # Log the artifact
            wandb.log_artifact(artifact)


@evaluators.register("UncertaintyPlots")
class UncertaintyPlots(EvalBase):
    """
    Evaluator for generating uncertainty plots.

    Attributes
    ----------
    use_wandb : bool
        Whether to use wandb for logging.
    dpi : int
        DPI for the generated plots.
    _plots : dict
        Mapping of plot tags to plotting functions.
    _plot_data : dict
        Stores generated plot figures.

    """

    use_wandb: bool = Field(False, description="Whether to use wandb")
    dpi: int = Field(300, description="DPI for the plot")
    _plots: dict = {}
    _plot_data: dict = {}

    def model_post_init(self, __context):
        """
        Post-initialization hook to set plot types.

        Parameters
        ----------
        __context : Any
            Pydantic context (unused).

        """
        self._set_plot_types()

    def _set_plot_types(self):
        """Set the available plot types."""
        # Specify plots
        self._plots = {
            "uncertainty-calibration-plot": self.calibration_plot,
        }

    def evaluate(self, y_true, y_pred, y_std, target_labels=None, **kwargs):
        """
        Generate uncertainty plots for each task.

        Parameters
        ----------
        y_true : array-like
            Ground truth values.
        y_pred : array-like
            Predicted mean values.
        y_std : array-like
            Predicted standard deviations.
        target_labels : list of str, optional
            List of target labels for each task.
        **kwargs
            Additional keyword arguments.

        Returns
        -------
        dict
            Dictionary of generated plot figures.

        Raises
        ------
        ValueError
            If required inputs are missing or shapes are inconsistent.

        """
        # Check inputs
        if y_true is None or y_pred is None or y_std is None:
            raise ValueError("Must provide `y_true`, `y_pred`, and `y_std`")

        # Convert to numpy array if needed
        if isinstance(y_true, (pd.Series, pd.DataFrame)):
            y_true = y_true.to_numpy()

        # Ensure 2D arrays for consistency
        if y_pred.ndim == 1:
            y_pred = y_pred.reshape(-1, 1)
        if y_true.ndim == 1:
            y_true = y_true.reshape(-1, 1)
        if y_std.ndim == 1:
            y_std = y_std.reshape(-1, 1)

        # Verify number of tasks
        n_tasks = y_true.shape[1]
        if (n_tasks != y_pred.shape[1]) or (n_tasks != y_std.shape[1]):
            raise ValueError(
                "`y_true`, `y_pred`, and `y_std` must have the same number of tasks"
            )

        # Construct target labels if not provided
        if target_labels is None:
            target_labels = [f"task_{i}" for i in range(n_tasks)]

        # Enumerate targets
        for task_id, task_label in enumerate(target_labels):
            # Task target values
            t_true = y_true[:, task_id].flatten()
            t_pred = y_pred[:, task_id].flatten()
            t_std = y_std[:, task_id].flatten()

            # Mask nans
            t_true, t_pred, t_std = mask_nans_std(t_true, t_pred, t_std)

            # Enumerate plots
            for plot_tag, plot in self._plots.items():
                self._plot_data[f"{task_label}_{plot_tag}"] = plot(
                    t_true,
                    t_pred,
                    t_std,
                    title=f"Uncertainty Calibration\nTask {task_label}",
                    dpi=self.dpi,
                )

        return self._plot_data

    @staticmethod
    def calibration_plot(y_true, y_pred, y_std, title="", dpi=300):
        """
        Create a calibration plot for uncertainty estimates.

        Parameters
        ----------
        y_true : array-like
            Ground truth values.
        y_pred : array-like
            Predicted mean values.
        y_std : array-like
            Predicted standard deviations.
        title : str, default=""
            Title for the plot.
        dpi : int, default=300
            DPI for the plot.

        Returns
        -------
        matplotlib.figure.Figure
            The generated calibration plot figure.

        """
        # Plot calibration
        fig, ax = plt.subplots(dpi=dpi)
        ax = uct.viz.plot_calibration(
            y_pred,
            y_std,
            y_true,
            ax=ax,
        )

        # Change dashed line color
        ax.get_lines()[0].set_color("black")

        # Set title
        ax.set_title(title)

        return fig

    def report(self, write=False, output_dir=None):
        """
        Report the generated plots.

        Parameters
        ----------
        write : bool, default=False
            Whether to write the plots to disk.
        output_dir : Path or str, optional
            Directory to write the plots to.

        Returns
        -------
        dict
            Dictionary of generated plot figures.

        """
        if write:
            self.write_report(output_dir)

        return self._plot_data

    def write_report(self, output_dir):
        """
        Write the generated plots to disk and optionally log to wandb.

        Parameters
        ----------
        output_dir : Path or str
            Directory to write the plots to.

        """
        for plot_tag, plot in self._plot_data.items():
            plot_path = output_dir / f"{plot_tag}.png"
            plot.savefig(plot_path, dpi=self.dpi)
            if self.use_wandb:
                wandb.log({plot_tag: wandb.Image(str(plot_path))})
