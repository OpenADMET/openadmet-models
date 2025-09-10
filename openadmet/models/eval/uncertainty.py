import matplotlib.pyplot as plt
import pandas as pd
import uncertainty_toolbox as uct
from pydantic import Field

from openadmet.models.eval.eval_base import EvalBase, evaluators


@evaluators.register("UncertaintyPlots")
class UncertaintyPlots(EvalBase):
    use_wandb: bool = Field(False, description="Whether to use wandb")
    dpi: int = Field(300, description="DPI for the plot")
    _plots = {}
    _plot_data = {}

    def evaluate(self, y_true, y_pred, y_std, target_labels=None):
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

        # Specify plots
        self._plots = {
            "calibration": self.plot_calibration,
        }

        # Enumerate targets
        for task_id, task_label in enumerate(target_labels):
            # Enumerate plots
            for plot_tag, plot in self._plots.items():
                self.plot_data[f"{task_label}_{plot_tag}"] = plot(
                    y_true[:, task_id],
                    y_pred[:, task_id],
                    y_std[:, task_id],
                    title=f"Uncertainty Calibration\nTask {task_label}",
                    dpi=self.dpi,
                )

        return self._plot_data

    @staticmethod
    def plot_calibration(y_true, y_pred, y_std, title="", dpi=300):
        # Plot calibration
        fig, ax = plt.subplots(dpi=dpi)
        ax = uct.viz.plot_calibration(
            y_pred,
            y_std,
            y_true,
            ax=ax,
        )

        # Set title
        ax.set_title(title)

        return fig

    def report(self):
        pass
