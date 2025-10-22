"""PostHoc multi-model comparison implementation."""

import os
import glob
import yaml
import boto3
from urllib.parse import urlparse
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import json
from pingouin import plot_paired
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)
from scipy import stats
from scipy.stats import levene, tukey_hsd, ttest_rel
from statsmodels.stats.anova import AnovaRM
from itertools import combinations
import tabulate
import warnings
from loguru import logger

from openadmet.models.comparison.compare_base import ComparisonBase, comparisons


def _download_s3_dir(s3_uri, local_dir):
    """Download all files from an S3 directory to a local directory."""
    s3 = boto3.client("s3")
    parsed = urlparse(s3_uri)
    bucket = parsed.netloc
    prefix = parsed.path.lstrip("/")
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            rel_path = os.path.relpath(key, prefix)
            local_path = os.path.join(local_dir, rel_path)
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            s3.download_file(bucket, key, local_path)


@comparisons.register("PostHoc")
class PostHocComparison(ComparisonBase):
    """
    PostHoc multi-model comparison.

    Attributes
    ----------
    _metrics_names : list
        List of metrics to compare.
    _direction_dict : dict
        Dictionary indicating whether to minimize or maximize each metric.
    _sig_levels : list
        List of significance levels for statistical tests.
    _confidence_level : float
        Confidence level for statistical tests.
    _stats_names : list
        List of statistical tests to perform.

    """

    _metrics_names: list = ["mse", "mae", "r2", "ktau", "spearmanr"]

    _direction_dict: dict = {
        "mae": "minimize",
        "mse": "minimize",
        "r2": "maximize",
        "ktau": "maximize",
        "spearmanr": "maximize",
    }

    _sig_levels: list = [0.05, 0.01, 0.001]

    _confidence_level: float = 0.95

    _stats_names: list[str] = ["Levene", "Tukey_HSD"]

    @property
    def metrics(self):
        """Get metrics."""
        return self._metrics_names

    @property
    def direction_dict(self):
        """Get direction dictionary."""
        return self._direction_dict

    @property
    def sig_levels(self):
        """Get significance levels."""
        return self._sig_levels

    @property
    def cl(self):
        """Get confidence level."""
        return self._confidence_level

    @property
    def stats_names(self):
        """Get statistics names."""
        return self._stats_names

    def safe_dirs(self, dirs):
        """Ensure dirs is a list and contains only valid paths."""
        if not isinstance(dirs, list):
            dirs = [dirs]
        clean_dirs = []
        for dir in dirs:
            # If dir is a tuple, take the first element
            if isinstance(dir, tuple):
                dir = dir[0]
            if not isinstance(dir, (str, os.PathLike)):
                raise ValueError(f"Directory {dir} is not a valid path")
            if not os.path.exists(dir):
                raise ValueError(f"Directory {dir} does not exist")
            clean_dirs.append(dir)
        return clean_dirs

    def compare(
        self,
        model_dirs=None,
        label_types: list = None,
        model_stats_fns: list = None,
        labels: list = None,
        task_names: list = None,
        mt_id: str = None,
        report: bool = False,
        output_dir: bool = None,
    ):
        """
        Compare models using post-hoc statistical tests and generate plots and reports.

        Required arguments are either (model_dirs and label_types) OR (model_stats_fns, labels, and task_names).
        If the user has the full training directories for the models, it is recommended to use the model_dirs and label_types
        option. If the user only has the JSON files with model statistics, then use the model_stats_fns, labels, and task_names option.

        Parameters
        ----------
        model_dirs : list, optional
            Path to the main training directory containing model subdirectories with
            `anvil_recipe.yaml` and `cross_validation_metrics.json` files.
        label_types : list of str, optional
            List of categories from the `anvil_recipe.yaml` file to use for labeling each model.
            Supported values are 'biotarget', 'model', 'feat', and 'tasks'.
        model_stats_fns : list of str, optional
            List of file paths to JSON files containing model statistics.
        labels : list of str, optional
            List of tags for the models, used for plotting and reporting.
        task_names : list of str, optional
            List of task names as they appear in the model statistics JSON files.
        mt_id : str, optional
            Identifier for the target column when comparing multitask models. Used to select
            the appropriate task from the `anvil_recipe.yaml` file.  Must be a unique string
            not appearing in any target columns for other models in the file. Required if comparing
            multitask models.
        report : bool, optional
            Whether to generate a PDF report of the comparison results. Default is False.
        output_dir : str, optional
            Directory to save the output plots and report. Default is None.

        """
        if model_dirs:
            model_dirs = self.safe_dirs(dirs=model_dirs)

        # Download from S3 if needed
        local_model_dirs = []
        for model_dir in model_dirs or []:
            if model_dir.startswith("s3://"):
                local_dir = f"/tmp/{os.path.basename(model_dir.rstrip('/'))}"
                _download_s3_dir(model_dir, local_dir)
                local_model_dirs.append(local_dir)
            else:
                local_model_dirs.append(model_dir)
        model_dirs = local_model_dirs

        if not (
            (model_dirs is not None and label_types is not None)
            or (
                model_stats_fns is not None
                and labels is not None
                and task_names is not None
            )
        ):
            raise ValueError(
                "You must provide either (model_dir and label_types) OR (model_stats_fns, labels, and task_names)."
            )

        if not model_stats_fns:
            model_stats_fns, labels, task_names = self.label_and_task_name_from_anvil(
                model_dirs, label_types, mt_id=mt_id
            )

        if len(set(labels)) != len(labels):
            raise ValueError("Labels must be unique")

        df = self.json_to_df(model_stats_fns, labels, task_names)

        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)

        stats_dfs = []
        stats_dfs.append(self.levene_test(df, labels))
        stats_dfs.append(self.get_tukeys_df(df, labels))
        if output_dir:
            self.stats_to_json(stats_dfs, output_dir=output_dir)

        plot_data = {}
        plot_data["normality"] = self.normality_plots(df, output_dir)
        plot_data["anova"] = self.anova(df, labels, output_dir)
        plot_data["mcs"] = self.mcs_plots(df, labels, output_dir)
        plot_data["mean_diff"] = self.mean_diff_plots(df, labels, self.cl, output_dir)
        plot_data["paired"] = self.paired_plots(df, labels, output_dir)

        self.print_table(stats_dfs[0], stats_dfs[1])
        self.report(stats_dfs, report, output_dir)

        return stats_dfs

    def label_and_task_name_from_anvil(
        self, model_dirs: list, label_types: list[str], mt_id: str = None
    ):
        """
        Extract model statistics file paths, labels, and task names from an Anvil training directory.

        Parameters
        ----------
        model_dirs : str
            Path to the main training directory containing model subdirectories with
            `anvil_recipe.yaml` and `cross_validation_metrics.json` files.
        label_types : list of str
            List of categories from the `anvil_recipe.yaml` file to use for labeling each model.
            Supported values are 'biotarget', 'model', 'feat', and 'tasks'.
        mt_id : str, optional
            Identifier for the target column when comparing multitask models. Used to select
            the appropriate task from the `anvil_recipe.yaml` file.  Must be a unique string
            not appearing in any target columns for other models in the file. Required if comparing
            multitask models.

        Returns
        -------
        model_stats_fns : list of str
            List of file paths to JSON files containing model statistics.
        labels : list of str
            List of tags for the models, used for plotting and reporting.
        task_names : list of str
            List of task names as they appear in the model statistics JSON files.

        """
        all_labels = []
        all_task_names = []

        if isinstance(label_types, str):
            label_types = [label_types]
        elif isinstance(label_types, tuple):
            label_types = list(label_types)
        elif not isinstance(label_types, list):
            raise ValueError("label_types must be lists")

        model_dirs = self.safe_dirs(dirs=model_dirs)

        for model_dir in model_dirs:
            # find all directories containing an anvil_recipe.yaml and cross_validation_metrics.json within model_dir
            logger.info(f"Searching for models in {model_dir}...")

            anvil_recipes = [
                os.path.dirname(i)
                for i in glob.glob(f"{model_dir}/**/anvil_recipe.yaml", recursive=True)
            ]
            cv_metrics = [
                os.path.dirname(i)
                for i in glob.glob(
                    f"{model_dir}/**/cross_validation_metrics.json", recursive=True
                )
            ]
            model_dirs = list(set(anvil_recipes).intersection(set(cv_metrics)))
            print(f"Found {len(model_dirs)} models in {model_dir}")

            model_stats_fns = [
                f"{model_dir}/cross_validation_metrics.json" for model_dir in model_dirs
            ]
            logger.info(
                f"Found {len(model_stats_fns)} cross_validation_metrics.json and anvil_recipe.yaml files"
            )

            for model_dir in model_dirs:
                with open(f"{model_dir}/anvil_recipe.yaml") as f:
                    anvil = yaml.safe_load(f)

                full_label = []
                target_cols = anvil["data"]["target_cols"]
                if type(target_cols) is str:
                    target_cols = [target_cols]

                # NOTE: this logic assumes that if multitask, the tasks will have
                # different biotargets and that # biotargets == # tasks
                if len(target_cols) > 1 and mt_id:
                    col_match = []
                    ind_for_biotarget = 0
                    for ind, col in enumerate(target_cols):
                        if mt_id.lower() in col.lower():
                            col_match.append(col)
                            ind_for_biotarget = (
                                ind.copy()
                            )  # this is to get the index for the list of biotargets

                    # check that the multitask id provided by the user does not
                    # appear in multiple target columns
                    if len(col_match) == 1:
                        all_task_names.append(col_match[0])
                    elif len(col_match) == 0:
                        raise ValueError(
                            f"Target {mt_id} not found in target columns {target_cols}"
                        )
                    else:
                        raise ValueError(
                            f"Target {mt_id} found multiple times in target columns {target_cols}, please be more specific"
                        )

                # single-task case
                else:
                    all_task_names.append(target_cols[0])
                    ind_for_biotarget = (
                        0  # if single task, there will be only one biotarget
                    )

                for lab in label_types:
                    if lab == "biotarget":
                        full_label.append(
                            anvil["metadata"]["biotargets"][ind_for_biotarget]
                        )

                    # sets model label based on the class names of the model, as specified in anvil recipe
                    elif lab == "model":
                        to_remove = [
                            "Regressor",
                            "Classifier",
                            "Model",
                            "Module",
                            "Lightning",
                        ]
                        label = anvil["procedure"]["model"]["type"]
                        for r in to_remove:
                            label = label.replace(r, "")
                        # chemeleon special case
                        if label == "ChemProp":
                            if (
                                anvil["procedure"]["model"]["params"]["from_chemeleon"]
                                == True
                            ):
                                label = "Chemeleon"
                        full_label.append(label)

                    elif lab == "feat":
                        to_remove = ["Featurizer"]
                        label = anvil["procedure"]["feat"]["type"]
                        if label == "DescriptorFeaturizer":
                            label = anvil["procedure"]["feat"]["params"]["descr_type"]
                        if label == "FingerprintFeaturizer":
                            label = anvil["procedure"]["feat"]["params"]["fp_type"]
                        if label == "FeatureConcatenator":
                            label = ""
                            for ind, f in enumerate(
                                anvil["procedure"]["feat"]["params"]["featurizers"]
                            ):
                                if f == "DescriptorFeaturizer":
                                    label += anvil["procedure"]["feat"]["params"][
                                        "featurizers"
                                    ]["DescriptorFeaturizer"]["descr_type"]
                                if f == "FingerprintFeaturizer":
                                    label += anvil["procedure"]["feat"]["params"][
                                        "featurizers"
                                    ]["FingerprintFeaturizer"]["fp_type"]
                                if (
                                    ind
                                    < len(
                                        anvil["procedure"]["feat"]["params"][
                                            "featurizers"
                                        ]
                                    )
                                    - 1
                                ):
                                    label += "+"
                        for r in to_remove:
                            label = label.replace(r, "")
                        full_label.append(label)

                    elif lab == "tasks":
                        num_tasks = len(target_cols)
                        if num_tasks > 1:
                            full_label.append("MT")
                        else:
                            full_label.append("ST")

                    else:
                        print("here")
                        raise ValueError(
                            f"Label type {lab} not recognized, must be one of ['biotarget', 'model', 'feat', 'tasks']"
                        )

                all_labels.append("_".join(full_label))

        return (model_stats_fns, all_labels, all_task_names)

    def json_to_df(
        self, model_stats_fns: list[str], labels: list[str], task_names: list[str]
    ):
        """
        Load and aggregate model statistics from cross-validation JSON files into a single DataFrame.

        Each JSON file should contain metrics for a specific model and task. This function extracts
        the specified metrics for each model and task, and combines them into a DataFrame suitable
        for statistical comparison and plotting.

        Parameters
        ----------
        model_stats_fns : list of str
            List of file paths to JSON files with model statistics.
        labels : list of str
            List of tags for the models, used for plotting and reporting.
        task_names : list of str
            List of task names as they appear in the model statistics JSON files.

        Returns
        -------
        df : pandas.DataFrame
            DataFrame containing the extracted statistics for each model and task, with columns for
            each metric and a 'method' column indicating the model tag.

        Raises
        ------
        ValueError
            If a specified task or metric is not found in the JSON data.

        """
        df = pd.DataFrame()
        for model, tag, task in zip(model_stats_fns, labels, task_names):
            with open(model) as f:
                data = json.load(f)
            method_data = pd.DataFrame()
            for m in self.metrics:
                if task not in data:
                    raise ValueError(f"Task {task} not found in data.")
                if m not in data[task]:
                    raise ValueError(f"Metric {m} not found in task {task} data.")
                values = data[task][m]["value"]
                if np.isnan(values).any():
                    if m == "spearmanr":
                        method_data[m] = pd.Series(values).fillna(-1.0)
                    else:
                        method_data[m] = pd.Series(values).fillna(0)
                else:
                    method_data[m] = values
            method_data["method"] = tag
            df = pd.concat([df, method_data])
            print(
                "Reading in model: " + method_data["method"].values[0],
                method_data.shape,
            )
        return df

    def levene_test(self, df: pd.DataFrame, labels: list[str]):
        """
        Perform Levene's test across models.

        Parameters
        ----------
        df : pandas.DataFrame
            DataFrame containing the extracted statistics for each model.
        labels : list of str
            List of tags for the models, used to group data for the test.

        Returns
        -------
        result : pandas.DataFrame
            DataFrame with Levene's test statistic and p-value for each metric.

        """
        result = pd.DataFrame()
        lev_vecs = [df[df["method"] == tag] for tag in labels]
        for m in self.metrics:
            lev = levene(*[vec[m] for vec in lev_vecs])
            result[m] = {"stat": lev.statistic, "pvalue": lev.pvalue}
        return result

    def normality_plots(self, df: pd.DataFrame, output_dir: str = None):
        """
        Generate normality plots for each metric in the DataFrame.

        Parameters
        ----------
        df : pandas.DataFrame
            DataFrame containing the extracted statistics for each model.
        output_dir : str, optional
            Directory to save the plots. If None, plots are not saved.

        Returns
        -------
        fig : matplotlib.figure.Figure
            Figure object containing the normality plots.

        """
        fig, axes = plt.subplots(2, len(self.metrics), figsize=(20, 10))

        for i, metric in enumerate(self.metrics):
            ax = axes[0, i]
            sns.histplot(df[metric], kde=True, ax=ax)
            ax.set_title(f"{metric}", fontsize=16)

        for i, metric in enumerate(self.metrics):
            ax = axes[1, i]
            stats.probplot(df[metric], dist="norm", plot=ax)
            ax.set_title("")

        plt.tight_layout()

        if output_dir:
            plt.savefig(f"{output_dir}/normality_plots.pdf")

        return fig

    def anova(self, df: pd.DataFrame, labels: list[str], output_dir: str = None):
        """
        Perform repeated measures ANOVA for each metric and plot means with error bars.

        Parameters
        ----------
        df : pandas.DataFrame
            Balanced DataFrame containing the extracted statistics for each model.
        labels : list of str
            List of tags for the models, used for grouping and labeling.
        output_dir : str, optional
            Directory to save the ANOVA plots. If None, plots are not saved.

        Returns
        -------
        fig : matplotlib.figure.Figure
            Figure object containing the ANOVA plots for each metric.

        """
        # Assume df is already balanced: each method has the same number of cv_cycles per metric
        fig, axes = plt.subplots(
            len(self.metrics),
            1,
            sharex=False,
            sharey=False,
            figsize=(8, 4 * len(self.metrics)),
        )
        if len(self.metrics) == 1:
            axes = [axes]
        for i, metric in enumerate(self.metrics):
            anova_df = df[["method", metric]].copy()
            anova_df["cv_cycle"] = anova_df.groupby("method").cumcount()

            # Run repeated measures ANOVA
            model = AnovaRM(
                anova_df, depvar=metric, subject="cv_cycle", within=["method"]
            ).fit()

            # Calculate means and standard errors for error bars
            means = anova_df.groupby("method")[metric].mean()
            ses = anova_df.groupby("method")[metric].sem()
            ax = axes[i]

            ax.set_xlim(left=0, right=None)

            # Get Tukey HSD results for determining significance
            hsd_df = self.get_tukeys_df(df, labels)

            # Plot means with error bars, colored by Tukey HSD results
            # Find the best model (lowest mean if minimize, highest if maximize)
            direction = self.direction_dict.get(metric, "minimize")
            if direction == "minimize":
                best_idx = means.argmin()
            else:
                best_idx = means.argmax()
            best_method = means.index[best_idx]

            # Get Tukey HSD results for this metric
            tukey_metric_df = hsd_df[hsd_df["metric_name"] == metric]

            # Determine colors for the bars based on Tukey HSD results
            bar_colors = []
            for method in means.index:
                color = "grey"
                if method == best_method:
                    color = "blue"
                else:
                    mask1 = tukey_metric_df["method"] == f"{best_method} - {method}"
                    mask2 = tukey_metric_df["method"] == f"{method} - {best_method}"
                    pvals = tukey_metric_df[mask1 | mask2]["pvalue"]
                    if not pvals.empty and (pvals <= 0.05).any():
                        color = "red"
                bar_colors.append(color)

            # Plot means with error bars
            for j, (mean, se, color) in enumerate(
                zip(means.values, ses.values, bar_colors)
            ):
                ax.errorbar(
                    x=mean,
                    y=j,
                    xerr=se,
                    fmt="o",
                    capsize=0,
                    color=color,
                    ecolor=color,
                    elinewidth=2,
                )
                if means.index[j] == best_method:
                    ax.axvline(mean - se, color="grey", linestyle="--", linewidth=1)
                    ax.axvline(mean + se, color="grey", linestyle="--", linewidth=1)
            ax.set_yticks(np.arange(len(means)))
            ax.set_yticklabels(means.index)
            ax.set_title(f"p={model.anova_table['Pr > F'].iloc[0]}")
            ax.set_xlabel(metric)
            ax.set_ylabel("method")
        plt.tight_layout()

        if output_dir:
            plt.savefig(f"{output_dir}/anova.pdf")

        return fig

    @staticmethod
    def tukey_hsd_by_metric(df: pd.DataFrame, metric: str, labels: str):
        """
        Perform Tukey's HSD test for a specific metric across multiple models.

        Parameters
        ----------
        df : pandas.DataFrame
            DataFrame containing the extracted statistics for each model.
        metric : str
            The metric for which to perform Tukey's HSD test.
        labels : list of str
            List of tags for the models, used to group data for the test.

        Returns
        -------
        hsd : TukeyHSDResults
            Results of Tukey's HSD test, including statistics and p-values.

        """
        return tukey_hsd(*[np.array(df[df["method"] == tag][metric]) for tag in labels])

    def get_tukeys_df(self, df: pd.DataFrame, labels: list[str], cl: float = 0.95):
        """
        Generate a DataFrame with Tukey's HSD results for multiple metrics.

        Parameters
        ----------
        df : pandas.DataFrame
            DataFrame containing the extracted statistics for each model.
        labels : list of str
            List of tags for the models, used to group data for the test.
        cl : float, optional
            Confidence level for the Tukey's HSD test. Default is 0.95.

        Returns
        -------
        hsd_df : pandas.DataFrame
            DataFrame containing the results of Tukey's HSD test, including method comparisons,
            metric names, statistics, error bars, and p-values.

        """
        tukeys = [
            self.tukey_hsd_by_metric(df, metric, labels) for metric in self.metrics
        ]
        method_compare = []
        stats = []
        errorbars = []
        metric = []
        pvalue = []
        for metric_ind, hsd in enumerate(tukeys):
            for i in range(len(hsd.statistic) - 1):
                for j in range(i + 1, len(hsd.statistic)):
                    s = hsd.statistic[i, j]
                    method_compare.append(f"{labels[i]} - {labels[j]}")
                    stats.append(s)
                    errorbars.append(
                        [
                            s - hsd.confidence_interval(confidence_level=cl).low[i, j],
                            hsd.confidence_interval(confidence_level=cl).high[i, j] - s,
                        ]
                    )
                    metric.append(self.metrics[metric_ind])
                    pvalue.append(hsd.pvalue[i, j])
        hsd_df = pd.DataFrame(
            {
                "method": method_compare,
                "metric_name": metric,
                "metric_val": stats,
                "errorbars": np.array(errorbars)[:, 0],
                "pvalue": pvalue,
            }
        )
        return hsd_df

    def mcs_plots(self, df: pd.DataFrame, labels: list[str], output_dir: str = None):
        """
        Generate and save multiple comparison of means (MCS) plots for each metric.

        Parameters
        ----------
        df : pandas.DataFrame
            DataFrame containing the extracted statistics for each model.
        labels : list of str
            List of tags for the models, used for grouping and labeling.
        output_dir : str, optional
            Directory to save the MCS plots. If None, plots are not saved.

        Returns
        -------
        fig : matplotlib.figure.Figure
            Figure object containing the MCS plots for each metric.

        """
        figsize = (20, 10)
        nrow = -(-len(self.metrics) // 3)
        fig, ax = plt.subplots(nrow, 3, figsize=figsize)

        for i, metric in enumerate(self.metrics):
            metric = metric.lower()

            row = i // 3
            col = i % 3

            reverse_cmap = False
            if self.direction_dict[metric] == "minimize":
                reverse_cmap = True

            hsd = self.tukey_hsd_by_metric(df, metric, labels)

            cmap = "coolwarm"
            if reverse_cmap:
                cmap = cmap + "_r"

            significance = pd.DataFrame(hsd.pvalue)
            significance[(hsd.pvalue < self.sig_levels[2]) & (hsd.pvalue >= 0)] = "***"
            significance[
                (hsd.pvalue < self.sig_levels[1]) & (hsd.pvalue >= self.sig_levels[2])
            ] = "**"
            significance[
                (hsd.pvalue < self.sig_levels[0]) & (hsd.pvalue >= self.sig_levels[1])
            ] = "*"
            significance[(hsd.pvalue >= self.sig_levels[0])] = ""

            # Create a DataFrame for the annotations
            annotations = (
                pd.DataFrame(hsd.statistic).round(3).astype(str) + significance
            )

            hax = sns.heatmap(
                pd.DataFrame(hsd.statistic),
                cmap=cmap,
                annot=annotations,
                fmt="",
                ax=ax[row, col],
                vmin=None,
                vmax=None,
            )

            x_label_list = [x for x in labels]
            y_label_list = [x for x in labels]
            hax.set_xticklabels(
                x_label_list, ha="center", va="top", rotation=0, rotation_mode="anchor"
            )
            hax.set_yticklabels(
                y_label_list,
                ha="center",
                va="center",
                rotation=90,
                rotation_mode="anchor",
            )

            hax.set_xlabel("")
            hax.set_ylabel("")
            hax.set_title(metric.upper())

        # If there are less plots than cells in the grid, hide the remaining cells
        if (len(self.metrics) % 3) != 0:
            for i in range(len(self.metrics), nrow * 3):
                row = i // 3
                col = i % 3
                ax[row, col].set_visible(False)

        plt.tight_layout()

        if output_dir:
            plt.savefig(f"{output_dir}/mcs_plots.pdf")

        return fig

    def mean_diff_plots(
        self,
        df: pd.DataFrame,
        labels: list[str],
        cl: float = None,
        output_dir: str = None,
    ):
        """
        Generate and save mean difference plots with error bars for each metric.

        Parameters
        ----------
        df : pandas.DataFrame
            DataFrame containing the extracted statistics for each model.
        labels : list of str
            List of tags for the models, used for grouping and labeling.
        cl : float, optional
            Confidence level for the error bars. Default is 0.95.
        output_dir : str, optional
            Directory to save the mean difference plots. If None, plots are not saved.

        Returns
        -------
        fig : matplotlib.figure.Figure
            Figure object containing the mean difference plots for each metric.

        """
        fig, axes = plt.subplots(
            len(self.metrics), 1, figsize=(8, 2 * len(self.metrics)), sharex=False
        )
        ax_ind = 0
        if not cl:
            cl = 0.95
        tukeys_df = self.get_tukeys_df(df, labels, cl=cl)

        for metric in self.metrics:
            tukey_metric_df = tukeys_df[tukeys_df["metric_name"] == metric]
            errorbars = [i for i in np.transpose(tukey_metric_df["errorbars"])]
            to_plot_df = pd.DataFrame(
                {
                    "method": tukey_metric_df["method"],
                    "metric_val": tukey_metric_df["metric_val"],
                }
            )
            ax = axes[ax_ind]
            ax.errorbar(
                data=to_plot_df,
                x="metric_val",
                y="method",
                xerr=errorbars,
                fmt="o",
                capsize=5,
            )
            ax.axvline(0, ls="--", lw=3)
            ax.set_title(metric)
            ax.set_xlabel("Mean Difference")
            ax.set_ylabel("")
            ax.set_xlim(-0.2, 0.2)
            ax_ind += 1

        fig.suptitle("Multiple Comparison of Means\nTukey HSD, FWER=0.05")
        plt.tight_layout()

        if output_dir:
            plt.savefig(f"{output_dir}/mean_diffs.pdf")

        return fig

    def paired_plots(self, df: pd.DataFrame, labels: list[str], output_dir: str = None):
        """
        Generate and save paired plots comparing all pairs of methods for 'mse' as subplots in a single PDF.

        Parameters
        ----------
        df : pandas.DataFrame
            DataFrame containing the extracted statistics for each model.
        labels : list of str
            List of tags for the models, used for grouping and labeling.
        output_dir : str, optional
            Directory to save the paired plots. If None, plots are not saved.

        Returns
        -------
        fig : matplotlib.figure.Figure
            Figure object containing all paired plots as subplots.

        """
        import warnings
        from itertools import product

        pairs = list(combinations(labels, 2))
        metrics = self.metrics
        nrows = len(pairs)
        ncols = len(metrics)

        fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4 * nrows))
        axes = axes.flatten() if nrows * ncols > 1 else [axes]

        # Tukey HSD DataFrame for significance
        hsd_df = self.get_tukeys_df(df, labels)

        plot_idx = 0
        for metric in metrics:
            tukey_metric_df = hsd_df[hsd_df["metric_name"] == metric]

            for method1, method2 in pairs:
                method_list = [method1, method2]
                tmp_df = df[df["method"].isin(method_list)].copy()
                tmp_df = tmp_df.reset_index(drop=True)
                tmp_df["cycle"] = tmp_df.groupby("method").cumcount()
                title = f"Paired: {method1} - {method2} ({metric})"

                # Determine colors for the bars based on Tukey HSD results
                mask1 = tukey_metric_df["method"] == f"{method1} - {method2}"
                mask2 = tukey_metric_df["method"] == f"{method2} - {method1}"
                pvals = tukey_metric_df[mask1 | mask2]["pvalue"]
                if not pvals.empty and (pvals > 0.05).any():
                    title_color = "black"
                else:
                    title_color = "red"
                    if np.mean(tmp_df[tmp_df["method"] == method1][metric]) > np.mean(
                        tmp_df[tmp_df["method"] == method2][metric]
                    ):
                        title = "<-  " + title
                    else:
                        title = title + "  ->"

                ax = axes[plot_idx]
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=FutureWarning)
                    plot_paired(
                        data=tmp_df, dv=metric, within="method", subject="cycle", ax=ax
                    )
                ax.set_title(title, color=title_color)
                ax.set_xlabel("Method")
                ax.set_ylabel(metric.upper())
                plot_idx += 1

        # Hide any unused subplots
        for j in range(plot_idx, len(axes)):
            fig.delaxes(axes[j])

        plt.tight_layout()

        if output_dir:
            plt.savefig(f"{output_dir}/paired_plots.pdf")

        return fig

    def stats_to_json(self, stats_dfs: list[pd.DataFrame], output_dir: str):
        """
        Save statistical test results to JSON files.

        Parameters
        ----------
        stats_dfs : list of pandas.DataFrame
            List of DataFrames containing statistical test results (e.g., Levene, Tukey HSD).
        output_dir : str
            Directory to save the JSON files.

        """
        for stat_df, name in zip(stats_dfs, self.stats_names):
            stat_df.to_json(f"{output_dir}/{name}.json")

    def convert_float_round(self, val: float):
        """
        Convert a float to scientific notation rounded to 3 decimal places.

        If conversion fails, return the original value.

        Parameters
        ----------
        ----------
        val : float
            The value to convert.

        Returns
        -------
        str
            The converted value as a string in scientific notation, or the original value if conversion fails.

        """
        try:
            return str(f"{float(val):0.3e}")
        except ValueError:
            return val

    def report(
        self, data_dfs: list[pd.DataFrame], write: bool = False, output_dir: str = None
    ):
        """
        Generate and optionally save a report summarizing the statistical analysis.

        Parameters
        ----------
        data_dfs : list of pandas.DataFrame
            List of DataFrames containing statistical test results (e.g., Levene, Tukey HSD).
        write : bool, optional
            If True, writes the report to a PDF file. Default is False.
        output_dir : str, optional
            Directory to save the report. Required if write is True.

        """
        if write:
            self.write_report(data_dfs, output_dir)

    def write_report(self, data_dfs: list[pd.DataFrame], output_dir: str):
        """
        Generate and save a PDF report summarizing the statistical analysis.

        Parameters
        ----------
        data_dfs : list of pandas.DataFrame
            List of DataFrames containing statistical test results (e.g., Levene, Tukey HSD).
        output_dir : str
            Directory to save the PDF report.

        """
        doc = SimpleDocTemplate(
            f"{output_dir}/posthoc.pdf",
            pagesize=letter,
        )
        elements = []
        styles = getSampleStyleSheet()
        styleH = styles["Heading1"]
        style = TableStyle(
            [
                ("FONT", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, 0), 9),
                ("ALIGN", (0, 0), (-1, 0), "CENTER"),
                ("ALIGN", (0, 0), (0, -1), "LEFT"),
                ("INNERGRID", (0, 0), (-1, -1), 0.50, colors.black),
                ("BOX", (0, 0), (-1, -1), 0.25, colors.black),
            ]
        )

        # Levene table
        lev_df = data_dfs[0]
        elements.append(Paragraph("Levene", styleH))
        elements.append(Spacer(1, 0.25 * inch))
        data = [lev_df.columns.to_list()] + lev_df.values.tolist()
        data = [[self.convert_float_round(val) for val in row] for row in data]
        data[0].insert(0, "value")
        data[1].insert(0, "statistic")
        data[2].insert(0, "p-value")
        table = Table(data, hAlign="LEFT")
        table.setStyle(style)
        elements.append(table)
        elements.append(Spacer(1, 0.2 * inch))

        # Tukey HSD tables
        tukey_df = data_dfs[1]
        elements.append(Paragraph("Tukey HSD", styleH))
        elements.append(Spacer(1, 0.25 * inch))

        for m in self.metrics:
            tukey_metric_df = tukey_df[tukey_df["metric_name"] == m]
            errorbars = tukey_metric_df["errorbars"].values.tolist()
            metric_val = tukey_metric_df["metric_val"].values.tolist()
            tukey_metric_df.insert(
                4, "coeff of var", [i / j for i, j in zip(errorbars, metric_val)]
            )
            data = [tukey_metric_df.columns.to_list()] + tukey_metric_df.values.tolist()
            data = [[self.convert_float_round(val) for val in row] for row in data]
            table = Table(data, hAlign="LEFT")
            table.setStyle(style)
            elements.append(table)
            elements.append(Spacer(1, 0.2 * inch))

        doc.build(elements)

    def print_table(self, levene_df: pd.DataFrame, tukeys_df: pd.DataFrame):
        """
        Print a DataFrame as a table.

        Parameters
        ----------
        levene_df : pandas.DataFrame
            DataFrame containing Levene's test statistics for each metric.
        tukeys_df : pandas.DataFrame
            DataFrame containing Tukey's HSD test results, including method comparisons,
            metrics, values, error bars, and p-values.

        """
        print("Levene's test results")
        print("-------------------------")
        print(
            tabulate.tabulate(
                levene_df, headers=self._metrics_names, tablefmt="psql", showindex=False
            )
        )
        print("\nTukey's HSD results")
        print("-------------------------")
        print(
            tabulate.tabulate(
                tukeys_df,
                headers=["method", "metric", "value", "errorbars", "p-value"],
                tablefmt="psql",
                showindex=False,
            )
        )
