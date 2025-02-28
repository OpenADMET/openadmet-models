import json
import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from reportlab.pdfgen.canvas import Canvas
from scipy import stats
from scipy.stats import f_oneway, levene, tukey_hsd

import openadmet_models
from openadmet_models.comparison.compare_base import ComparisonBase, comparisons


@comparisons.register("PostHoc")
class PostHocComparison(ComparisonBase):

    _metrics_names: list = ["mse", "mae", "r2", "ktau", "spearmanr"]

    @property
    def metrics(self):
        return self._metrics_names

    def compare(self, model_stats_fns, model_tags, save_dir=None):
        df = self.json_to_df(model_stats_fns, model_tags)
        levene = self.levene_test(df, model_tags)
        tukeys = [
            self.tukey_hsd_by_metric(df, metric, model_tags) for metric in self.metrics
        ]
        tukeys_df = self.tukeys_to_df(tukeys)
        self.to_json(levene, tukeys_df)
        self.normality_plots(df, save_dir)
        self.anova(df, model_tags, save_dir)
        self.mcs_plots(df, model_tags, save_dir)
        self.mean_diff_plots(df, model_tags, save_dir)

    def json_to_df(self, model_stats_fns, model_tags):
        df = pd.DataFrame()
        for model, tag in zip(model_stats_fns, model_tags):
            data = pd.read_json(model)
            method_data = pd.DataFrame()
            for m in self.metrics:
                values = data[m].value
                method_data[m] = values
            method_data["method"] = tag
            df = pd.concat([df, method_data])
        return df

    def levene_test(self, df, model_tags):
        result = pd.DataFrame()
        lev_vecs = [df[df["method"] == tag] for tag in model_tags]
        for m in self.metrics:
            result[m] = [levene(*[vec[m] for vec in lev_vecs])]
        return result

    def normality_plots(self, df, save_dir=None):
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
        plt.show()

        if save_dir:
            plt.savefig(f"{save_dir}/normality_plot.pdf")

    def anova(self, df, model_tags, save_dir=None):
        figure, axes = plt.subplots(
            1, len(self.metrics), sharex=False, sharey=False, figsize=(28, 8)
        )
        for i, metric in enumerate(self.metrics):
            model = f_oneway(*[df[df["method"] == tag][metric] for tag in model_tags])
            ax = sns.boxplot(
                y=metric,
                x="method",
                hue="method",
                ax=axes[i],
                data=df,
                palette="Set2",
                legend=False,
            )
            title = metric.upper()
            ax.set_title(f"p={model.pvalue:.1e}")
            ax.set_xlabel("")
            ax.set_ylabel(title)
            x_tick_labels = ax.get_xticklabels()
            label_text_list = [x.get_text() for x in x_tick_labels]
            new_xtick_labels = ["\n".join(x.split("_")) for x in label_text_list]
            ax.set_xticks(list(range(0, len(x_tick_labels))))
            ax.set_xticklabels(new_xtick_labels)
        plt.tight_layout()
        plt.show()

        if save_dir:
            plt.savefig(f"{save_dir}/anova.pdf")

    @staticmethod
    def tukey_hsd_by_metric(df, metric, model_tags):
        return tukey_hsd(*[df[df["method"] == tag][metric] for tag in model_tags])

    def tukeys_to_df(tukeys, cl=0.95):
        result = pd.DataFrame()
        for i in range(len(hsd.statistic) - 1):
            for j in range(i + 1, len(hsd.statistic)):
                s = hsd.statistic[i, j]
                method_compare.append(f"{model_tags[i]}-{model_tags[j]}")
                stats.append(s)
                errorbars.append(
                    [
                        s - hsd.confidence_interval(confidence_level=cl).low[i, j],
                        hsd.confidence_interval(confidence_level=cl).high[i, j] - s,
                    ]
                )
                stat_ind += 1
                errorbars = np.transpose(errorbars)
                tukeys.confidence_interval(confidence_level=cl).low[i, j]

    def mcs_plots(
        self, df, model_tags, direction_dict=None, sig_levels=None, save_dir=None
    ):
        figsize = (20, 10)
        nrow = math.ceil(len(self.metrics) / 3)
        fig, ax = plt.subplots(nrow, 3, figsize=figsize)

        if not direction_dict:
            direction_dict = {
                "mae": "minimize",
                "mse": "minimize",
                "r2": "maximize",
                "ktau": "maximize",
                "spearmanr": "maximize",
            }

        if not sig_levels:
            sig_levels = [0.05, 0.01, 0.001]

        for i, metric in enumerate(self.metrics):
            metric = metric.lower()

            row = i // 3
            col = i % 3

            reverse_cmap = False
            if direction_dict[metric] == "minimize":
                reverse_cmap = True

            hsd = self.tukey_hsd_by_metric(df, metric, model_tags)

            cmap = "coolwarm"
            if reverse_cmap:
                cmap = cmap + "_r"

            significance = pd.DataFrame(hsd.pvalue)
            significance[(hsd.pvalue < sig_levels[2]) & (hsd.pvalue >= 0)] = "***"
            significance[
                (hsd.pvalue < sig_levels[1]) & (hsd.pvalue >= sig_levels[2])
            ] = "**"
            significance[
                (hsd.pvalue < sig_levels[0]) & (hsd.pvalue >= sig_levels[1])
            ] = "*"
            significance[(hsd.pvalue >= sig_levels[0])] = ""

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

            x_label_list = [x for x in model_tags]
            y_label_list = [x for x in model_tags]
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
        plt.show()

        if save_dir:
            plt.savefig(f"{save_dir}/mcs_plots.pdf")

    def mean_diff_plots(self, df, model_tags, cl=None, save_dir=None):
        figure, axes = plt.subplots(
            len(self.metrics), 1, figsize=(8, 2 * len(self.metrics)), sharex=False
        )
        ax_ind = 0
        if not cl:
            cl = 0.95
        for metric in self.metrics:
            hsd = self.tukey_hsd_by_metric(df, metric, model_tags)
            hsd_df = pd.DataFrame()
            method_compare = []
            stats = []
            errorbars = []
            stat_ind = 0
            for i in range(len(hsd.statistic) - 1):
                for j in range(i + 1, len(hsd.statistic)):
                    s = hsd.statistic[i, j]
                    method_compare.append(f"{model_tags[i]}-{model_tags[j]}")
                    stats.append(s)
                    errorbars.append(
                        [
                            s - hsd.confidence_interval(confidence_level=cl).low[i, j],
                            hsd.confidence_interval(confidence_level=cl).high[i, j] - s,
                        ]
                    )
                    stat_ind += 1
            errorbars = np.transpose(errorbars)
            hsd_df = pd.DataFrame({"method": method_compare, "stat": stats})
            ax = axes[ax_ind]
            ax.errorbar(
                data=hsd_df, x="stat", y="method", xerr=errorbars, fmt="o", capsize=5
            )
            ax.axvline(0, ls="--", lw=3)
            ax.set_title(metric)
            ax.set_xlabel("Mean Difference")
            ax.set_ylabel("")
            ax.set_xlim(-0.2, 0.2)
            ax_ind += 1
        figure.suptitle("Multiple Comparison of Means\nTukey HSD, FWER=0.05")
        plt.tight_layout()
        plt.show()

        if save_dir:
            plt.savefig(f"{save_dir}/mean_diffs.pdf")

    def to_json(levene, tukeys):
        pass

    def report():
        pass

    def write_report():
        pass
