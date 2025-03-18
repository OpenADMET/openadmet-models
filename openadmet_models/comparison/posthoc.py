import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    Image,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)
from scipy import stats
from scipy.stats import levene, tukey_hsd
from statsmodels.stats.anova import AnovaRM

from openadmet_models.comparison.compare_base import ComparisonBase, comparisons


@comparisons.register("PostHoc")
class PostHocComparison(ComparisonBase):

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
        return self._metrics_names

    @property
    def direction_dict(self):
        return self._direction_dict

    @property
    def sig_levels(self):
        return self._sig_levels

    @property
    def cl(self):
        return self._confidence_level

    @property
    def stats_names(self):
        return self._stats_names

    def compare(self, model_stats_fns, model_tags, report=False, output_dir=None):
        df = self.json_to_df(model_stats_fns, model_tags)
        stats_dfs = []
        stats_dfs.append(self.levene_test(df, model_tags))
        stats_dfs.append(self.get_tukeys_df(df, model_tags))
        if output_dir:
            self.stats_to_json(stats_dfs, output_dir=output_dir)

        plot_data = {}
        plot_data["normality"] = self.normality_plots(df, output_dir)
        plot_data["anova"] = self.anova(df, model_tags, output_dir)
        plot_data["mcs"] = self.mcs_plots(df, model_tags, output_dir)
        plot_data["mean_diff"] = self.mean_diff_plots(
            df, model_tags, self.cl, output_dir
        )

        self.report(stats_dfs, plot_data, report, output_dir)

        return stats_dfs

    def json_to_df(self, model_stats_fns, model_tags):
        """
        Takes the model statistics cross validation json from an anvil run,
        will likely have the name (anvil_run/cross_validation_metrics.json)
        """
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
            l = levene(*[vec[m] for vec in lev_vecs])
            result[m] = {"stat": l.statistic, "pvalue": l.pvalue}
        return result

    def normality_plots(self, df, output_dir=None):
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
            plt.savefig(f"{output_dir}/normality_plot.pdf")

        return fig

    def anova(self, df, model_tags, output_dir=None):
        fig, axes = plt.subplots(
            1, len(self.metrics), sharex=False, sharey=False, figsize=(28, 8)
        )
        for i, metric in enumerate(self.metrics):
            anova_df = pd.DataFrame({metric: df[metric]})
            anova_df["cv_cycle"] = np.tile([i for i in range(1, 26)], 3)
            anova_df["method"] = df["method"]
            model = AnovaRM(
                anova_df, depvar=metric, subject="cv_cycle", within=["method"]
            ).fit()
            ax = sns.boxplot(
                y=metric,
                x="method",
                hue="method",
                ax=axes[i],
                data=anova_df,
                palette="Set2",
                legend=False,
            )
            title = metric.upper()
            ax.set_title(f"p={model.anova_table['Pr > F'].iloc[0]}")
            ax.set_xlabel("")
            ax.set_ylabel(title)
            x_tick_labels = ax.get_xticklabels()
            label_text_list = [x.get_text() for x in x_tick_labels]
            new_xtick_labels = ["\n".join(x.split("_")) for x in label_text_list]
            ax.set_xticks(list(range(0, len(x_tick_labels))))
            ax.set_xticklabels(new_xtick_labels)
        plt.tight_layout()

        if output_dir:
            plt.savefig(f"{output_dir}/anova.pdf")

        return fig

    @staticmethod
    def tukey_hsd_by_metric(df, metric, model_tags):
        return tukey_hsd(
            *[np.array(df[df["method"] == tag][metric]) for tag in model_tags]
        )

    def get_tukeys_df(self, df, model_tags, cl=0.95):
        tukeys = [
            self.tukey_hsd_by_metric(df, metric, model_tags) for metric in self.metrics
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
                    method_compare.append(f"{model_tags[i]}-{model_tags[j]}")
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

    def mcs_plots(self, df, model_tags, output_dir=None):
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

            hsd = self.tukey_hsd_by_metric(df, metric, model_tags)

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

        if output_dir:
            plt.savefig(f"{output_dir}/mcs_plots.pdf")

        return fig

    def mean_diff_plots(self, df, model_tags, cl=None, output_dir=None):

        fig, axes = plt.subplots(
            len(self.metrics), 1, figsize=(8, 2 * len(self.metrics)), sharex=False
        )
        ax_ind = 0
        if not cl:
            cl = 0.95
        tukeys_df = self.get_tukeys_df(df, model_tags, cl=cl)

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

    def stats_to_json(self, stats_dfs, output_dir):
        for stat_df, name in zip(stats_dfs, self.stats_names):
            stat_df.to_json(f"{output_dir}/{name}.json")

    def report(self, data_dfs, plot_data, write=False, output_dir=None):
        """
        Report the analysis and save figures
        """
        if write:
            self.write_report(data_dfs, plot_data, output_dir)

    def write_report(self, data_dfs, plot_data, output_dir):
        doc = SimpleDocTemplate(
            f"{output_dir}/posthoc.pdf",
            pagesize=letter,
            topMargin=0.5 * inch,
            leftMargin=0.25 * inch,
        )
        elements = []
        styles = getSampleStyleSheet()
        styleH = styles["Heading1"]

        for df, name in zip(data_dfs, self.stats_names):
            elements.append(Paragraph(name, styleH))
            elements.append(Spacer(1, 0.25 * inch))
            data = [df.columns.to_list()] + df.values.tolist()
            table = Table(data, hAlign="LEFT")
            table.setStyle(
                TableStyle(
                    [
                        ("FONT", (0, 0), (-1, 0), "Helvetica-Bold"),
                        ("FONTSIZE", (0, 0), (-1, 0), 9),
                        ("ALIGN", (0, 0), (-1, 0), "CENTER"),
                        ("ALIGN", (0, 0), (0, -1), "LEFT"),
                        ("INNERGRID", (0, 0), (-1, -1), 0.50, colors.black),
                        ("BOX", (0, 0), (-1, -1), 0.25, colors.black),
                    ]
                )
            )
            elements.append(table)
            elements.append(Spacer(1, 0.2 * inch))

        # for plot in plot_data:
        #     print(plot)
        #     elements.append(Spacer(1, 0.2 * inch))
        #     elements.append(Image(plot))

        doc.build(elements)
