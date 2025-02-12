import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pingouin as pg
from scipy import stats
import seaborn as sns
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, precision_score, recall_score
from statsmodels.stats.anova import AnovaRM
from statsmodels.stats.libqsturng import psturng, qsturng
from matplotlib import cm
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_curve, precision_recall_curve
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score, average_precision_score, matthews_corrcoef, precision_score, recall_score
import scikit_posthocs as sp


import math


def calc_regression_metrics(df, cycle_col, val_col, pred_col, thresh):
    """
    Calculate regression metrics (MAE, MSE, R2, prec, recall) for each method and split

    :param df: input dataframe must contain columns [method, split] as well the columns specified in the arguments
    :param cycle_col: column indicating the cross-validation fold
    :param val_col: column with the ground truth value
    :param pred_col: column with predictions
    :param thresh: threshold for binary classification
    :return: a dataframe with [cv_cycle, method, split, mae, mse, r2, prec, recall]
    """
    df_in = df.copy()
    metric_ls = ["mae", "mse", "r2", "rho", "prec", "recall"]
    metric_list = []
    df_in['true_class'] = df_in[val_col] > thresh
    # Make sure the thresh variable creates 2 classes
    assert len(df_in.true_class.unique()) == 2, "Binary classification requires two classes"
    df_in['pred_class'] = df_in[pred_col] > thresh

    for k, v in df_in.groupby([cycle_col, "method", "split"]):
        cycle, method, split = k
        mae = mean_absolute_error(v[val_col], v[pred_col])
        mse = mean_squared_error(v[val_col], v[pred_col])
        r2 = r2_score(v[val_col], v[pred_col])
        recall = recall_score(v.true_class, v.pred_class)
        prec = precision_score(v.true_class, v.pred_class)
        rho, _ = spearmanr(v[val_col], v[pred_col])
        metric_list.append([cycle, method, split, mae, mse, r2, rho, prec, recall])
    metric_df = pd.DataFrame(metric_list, columns=["cv_cycle", "method", "split"] + metric_ls)
    return metric_df


def rm_tukey_hsd(df, metric, group_col, alpha=0.05, sort = False, direction_dict=None):
    """
    Perform repeated measures Tukey HSD test on the given dataframe.

    Parameters:
    df (pd.DataFrame): Input dataframe containing the data.
    metric (str): The metric column name to perform the test on.
    group_col (str): The column name indicating the groups.
    alpha (float): Significance level for the test. Default is 0.05.
    sort (bool): Whether to sort the output tables. Default is False.

    Returns:
    tuple: A tuple containing:
        - result_tab (pd.DataFrame): DataFrame with pairwise comparisons and adjusted p-values.
        - df_means (pd.DataFrame): DataFrame with mean values for each group.
        - df_means_diff (pd.DataFrame): DataFrame with mean differences between groups.
        - pc (pd.DataFrame): DataFrame with adjusted p-values for pairwise comparisons.
    """
    if sort and direction_dict and metric in direction_dict:
        if direction_dict[metric] == 'maximize':
            df_means = df.groupby(group_col).mean(numeric_only=True).sort_values(metric, ascending=False)
        elif direction_dict[metric] == 'minimize':
            df_means = df.groupby(group_col).mean(numeric_only=True).sort_values(metric, ascending=True)
        else:
            raise ValueError("Invalid direction. Expected 'maximize' or 'minimize'.")
    else:
        df_means = df.groupby(group_col).mean(numeric_only=True)

    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=RuntimeWarning,
                                message='divide by zero encountered in scalar divide')
        aov = pg.rm_anova(dv=metric, within=group_col, subject='cv_cycle', data=df, detailed=True)
    mse = aov.loc[1, 'MS']
    df_resid = aov.loc[1, 'DF']

    methods = df_means.index
    n_groups = len(methods)
    n_per_group = df[group_col].value_counts().mean()

    tukey_se = np.sqrt(2 * mse / (n_per_group))
    q = qsturng(1 - alpha, n_groups, df_resid)

    num_comparisons = len(methods) * (len(methods) - 1) // 2
    result_tab = pd.DataFrame(index=range(num_comparisons),
                              columns=["group1", "group2", "meandiff", "lower", "upper", "p-adj"])

    df_means_diff = pd.DataFrame(index=methods, columns=methods, data=0.0)
    pc = pd.DataFrame(index=methods, columns=methods, data=1.0)

    # Calculate pairwise mean differences and adjusted p-values
    row_idx = 0
    for i, method1 in enumerate(methods):
        for j, method2 in enumerate(methods):
            if i < j:
                group1 = df[df[group_col] == method1][metric]
                group2 = df[df[group_col] == method2][metric]
                mean_diff = group1.mean() - group2.mean()
                studentized_range = np.abs(mean_diff) / tukey_se
                adjusted_p = psturng(studentized_range * np.sqrt(2), n_groups, df_resid)
                if isinstance(adjusted_p, np.ndarray):
                    adjusted_p = adjusted_p[0]
                lower = mean_diff - (q / np.sqrt(2) * tukey_se)
                upper = mean_diff + (q / np.sqrt(2) * tukey_se)
                result_tab.loc[row_idx] = [method1, method2, mean_diff, lower, upper, adjusted_p]
                pc.loc[method1, method2] = adjusted_p
                pc.loc[method2, method1] = adjusted_p
                df_means_diff.loc[method1, method2] = mean_diff
                df_means_diff.loc[method2, method1] = -mean_diff
                row_idx += 1

    df_means_diff = df_means_diff.astype(float)

    result_tab["group1_mean"] = result_tab["group1"].map(df_means[metric])
    result_tab["group2_mean"] = result_tab["group2"].map(df_means[metric])

    result_tab.index = result_tab['group1'] + ' - ' + result_tab['group2']

    return result_tab, df_means, df_means_diff, pc


# -------------- Plotting routines -------------------#


def make_boxplots_parametric(df, metric_ls):
    """
    Create boxplots for each metric using repeated measures ANOVA.

    Parameters:
    df (pd.DataFrame): Input dataframe containing the data.
    metric_ls (list of str): List of metric column names to create boxplots for.

    Returns:
    None
    """
    sns.set_context('notebook')
    sns.set(rc={'figure.figsize': (4, 3)}, font_scale=1.5)
    sns.set_style('whitegrid')
    figure, axes = plt.subplots(1, len(metric_ls), sharex=False, sharey=False, figsize=(28, 8))
    # figure, axes = plt.subplots(1, 3, sharex=False, sharey=False, figsize=(16, 8))

    for i, stat in enumerate(metric_ls):
        model = AnovaRM(data=df, depvar=stat, subject='cv_cycle', within=['method']).fit()
        p_value = model.anova_table['Pr > F'].iloc[0]
        ax = sns.boxplot(y=stat, x="method", hue="method", ax=axes[i], data=df, palette="Set2", legend=False)
        title = stat.upper()
        ax.set_title(f"p={p_value:.1e}")
        ax.set_xlabel("")
        ax.set_ylabel(title)
        x_tick_labels = ax.get_xticklabels()
        label_text_list = [x.get_text() for x in x_tick_labels]
        new_xtick_labels = ["\n".join(x.split("_")) for x in label_text_list]
        ax.set_xticks(list(range(0, len(x_tick_labels))))
        ax.set_xticklabels(new_xtick_labels)
    plt.tight_layout()

def make_boxplots_nonparametric(df, metric_ls):
    sns.set_context('notebook')
    sns.set(rc={'figure.figsize': (4, 3)}, font_scale=1.5)
    sns.set_style('whitegrid')
    figure, axes = plt.subplots(1, 6, sharex=False, sharey=False, figsize=(28, 8))

    for i, stat in enumerate(metric_ls):
        friedman = pg.friedman(df, dv=stat, within="method", subject="cv_cycle")['p-unc'].values[0]
        ax = sns.boxplot(y=stat, x="method", hue="method", ax=axes[i], data=df, palette="Set2", legend=False)
        title = stat.replace("_", " ").upper()
        ax.set_title(f"p={friedman:.1e}")
        ax.set_xlabel("")
        ax.set_ylabel(title)
        x_tick_labels = ax.get_xticklabels()
        label_text_list = [x.get_text() for x in x_tick_labels]
        new_xtick_labels = ["\n".join(x.split("_")) for x in label_text_list]
        ax.set_xticks(list(range(0, len(x_tick_labels))))
        ax.set_xticklabels(new_xtick_labels)
    plt.tight_layout()
        
def make_sign_plots_nonparametric(df, metric_ls):
    heatmap_args = {'linewidths': 0.25, 'linecolor': '0.5', 'clip_on': True, 'square': True}
    sns.set(rc={'figure.figsize': (4, 3)}, font_scale=1.5)
    figure, axes = plt.subplots(1, 6, sharex=False, sharey=True, figsize=(26, 8))

    for i, stat in enumerate(metric_ls):
        pc = sp.posthoc_conover_friedman(df, y_col=stat, group_col="method", block_col="cv_cycle", p_adjust="holm",
                                         melted=True)
        sub_ax, sub_c = sp.sign_plot(pc, **heatmap_args, ax=axes[i], xticklabels=True)  # Update xticklabels parameter
        sub_ax.set_title(stat.upper())

def make_critical_difference_diagrams(df, metric_ls):
    figure, axes = plt.subplots(6, 1, sharex=True, sharey=False, figsize=(16, 10))
    for i, stat in enumerate(metric_ls):
        avg_rank = df.groupby("cv_cycle")[stat].rank(pct=True).groupby(df.method).mean()
        pc = sp.posthoc_conover_friedman(df, y_col=stat, group_col="method", block_col="cv_cycle", p_adjust="holm",
                                         melted=True)
        sp.critical_difference_diagram(avg_rank, pc, ax=axes[i])
        axes[i].set_title(stat.upper())
    plt.tight_layout()

def make_normality_diagnostic(df, metric_ls):
    """
    Create a normality diagnostic plot grid with histograms and QQ plots for the given metrics.

    Parameters:
    df (pd.DataFrame): Input dataframe containing the data.
    metric_ls (list of str): List of metrics to create plots for.

    Returns:
    None
    """
    df_norm = df.copy()
    
    for metric in metric_ls:
        df_norm[metric] = df_norm[metric] - df_norm.groupby("method")[metric].transform("mean")

    df_norm = df_norm.melt(id_vars=["cv_cycle", "method", "split"],
                                   value_vars=metric_ls,
                                   var_name="metric",
                                   value_name="value")

    sns.set_context('notebook', font_scale=1.5)
    sns.set_style('whitegrid')
    
    metrics = df_norm['metric'].unique()
    n_metrics = len(metrics)
    
    fig, axes = plt.subplots(2, n_metrics, figsize=(20, 10))
    
    for i, metric in enumerate(metrics):
        ax = axes[0, i]
        sns.histplot(df_norm[df_norm['metric'] == metric]['value'], kde=True, ax=ax)
        ax.set_title(f'{metric}', fontsize=16)
    
    for i, metric in enumerate(metrics):
        ax = axes[1, i]
        metric_data = df_norm[df_norm['metric'] == metric]['value']
        stats.probplot(metric_data, dist="norm", plot=ax)
        ax.set_title("")
    
    plt.tight_layout()


def mcs_plot(pc, effect_size, means, labels=True, cmap=None, cbar_ax_bbox=None,
             ax=None, show_diff=True, cell_text_size=16, axis_text_size=12,
             show_cbar=True, reverse_cmap=False, vlim=None, **kwargs):
    """
    Create a multiple comparison of means plot using a heatmap.

    Parameters:
    pc (pd.DataFrame): DataFrame containing p-values for pairwise comparisons.
    effect_size (pd.DataFrame): DataFrame containing effect sizes for pairwise comparisons.
    means (pd.Series): Series containing mean values for each group.
    labels (bool): Whether to show labels on the axes. Default is True.
    cmap (str): Colormap to use for the heatmap. Default is None.
    cbar_ax_bbox (tuple): Bounding box for the colorbar axis. Default is None.
    ax (matplotlib.axes.Axes): The axes on which to plot the heatmap. Default is None.
    show_diff (bool): Whether to show the mean differences in the plot. Default is True.
    cell_text_size (int): Font size for the cell text. Default is 16.
    axis_text_size (int): Font size for the axis text. Default is 12.
    show_cbar (bool): Whether to show the colorbar. Default is True.
    reverse_cmap (bool): Whether to reverse the colormap. Default is False.
    vlim (float): Limit for the colormap. Default is None.
    **kwargs: Additional keyword arguments for the heatmap.

    Returns:
    matplotlib.axes.Axes: The axes with the heatmap.
    """
    for key in ['cbar', 'vmin', 'vmax', 'center']:
        if key in kwargs:
            del kwargs[key]

    if not cmap:
        cmap = "coolwarm"
    if reverse_cmap:
        cmap = cmap + "_r"

    significance = pc.copy().astype(object)
    significance[(pc < 0.001) & (pc >= 0)] = '***'
    significance[(pc < 0.01) & (pc >= 0.001)] = '**'
    significance[(pc < 0.05) & (pc >= 0.01)] = '*'
    significance[(pc >= 0.05)] = ''

    np.fill_diagonal(significance.values, '')

    # Create a DataFrame for the annotations
    if show_diff:
        annotations = effect_size.round(3).astype(str) + significance
    else:
        annotations = significance

    hax = sns.heatmap(effect_size, cmap=cmap, annot=annotations, fmt='', cbar=show_cbar, ax=ax,
                      annot_kws={"size": cell_text_size},
                      vmin=-2*vlim if vlim else None, vmax=2*vlim if vlim else None, **kwargs)

    if labels:
        label_list = list(means.index)
        x_label_list = [x + f'\n{means.loc[x].round(2)}' for x in label_list]
        y_label_list = [x + f'\n{means.loc[x].round(2)}\n' for x in label_list]
        hax.set_xticklabels(x_label_list, size=axis_text_size, ha='center', va='top', rotation=0,
                            rotation_mode='anchor')
        hax.set_yticklabels(y_label_list, size=axis_text_size, ha='center', va='center', rotation=90,
                            rotation_mode='anchor')

    hax.set_xlabel('')
    hax.set_ylabel('')

    return hax


def make_mcs_plot_grid(df, stats, group_col, alpha=.05,
                       figsize=(20, 10), direction_dict={}, effect_dict={}, show_diff=True,
                       cell_text_size=16, axis_text_size=12, title_text_size=16, sort_axes=False):
    """
    Create a grid of multiple comparison of means plots using Tukey HSD test results.

    Parameters:
    df (pd.DataFrame): Input dataframe containing the data.
    stats (list of str): List of statistical metrics to create plots for.
    group_col (str): The column name indicating the groups.
    alpha (float): Significance level for the Tukey HSD test. Default is 0.05.
    figsize (tuple): Size of the figure. Default is (20, 10).
    direction_dict (dict): Dictionary indicating whether to minimize or maximize each metric.
    effect_dict (dict): Dictionary with effect size limits for each metric.
    show_diff (bool): Whether to show the mean differences in the plot. Default is True.
    cell_text_size (int): Font size for the cell text. Default is 16.
    axis_text_size (int): Font size for the axis text. Default is 12.
    title_text_size (int): Font size for the title text. Default is 16.
    sort (bool): Whether to sort the axes. Default is False.

    Returns:
    None
    """
    nrow = math.ceil(len(stats) / 3)
    fig, ax = plt.subplots(nrow, 3, figsize=figsize)

    # Set defaults
    for key in ['r2', 'rho', 'prec', 'recall', 'mae', 'mse']:
        direction_dict.setdefault(key, 'maximize' if key in ['r2', 'rho', 'prec', 'recall'] else 'minimize')

    for key in ['r2', 'rho', 'prec', 'recall']:
        effect_dict.setdefault(key, 0.1)

    direction_dict = {k.lower(): v for k, v in direction_dict.items()}
    effect_dict = {k.lower(): v for k, v in effect_dict.items()}

    for i, stat in enumerate(stats):
        stat = stat.lower()

        row = i // 3
        col = i % 3

        if stat not in direction_dict:
            raise ValueError(f"Stat '{stat}' is missing in direction_dict. Please set its value.")
        if stat not in effect_dict:
            raise ValueError(f"Stat '{stat}' is missing in effect_dict. Please set its value.")

        reverse_cmap = False
        if direction_dict[stat] == 'minimize':
            reverse_cmap = True

        _, df_means, df_means_diff, pc = rm_tukey_hsd(df, stat, group_col, alpha,
                                                       sort_axes, direction_dict)

        hax = mcs_plot(pc, effect_size=df_means_diff, means=df_means[stat],
                       show_diff=show_diff, ax=ax[row, col], cbar=True,
                       cell_text_size=cell_text_size, axis_text_size=axis_text_size,
                       reverse_cmap=reverse_cmap, vlim=effect_dict[stat])
        hax.set_title(stat.upper(), fontsize=title_text_size)

    # If there are less plots than cells in the grid, hide the remaining cells
    if (len(stats) % 3) != 0:
        for i in range(len(stats), nrow * 3):
            row = i // 3
            col = i % 3
            ax[row, col].set_visible(False)

    plt.tight_layout()


def make_scatterplot(df, val_col, pred_col, thresh, cycle_col="cv_cycle", group_col="method"):
    """
    Create scatter plots for each method showing the relationship between predicted and measured values.

    Parameters:
    df (pd.DataFrame): Input dataframe containing the data.
    val_col (str): The column name for the ground truth values.
    pred_col (str): The column name for the predicted values.
    thresh (float): Threshold for binary classification.
    cycle_col (str): The column name indicating the cross-validation fold. Default is "cv_cycle".
    group_col (str): The column name indicating the groups/methods. Default is "method".

    Returns:
    None
    """
    df_split_metrics = calc_regression_metrics(df, cycle_col=cycle_col, val_col=val_col, pred_col=pred_col,
                                               thresh=thresh)
    methods = df[group_col].unique()

    fig, axs = plt.subplots(nrows=1, ncols=len(methods), figsize=(25, 10))

    for ax, method in zip(axs, methods):
        df_method = df.query(f"{group_col} == @method")
        df_metrics = df_split_metrics.query(f"{group_col} == @method")
        ax.scatter(df_method[pred_col], df_method[val_col], alpha=0.3)
        ax.plot([df_method[val_col].min(), df_method[val_col].max()],
                [df_method[val_col].min(), df_method[val_col].max()], 'k--', lw=1)

        ax.axhline(y=thresh, color='r', linestyle='--')
        ax.axvline(x=thresh, color='r', linestyle='--')
        ax.set_title(method)

        y_true = df_method[val_col] > thresh
        y_pred = df_method[pred_col] > thresh
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)

        metrics_text = f"MAE: {df_metrics['mae'].mean():.2f}\nMSE: {df_metrics['mse'].mean():.2f}\nR2: {df_metrics['r2'].mean():.2f}\nrho: {df_metrics['rho'].mean():.2f}\nPrecision: {precision:.2f}\nRecall: {recall:.2f}"
        ax.text(0.05, .5, metrics_text, transform=ax.transAxes, verticalalignment='top')
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Measured')

    plt.tight_layout()
    plt.show()


def ci_plot(result_tab, ax_in, name):
    """
    Create a confidence interval plot for the given result table.

    Parameters:
    result_tab (pd.DataFrame): DataFrame containing the results with columns 'meandiff', 'lower', and 'upper'.
    ax_in (matplotlib.axes.Axes): The axes on which to plot the confidence intervals.
    name (str): The title of the plot.

    Returns:
    None
    """
    result_err = np.array([result_tab['meandiff'] - result_tab['lower'],
                           result_tab['upper'] - result_tab['meandiff']])
    sns.set(rc={'figure.figsize': (6, 2)})
    sns.set_context('notebook')
    sns.set_style('whitegrid')
    ax = sns.pointplot(x=result_tab.meandiff, y=result_tab.index, marker='o', linestyle='', ax=ax_in)
    ax.errorbar(y=result_tab.index, x=result_tab['meandiff'], xerr=result_err, fmt='o', capsize=5)
    ax.axvline(0, ls="--", lw=3)
    ax.set_xlabel("Mean Difference")
    ax.set_ylabel("")
    ax.set_title(name)
    ax.set_xlim(-0.2, 0.2) 


def make_ci_plot_grid(df_in, metric_list, group_col="method"):
    """
     Create a grid of confidence interval plots for multiple metrics using Tukey HSD test results.

     Parameters:
     df_in (pd.DataFrame): Input dataframe containing the data.
     metric_list (list of str): List of metric column names to create confidence interval plots for.
     group_col (str): The column name indicating the groups. Default is "method".

     Returns:
     None
     """
    figure, axes = plt.subplots(len(metric_list), 1, figsize=(8, 2 * len(metric_list)), sharex=False)
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])
    for i, metric in enumerate(metric_list):
        df_tukey, _, _, _ = rm_tukey_hsd(df_in, metric, group_col=group_col)
        ci_plot(df_tukey, ax_in=axes[i], name=metric)
    figure.suptitle("Multiple Comparison of Means\nTukey HSD, FWER=0.05")
    plt.tight_layout()


def recall_at_precision(y_true, y_score, precision_threshold=0.5, direction='greater'):
    if direction not in ['greater', 'lesser']:
        raise ValueError("Invalid direction. Expected one of: ['greater', 'lesser']")

    y_true = np.array(y_true)
    y_score = np.array(y_score)
    thresholds = np.unique(y_score)
    thresholds = np.sort(thresholds)

    if direction == 'greater':
        thresholds = np.sort(thresholds)
    else:  
        thresholds = np.sort(thresholds)[::-1]

    for threshold in thresholds:
        if direction == 'greater':
            y_pred = y_score >= threshold
        else:  
            y_pred = y_score <= threshold

        precision = precision_score(y_true, y_pred)
        if precision >= precision_threshold:
            recall = recall_score(y_true, y_pred)
            return recall, threshold
    return np.nan, None

def calc_classification_metrics(df_in, cycle_col, val_col, prob_col, pred_col):
    metric_list = []
    for k, v in df_in.groupby([cycle_col, "method", "split"]):
        cycle, method, split = k
        roc_auc = roc_auc_score(v[val_col], v[prob_col])
        pr_auc = average_precision_score(v[val_col], v[prob_col])
        mcc = matthews_corrcoef(v[val_col], v[pred_col])
        
        recall, _ = recall_at_precision(v[val_col].astype(bool), v[prob_col], precision_threshold=0.8, direction='greater')
        tnr, _ = recall_at_precision(~v[val_col].astype(bool), v[prob_col], precision_threshold=0.8, direction='lesser')

        metric_list.append([cycle, method, split, roc_auc, pr_auc, mcc, recall, tnr])
        
    metric_df = pd.DataFrame(metric_list, columns=["cv_cycle", "method", "split",
                                                    "roc_auc", "pr_auc", "mcc", "recall", "tnr"])
    return metric_df


def make_curve_plots(df):
    df_plot = df.query("cv_cycle == 0 and split == 'scaffold'").copy()
    color_map = plt.get_cmap('tab10')
    le = LabelEncoder()
    df_plot['color'] = le.fit_transform(df_plot['method'])
    colors = color_map(df_plot['color'].unique())
    val_col = "Sol"
    prob_col = "Sol_prob"

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    for (k, v), color in zip(df_plot.groupby("method"), colors):
        roc_auc = roc_auc_score(v[val_col], v[prob_col])
        pr_auc = average_precision_score(v[val_col], v[prob_col])
        fpr, recall_pos, thresholds_roc = roc_curve(v[val_col], v[prob_col])
        precision, recall, thresholds_pr = precision_recall_curve(v[val_col], v[prob_col])

        _, threshold_recall_pos = recall_at_precision(v[val_col].astype(bool), v[prob_col], precision_threshold=0.8, direction='greater')
        _, threshold_recall_neg = recall_at_precision(~v[val_col].astype(bool), v[prob_col], precision_threshold=0.8, direction='lesser')

        fpr_recall_pos = fpr[np.abs(thresholds_roc - threshold_recall_pos).argmin()]
        fpr_recall_neg = fpr[np.abs(thresholds_roc - threshold_recall_neg).argmin()]
        recall_recall_pos = recall[np.abs(thresholds_pr - threshold_recall_pos).argmin()]
        recall_recall_neg = recall[np.abs(thresholds_pr - threshold_recall_neg).argmin()]

        axes[0].plot(fpr, recall_pos, label=f"{k} (ROC AUC={roc_auc:.03f})", color=color, alpha=0.75)
        axes[1].plot(recall, precision, label=f"{k} (PR AUC={pr_auc:.03f})", color=color, alpha=0.75)

        axes[0].axvline(fpr_recall_pos, color=color, linestyle=':', alpha=0.75)
        axes[0].axvline(fpr_recall_neg, color=color, linestyle='--', alpha=0.75)
        axes[1].axvline(recall_recall_pos, color=color, linestyle=':', alpha=0.75)
        axes[1].axvline(recall_recall_neg, color=color, linestyle='--', alpha=0.75)

    axes[0].plot([0, 1], [0, 1], "--", color="black", lw=0.5)
    axes[0].set_xlabel("False Positive Rate")
    axes[0].set_ylabel("True Positive Rate")
    axes[0].set_title("ROC Curve")
    axes[0].legend()
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].set_title("Precision-Recall Curve")
    axes[1].legend()
    plt.tight_layout()