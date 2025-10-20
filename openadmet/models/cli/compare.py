"""CLI for comparing model performance."""

import click

from openadmet.models.comparison.posthoc import PostHocComparison


@click.command()
@click.option(
    "--model-dirs",
    help="Path to main model directory",
    required=False,
    type=click.Path(exists=True),
    multiple=True,
)
@click.option(
    "--label-types",
    help="Category from the yaml file with which to label each model",
    required=False,
    multiple=True,
)
@click.option(
    "--model-stats-fns",
    help="File paths to JSON cross validation files with model statistics",
    required=False,
    multiple=True,
)
@click.option(
    "--labels",
    help="User-defined labels for each model",
    required=False,
    multiple=True,
)
@click.option(
    "--task-names",
    help="Task names as they appear in the model stats JSON",
    required=False,
    multiple=True,
)
@click.option(
    "--mt-id",
    help="If using label and multitask, give the name of the target or other identifier",
    required=False,
    multiple=False,
)
@click.option(
    "--output-dir",
    help="Path to output directory",
    required=False,
    type=click.Path(exists=True),
)
@click.option(
    "--report",
    help="Whether to write summary pdf to output-dir",
    required=False,
    type=bool,
)
def compare(
    model_dirs,
    label_types,
    model_stats_fns,
    labels,
    task_names,
    mt_id=None,
    output_dir=None,
    report=False,
):
    """
    Compare two or more models from summary statistics.

    Either (`model_dirs` and `label_types`) OR (`model_stats_fns`, `labels`, and `task_names`) are required.
    If the full anvil directory is available, the first option is recommended. If only the JSON files
    with model statistics are available, use the second option.

    Parameters
    ----------
    model_dirs : list of str, optional
        List of paths to main model directories.
    label_types : list of str, optional
        List of categories from the `anvil_recipe.yaml` file to use for labeling each model.
        Supported values are 'biotarget', 'model', 'feat', and 'tasks'.
    model_stats_fns : list of str, optional
        List of file paths to JSON files containing model statistics.
    labels : list of str, optional
        User-defined list of tags for the models, used for plotting and reporting.
    task_names : list of str, optional
        List of task names as they appear in the model statistics JSON files.
    mt_id : str, optional
        Identifier for the target column when comparing multitask models. Used to select
        the appropriate task from the `anvil_recipe.yaml` file.  Must be a unique
        string not appearing in any target columns for other models in the file. Required if comparing
        multitask models.
    output_dir : str, optional
        Path to output directory.
    report : bool, optional
        Whether to generate a PDF report of the comparison results. Default is False.

    """
    comp = PostHocComparison()
    comp.compare(
        model_dirs,
        label_types,
        model_stats_fns,
        labels,
        task_names,
        mt_id=mt_id,
        output_dir=output_dir,
        report=report,
    )


if __name__ == "__main__":
    compare()
