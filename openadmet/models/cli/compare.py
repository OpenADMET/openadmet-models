import click

from openadmet.models.comparison.posthoc import PostHocComparison


@click.command()
@click.option(
    "--model-dir",
    help="Path to main model directory",
    required=True,
    type=click.path(exists=True),
)
@click.option(
    "--label",
    help="Category from the yaml file with which to label each model",
    required=True,
    multiple=True,
)
@click.option(
    "--task-name",
    help="Task names as they appear in the model stats JSON",
    required=False,
    multiple=True,
)
@click.option(
    "--target",
    help="If using label and multitask, give the name of the target",
    requried=False,
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
    model_stats, model_tag, task_name, target=None, output_dir=None, report=False,
):
    """Compare two or more models from summary statistics"""
    comp = PostHocComparison()
    comp.compare(model_stats, model_tag, task_name, target=target,
                 output_dir=output_dir, report=report)


if __name__ == "__main__":
    compare()
