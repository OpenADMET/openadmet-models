import click

from openadmet.models.comparison.posthoc import PostHocComparison


@click.command()
@click.option(
    "--model-stats",
    multiple=True,
    help="Path to JSON of model stats, needst to be in the same order as model-tag",
    required=True,
    type=click.Path(exists=True),
)
@click.option(
    "--model-tag",
    help="Names to identify different models, user specified in same order as model-stats",
    multiple=True,
)
@click.option(
    "--taskname",
    help="Task names as they appear in the model stats JSON",
    multiple=True,
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
@click.option(
    "--comparison",
    help="Type of comparison to do",
    required=False,
)
def compare(
    model_stats, model_tag, taskname, report=False, output_dir=None, comparison="posthoc"
):
    """Compare two or more models from summary statistics"""
    if comparison == "posthoc":
        comp = PostHocComparison()
    else:
        raise NotImplementedError
    comp.compare(model_stats, model_tag, task_tags=taskname,
                 output_dir=output_dir, report=report)


if __name__ == "__main__":
    compare()
