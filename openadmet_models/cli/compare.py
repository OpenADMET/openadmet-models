import click

from openadmet_models.comparison.posthoc import PostHocComparison


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
    model_stats, model_tag, report=False, output_dir=None, comparison="posthoc"
):
    """Compare two or more models from summary statistics"""
    if comparison == "posthoc":
        comp = PostHocComparison()
    else:
        raise NotImplementedError
    comp.compare(model_stats, model_tag, report, output_dir)


if __name__ == "__main__":
    compare()
