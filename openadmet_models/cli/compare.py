import click
from openadmet_models.comparison.posthoc import PostHocComparison

@click.command()
@click.option(
    "--model-stats",
    multiple=True,
    help="Path to YAML of model stats",
    required=True,
    type=click.Path(exists=True),
)
@click.option(
    "--model-tag",
    multiple=True,
)
@click.option(
    "--output-dir",
    help="Path to output directory",
    required=False,
    type=click.Path(exists=True),
)
@click.option(
    "--comparison",
    help="Type of comparison to do",
    required=False,
)
def compare(model_stats, model_tag, output_dir, comparison="posthoc"):
    """Compare two or more models from summary statistics"""
    if comparison == "posthoc":
        comp = PostHocComparison()
    else:
        raise NotImplementedError
    comp.compare(model_stats, model_tag)
    #comp.write_report(output_dir)

if __name__ == '__main__':
    compare()