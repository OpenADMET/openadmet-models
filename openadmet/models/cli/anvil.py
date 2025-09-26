"""CLI for running Anvil workflows."""

import click

from openadmet.models.anvil.specification import AnvilSpecification


@click.command()
@click.option(
    "--recipe-path",
    help="Path to the recipe YAML file",
    required=True,
    type=click.Path(exists=True),
)
@click.option("--debug", is_flag=True, help="Enable debug mode")
@click.option(
    "--output-dir",
    type=click.Path(file_okay=False, writable=True),
    required=False,
    help="Output directory path",
    default="anvil_training",
)
@click.option(
    "--tag", required=False, help="User-defined model tag to help ID this model"
)
def anvil(recipe_path, tag, debug, output_dir):
    """
    Run an Anvil workflow for model building from a recipe.

    Parameters
    ----------
    recipe_path : str
        Path to the recipe YAML file.
    tag : str, optional
        User-defined model tag to help ID this model, by default None.
    debug : bool, optional
        Enable debug mode, by default False.
    output_dir : str, optional
        Output directory path, by default "anvil_training".

    """
    spec = AnvilSpecification.from_recipe(recipe_path)
    wf = spec.to_workflow()
    click.echo(f"Workflow initialized successfully with recipe: {recipe_path}")
    wf.run(tag=tag, debug=debug, output_dir=output_dir)
    click.echo("Workflow completed successfully")
