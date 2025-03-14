import click

from openadmet_models.anvil.anvil_workflow import AnvilSpecification


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
)
def anvil(recipe_path, debug, output_dir):
    """Run an Anvil workflow for model building from a recipe"""
    spec = AnvilSpecification.from_recipe(recipe_path)
    wf = spec.to_workflow()
    click.echo(f"Workflow initialized successfully with recipe: {recipe_path}")
    wf.run(debug=debug, output_dir=output_dir)
    click.echo("Workflow completed successfully")
