import click

from openadmet_models.cli.anvil import anvil


@click.group()
def cli():
    """OpenADMET CLI"""
    pass


cli.add_command(anvil)
