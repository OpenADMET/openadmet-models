import click

from openadmet_models.cli.anvil import anvil
from openadmet_models.cli.compare import compare


@click.group()
def cli():
    """OpenADMET CLI"""
    pass


cli.add_command(anvil)
cli.add_command(compare)
