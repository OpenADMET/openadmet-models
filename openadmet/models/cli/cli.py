import click

from openadmet.models.cli.anvil import anvil
from openadmet.models.cli.compare import compare


@click.group()
def cli():
    """OpenADMET CLI"""
    pass


cli.add_command(anvil)
cli.add_command(compare)
