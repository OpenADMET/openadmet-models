"""CLI entry point for OpenADMET models."""

import click

from openadmet.models.cli.anvil import anvil
from openadmet.models.cli.compare import compare
from openadmet.models.cli.predict import predict


@click.group()
def cli():
    """OpenADMET CLI."""
    pass


cli.add_command(anvil)
cli.add_command(compare)
cli.add_command(predict)
