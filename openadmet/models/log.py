"""Logging configuration for the OpenADMET models package."""

from loguru import logger
from rich.logging import RichHandler
import sys


def is_notebook() -> bool:
    try:
        get_ipython  # type: ignore
        return True
    except NameError:
        return False


if not is_notebook():
    from rich.logging import RichHandler

    logger.configure(handlers=[{"sink": RichHandler(), "format": "{message}"}])
