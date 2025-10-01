"""Logging configuration for the OpenADMET models package."""

from loguru import logger
from rich.logging import RichHandler
import os


def is_notebook() -> bool:
    """Check if the code is running in a Jupyter notebook environment."""
    try:
        get_ipython  # type: ignore
        return True
    except NameError:
        return False


if not is_notebook() and not os.getenv("OADMET_NO_RICH_LOGGING"):
    logger.configure(handlers=[{"sink": RichHandler(), "format": "{message}"}])
