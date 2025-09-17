"""Logging configuration for the OpenADMET models package."""

from loguru import logger
from rich.logging import RichHandler

logger.configure(handlers=[{"sink": RichHandler(), "format": "{message}"}])
