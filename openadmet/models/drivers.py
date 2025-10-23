"""Stub for driver choices for ML workflows."""

from enum import StrEnum


class DriverType(StrEnum):
    """Enumeration of driver types for models and trainers."""

    SKLEARN = "sklearn"
    LIGHTNING = "lightning"
