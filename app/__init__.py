"""
Polymath Parser Application Package

This package contains the main application framework and the user interface.
"""

from .polymath_parser_framework import PolymathParserFramework
from .polymath_parser_ui import MathExplorerApp, GradioFileData

__all__ = [
    "PolymathParserFramework",
    "MathExplorerApp",
    "GradioFileData" # Exporting the type alias if it's useful externally
]