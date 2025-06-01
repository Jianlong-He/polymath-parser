"""
Polymath Parser Utilities Package

This package contains general utility functions for the project.
"""

from .log_utils import reformat_ansi_to_html, ANSI_TO_HTML_COLOR_MAP

__all__ = [
    "reformat_ansi_to_html",
    "ANSI_TO_HTML_COLOR_MAP" # Exporting the map if it's useful elsewhere
]
