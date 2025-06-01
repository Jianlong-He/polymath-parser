"""Provides an abstract base class for all agents in the system."""

import logging
from typing import List, Optional

class Agent:
    """
    An abstract superclass for Agents.
    Provides standardized logging with color-coding for easy identification.
    """

    # ANSI Color Codes
    RED = '\033[31m'
    GREEN = '\033[32m'
    YELLOW = '\033[33m'
    BLUE = '\033[34m'
    MAGENTA = '\033[35m'
    CYAN = '\033[36m'
    WHITE = '\033[37m'
    BG_BLACK = '\033[40m'
    RESET = '\033[0m'

    name: str = "Base Agent" # Default name
    color: str = WHITE      # Default color

    def __init__(self, log_level=logging.INFO):
        """Initializes the agent and its logger."""
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(log_level)
        
        if not self.logger.handlers: # Ensure handlers aren't added repeatedly
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                f"{self.BG_BLACK}{self.color}[%(asctime)s] [%(name)s]{self.RESET} %(message)s",
                 datefmt="%H:%M:%S",
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
        self.log(f"Initialized.")

    def log(self, message: str, level: int = logging.INFO, color_override: Optional[str] = None):
        """
        Logs a message, identifying the agent with its color.
        Color is primarily handled by the formatter; color_override is for special cases.
        """
        if color_override:
            log_message = f"{self.BG_BLACK}{color_override}{message}{self.RESET}"
            self.logger.log(level, log_message)
        else:
            self.logger.log(level, message)

    def log_error(self, message: str):
        """Logs an error message in red."""
        self.log(message, level=logging.ERROR, color_override=self.RED)

    def log_warning(self, message: str):
        """Logs a warning message in yellow."""
        self.log(message, level=logging.WARNING, color_override=self.YELLOW)