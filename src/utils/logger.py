"""
Logger module for the LangGraph agent router.

This module provides a configurable logging system with different verbosity levels
and color-coded output for better readability.
"""

import logging
import sys
from enum import Enum
from typing import Optional


# ANSI color codes for colored terminal output
class Colors:
    """ANSI color codes for terminal output."""
    RESET = "\033[0m"
    BOLD = "\033[1m"
    BLACK = "\033[30m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    WHITE = "\033[37m"
    BRIGHT_BLACK = "\033[90m"
    BRIGHT_RED = "\033[91m"
    BRIGHT_GREEN = "\033[92m"
    BRIGHT_YELLOW = "\033[93m"
    BRIGHT_BLUE = "\033[94m"
    BRIGHT_MAGENTA = "\033[95m"
    BRIGHT_CYAN = "\033[96m"
    BRIGHT_WHITE = "\033[97m"
    BG_BLACK = "\033[40m"
    BG_RED = "\033[41m"
    BG_GREEN = "\033[42m"
    BG_YELLOW = "\033[43m"
    BG_BLUE = "\033[44m"
    BG_MAGENTA = "\033[45m"
    BG_CYAN = "\033[46m"
    BG_WHITE = "\033[47m"


class LogLevel(Enum):
    """Enum for log levels with corresponding logging level values."""
    DEBUG = logging.DEBUG
    INFO = logging.INFO
    WARNING = logging.WARNING
    ERROR = logging.ERROR
    CRITICAL = logging.CRITICAL


class ColoredFormatter(logging.Formatter):
    """
    Custom formatter for colored log output.
    
    This formatter adds color to log messages based on their level.
    """
    
    # Color mapping for different log levels
    COLORS = {
        'DEBUG': Colors.BRIGHT_BLUE,
        'INFO': Colors.BRIGHT_GREEN,
        'WARNING': Colors.BRIGHT_YELLOW,
        'ERROR': Colors.BRIGHT_RED,
        'CRITICAL': Colors.BG_RED + Colors.WHITE + Colors.BOLD,
    }
    
    def format(self, record):
        """Format the log record with appropriate colors."""
        # Save the original format
        format_orig = self._style._fmt
        
        # Apply color formatting based on the log level
        levelname = record.levelname
        if levelname in self.COLORS:
            # Color the level name
            colored_levelname = f"{self.COLORS[levelname]}{levelname}{Colors.RESET}"
            record.levelname = colored_levelname
            
            # Add color to the timestamp and logger name
            self._style._fmt = (
                f"{Colors.BRIGHT_BLACK}%(asctime)s{Colors.RESET} - "
                f"{Colors.CYAN}%(name)s{Colors.RESET} - "
                f"%(levelname)s - "
                f"%(message)s"
            )
        
        # Format the record
        result = logging.Formatter.format(self, record)
        
        # Restore the original format
        self._style._fmt = format_orig
        
        return result


class AgentLogger:
    """
    Logger class for the agent router system.
    
    This class provides methods for logging messages at different verbosity levels
    with color-coded output.
    """
    
    _instance = None
    
    def __new__(cls):
        """Singleton pattern to ensure only one logger instance exists."""
        if cls._instance is None:
            cls._instance = super(AgentLogger, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        """Initialize the logger if it hasn't been initialized yet."""
        if not getattr(self, "_initialized", False):
            self.logger = logging.getLogger("agent_router")
            self.logger.setLevel(logging.INFO)  # Default level
            
            # Remove any existing handlers
            if self.logger.handlers:
                self.logger.handlers.clear()
            
            # Create console handler
            console_handler = logging.StreamHandler(sys.stdout)
            
            # Create colored formatter
            formatter = ColoredFormatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            
            # Add formatter to handler
            console_handler.setFormatter(formatter)
            
            # Add handler to logger
            self.logger.addHandler(console_handler)
            
            self._initialized = True
    
    def set_level(self, level: LogLevel) -> None:
        """
        Set the logging level.
        
        Args:
            level: The log level to set
        """
        self.logger.setLevel(level.value)
        self.logger.info(f"Log level set to {level.name}")
    
    def debug(self, message: str, *args, **kwargs) -> None:
        """
        Log a debug message.
        
        Args:
            message: The message to log
            *args: Additional positional arguments
            **kwargs: Additional keyword arguments
        """
        self.logger.debug(message, *args, **kwargs)
    
    def info(self, message: str, *args, **kwargs) -> None:
        """
        Log an info message.
        
        Args:
            message: The message to log
            *args: Additional positional arguments
            **kwargs: Additional keyword arguments
        """
        self.logger.info(message, *args, **kwargs)
    
    def warning(self, message: str, *args, **kwargs) -> None:
        """
        Log a warning message.
        
        Args:
            message: The message to log
            *args: Additional positional arguments
            **kwargs: Additional keyword arguments
        """
        self.logger.warning(message, *args, **kwargs)
    
    def error(self, message: str, *args, **kwargs) -> None:
        """
        Log an error message.
        
        Args:
            message: The message to log
            *args: Additional positional arguments
            **kwargs: Additional keyword arguments
        """
        self.logger.error(message, *args, **kwargs)
    
    def critical(self, message: str, *args, **kwargs) -> None:
        """
        Log a critical message.
        
        Args:
            message: The message to log
            *args: Additional positional arguments
            **kwargs: Additional keyword arguments
        """
        self.logger.critical(message, *args, **kwargs)


# Create a singleton instance
logger = AgentLogger()


def get_logger() -> AgentLogger:
    """
    Get the logger instance.
    
    Returns:
        The logger instance
    """
    return logger


def set_log_level(level: str) -> None:
    """
    Set the log level from a string.
    
    Args:
        level: The log level as a string (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    """
    try:
        log_level = LogLevel[level.upper()]
        logger.set_level(log_level)
    except KeyError:
        valid_levels = [level.name for level in LogLevel]
        logger.error(f"Invalid log level: {level}. Valid levels are: {', '.join(valid_levels)}")
