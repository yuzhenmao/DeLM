# -*- coding: utf-8 -*-
# Adapted from FoundationAgents/AOrchestra:
# https://github.com/FoundationAgents/AOrchestra
#
# Copyright notice and license of the original project are retained
# in accordance with the Apache License, Version 2.0.
# This file includes modifications for the current project.

# @Date    : 2025-03-31
# @Author  : Claude
# @Desc    : Simple colored logger with file output

import os
import sys
import time
from datetime import datetime
from enum import Enum
from typing import Optional, TextIO, Union

class Colors:
    """Terminal color codes for different log levels"""
    BLACK = '\033[30m'
    RED = '\033[31m'
    GREEN = '\033[32m'
    YELLOW = '\033[33m'
    BLUE = '\033[34m'
    MAGENTA = '\033[35m'
    CYAN = '\033[36m'
    WHITE = '\033[37m'
    RESET = '\033[0m'
    BOLD = '\033[1m'

class LogLevel(Enum):
    """Log levels with corresponding colors"""
    DEBUG = (10, Colors.BLUE)
    OPTIMIZE = (25, Colors.CYAN)
    INFO = (20, Colors.GREEN)
    WARNING = (30, Colors.YELLOW)
    ERROR = (40, Colors.RED)
    CRITICAL = (50, Colors.MAGENTA)

class SimpleLogger:
    """Simple logger class that supports both colored terminal output and file logging"""
    
    def __init__(
        self, 
        name: str = "AutoEnv",
        log_level: Union[int, LogLevel] = LogLevel.INFO,
        log_file: Optional[str] = None,
        log_dir: str = "workspace/logs",
        console_output: bool = True
    ):
        """
        Initialize the Logger
        
        Args:
            name: Logger name
            log_level: Minimum log level to display
            log_file: Log file name (if None, will use name_YYYY-MM-DD.log)
            log_dir: Directory to store log files
            console_output: Whether to output logs to console
        """
        self.name = name
        
        # Convert LogLevel enum to int if needed
        if isinstance(log_level, LogLevel):
            self.log_level = log_level.value[0]
        else:
            self.log_level = log_level
        
        self.console_output = console_output
        self.file_output = None
        
        # Define display names for log levels
        self.level_display_names = {
            LogLevel.DEBUG: "DEBUG",
            LogLevel.OPTIMIZE: "OPTIMIZE", 
            LogLevel.INFO: "INFO",
            LogLevel.WARNING: "WARNING",
            LogLevel.ERROR: "ERROR",
            LogLevel.CRITICAL: "CRITICAL"
        }
        
        # Set up file logging
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
            
            # Generate default log filename if not provided
            if log_file is None:
                current_date = datetime.now().strftime("%Y-%m-%d")
                log_file = f"{name}_{current_date}.log"
            
            file_path = os.path.join(log_dir, log_file)
            self.file_output = open(file_path, 'a', encoding='utf-8')
    
    def _log(self, level: LogLevel, message: str, exc_info: bool = False) -> None:
        """Internal method to log messages at specified level.

        If exc_info is True and an exception is currently being handled, the
        traceback is appended to the message (mirroring stdlib logging
        behavior). This lets callers use `logger.error(msg, exc_info=True)`
        without TypeError; previously the kwarg blew up at call sites.
        """
        if level.value[0] < self.log_level:
            return

        if exc_info:
            import sys
            import traceback as _tb
            exc = sys.exc_info()
            if exc and exc[0] is not None:
                message = f"{message}\n{''.join(_tb.format_exception(*exc))}".rstrip()

        # Format the log message
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        level_name = self.level_display_names.get(level, level.name)
        formatted_msg = f"{timestamp} - {level_name} - {message}"
        
        # Write to console if enabled
        if self.console_output:
            color = level.value[1]
            # Add bold to critical messages
            if level == LogLevel.CRITICAL:
                colored_msg = f"{Colors.BOLD}{color}{formatted_msg}{Colors.RESET}"
            else:
                colored_msg = f"{color}{formatted_msg}{Colors.RESET}"
            print(colored_msg)
        
        # Write to file if enabled
        if self.file_output:
            self.file_output.write(formatted_msg + "\n")
            self.file_output.flush()
    
    def log_to_file(self, level: LogLevel, message: str) -> None:
        """
        Log a message to file only, without printing to console
        
        Args:
            level: Log level
            message: Message to log
        """
        if level.value[0] < self.log_level:
            return
            
        # Format the log message
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        level_name = self.level_display_names.get(level, level.name)
        formatted_msg = f"{timestamp} - {level_name} - {message}"
        
        # Write to file if enabled
        if self.file_output:
            self.file_output.write(formatted_msg + "\n")
            self.file_output.flush()
    
    def debug(self, message: str, exc_info: bool = False) -> None:
        """Log a debug message"""
        self._log(LogLevel.DEBUG, message, exc_info=exc_info)

    def info(self, message: str, exc_info: bool = False) -> None:
        """Log an info message"""
        self._log(LogLevel.INFO, message, exc_info=exc_info)

    def optimize(self, message: str, exc_info: bool = False) -> None:
        """Log an optimization info message"""
        self._log(LogLevel.OPTIMIZE, message, exc_info=exc_info)

    def warning(self, message: str, exc_info: bool = False) -> None:
        """Log a warning message"""
        self._log(LogLevel.WARNING, message, exc_info=exc_info)

    def error(self, message: str, exc_info: bool = False) -> None:
        """Log an error message"""
        self._log(LogLevel.ERROR, message, exc_info=exc_info)

    def critical(self, message: str, exc_info: bool = False) -> None:
        """Log a critical message"""
        self._log(LogLevel.CRITICAL, message, exc_info=exc_info)
    
    def agent_action(self, message: str) -> None:
        """Log an agent action with special cyan color and bold formatting"""
        if self.log_level <= LogLevel.INFO.value[0]:  # Only log if INFO level or lower
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            formatted_msg = f"{timestamp} - AGENT_ACTION - {message}"
            
            # Write to console if enabled
            if self.console_output:
                colored_msg = f"{Colors.BOLD}{Colors.CYAN}{formatted_msg}{Colors.RESET}"
                print(colored_msg)
            
            # Write to file if enabled
        if self.file_output:
            self.file_output.write(formatted_msg + "\n")
            self.file_output.flush()

    def agent_thinking(self, message: str) -> None:
        """Log an agent thinking message with special white color and bold formatting"""
        if self.log_level <= LogLevel.INFO.value[0]:  # Only log if INFO level or lower
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            formatted_msg = f"{timestamp} - AGENT_THINKING - {message}"

            # Write to console if enabled
            if self.console_output:
                colored_msg = f"{Colors.BOLD}{Colors.WHITE}{formatted_msg}{Colors.RESET}"
                print(colored_msg)

            # Write to file if enabled
            if self.file_output:
                self.file_output.write(formatted_msg + "\n")
                self.file_output.flush()
    
    def __del__(self):
        """Close file handle when logger is destroyed"""
        if self.file_output:
            self.file_output.close()

# Create a singleton instance for easy import
logger = SimpleLogger()


def logger_to_optimize(message: str, file_path: Optional[str] = None, console: bool = True) -> None:
    """
    Log optimization-related information with special styling and to a dedicated file.

    - Prints with bold cyan (similar to agent_action) for visibility.
    - Writes to a separate file (default workspace/logs/optimize.log) or a provided path.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    formatted_msg = f"{timestamp} - OPTIMIZE - {message}"

    # Console styling
    if console:
        try:
            colored_msg = f"{Colors.BOLD}{Colors.CYAN}{formatted_msg}{Colors.RESET}"
            print(colored_msg)
        except Exception:
            # Fallback without colors
            print(formatted_msg)

    # File path default
    if not file_path:
        log_dir = os.path.join("workspace", "logs")
        os.makedirs(log_dir, exist_ok=True)
        file_path = os.path.join(log_dir, "optimize.log")
    else:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

    # Append to the optimize log file
    try:
        with open(file_path, "a", encoding="utf-8") as f:
            f.write(formatted_msg + "\n")
    except Exception:
        # Best-effort only; do not raise
        pass

def test_logger():
    """Test function to verify the SimpleLogger functionality"""
    
    # Create a test log directory
    test_log_dir = "test_logs"
    if not os.path.exists(test_log_dir):
        os.makedirs(test_log_dir)
    
    # Initialize the logger with a specific test file
    test_logger = SimpleLogger(
        name="test_logger",
        log_level=LogLevel.DEBUG,  # Set to lowest level to show all logs
        log_file="test_logger.log",
        log_dir=test_log_dir
    )
    
    print("\n===== Testing SimpleLogger =====\n")
    
    # Test all log levels
    test_logger.debug("This is a DEBUG message - Should appear in BLUE")
    test_logger.info("This is an INFO message - Should appear in GREEN")
    test_logger.warning("This is a WARNING message - Should appear in YELLOW")
    test_logger.error("This is an ERROR message - Should appear in RED")
    test_logger.critical("This is a CRITICAL message - Should appear in BOLD MAGENTA")
    
    # Test log filtering by creating a new logger with higher minimum level
    print("\n===== Testing Log Level Filtering =====\n")
    filtered_logger = SimpleLogger(
        name="filtered_logger",
        log_level=LogLevel.WARNING,  # Only WARNING and above will be shown
        log_file="filtered_logger.log",
        log_dir=test_log_dir
    )
    
    filtered_logger.debug("This DEBUG message should NOT appear")
    filtered_logger.info("This INFO message should NOT appear")
    filtered_logger.warning("This WARNING message should appear in YELLOW")
    filtered_logger.error("This ERROR message should appear in RED")
    filtered_logger.critical("This CRITICAL message should appear in BOLD MAGENTA")
    
    # Verify file output
    print("\n===== Verifying File Output =====\n")
    
    # Get path to the log file
    log_file_path = os.path.join(test_log_dir, "test_logger.log")
    
    # Check if the log file exists
    if os.path.exists(log_file_path):
        # Read and print the last 5 lines from the log file
        with open(log_file_path, 'r', encoding='utf-8') as file:
            lines = file.readlines()
            print(f"Last 5 lines from log file ({log_file_path}):")
            for line in lines[-5:]:
                print(f"  {line.strip()}")
        print(f"\nLog file successfully created at: {log_file_path}")
    else:
        print(f"ERROR: Log file was not created at: {log_file_path}")
    
    print("\n===== Testing Complete =====\n")


def test_in_app_scenario():
    """Test function to demonstrate how the logger would be used in an application"""
    
    logger = SimpleLogger(name="app_logger")
    
    print("\n===== Simulating Application Logs =====\n")
    
    # Simulate application startup
    logger.info("Application starting up...")
    
    # Simulate configuration loading
    logger.debug("Loading configuration from config.json")
    time.sleep(0.5)
    logger.info("Configuration loaded successfully")
    
    # Simulate some application processing
    logger.info("Processing data files...")
    time.sleep(0.5)
    
    # Simulate a warning condition
    logger.warning("Memory usage is high (85%)")
    
    # Simulate an error condition
    try:
        # Simulate a division by zero error
        result = 100 / 0
    except Exception as e:
        logger.error(f"Error during calculation: {str(e)}")
    
    # Simulate a critical error
    logger.critical("Database connection lost! System cannot continue.")
    
    # Simulate application shutdown
    logger.info("Application shutting down")
    
    print("\n===== Simulation Complete =====\n")


if __name__ == "__main__":
    # Run tests
    test_logger()
    test_in_app_scenario()
