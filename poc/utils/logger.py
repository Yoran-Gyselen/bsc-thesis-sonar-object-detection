import logging
import os

class Logger:
    def __init__(self, log_file):
        self.logger = logging.getLogger('train_logger')
        self.logger.setLevel(logging.DEBUG)
        self.logger.handlers = []  # Clear existing handlers

        # File handler
        self.file_handler = logging.FileHandler(log_file)
        self.file_handler.setLevel(logging.DEBUG)
        self.file_handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
        self.logger.addHandler(self.file_handler)

        # Console handler (for selective output)
        self.console_handler = logging.StreamHandler()
        self.console_handler.setLevel(logging.INFO)
        self.console_handler.setFormatter(logging.Formatter('%(message)s'))
        self.logger.addHandler(self.console_handler)

    def log(self, message, to_console=True):
        """Log a message to file, and optionally to console."""
        if to_console:
            # Log to both file and console
            self.logger.info(message)
        else:
            # Temporarily disable console handler
            self.logger.removeHandler(self.console_handler)
            self.logger.info(message)
            self.logger.addHandler(self.console_handler)
