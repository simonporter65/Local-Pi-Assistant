"""
core/log.py

Centralised logging configuration for the Pi Assistant.
All modules should import get_logger from here rather than using print().

Usage:
    from core.log import get_logger
    logger = get_logger(__name__)   # or get_logger("server"), get_logger("heartbeat")
    logger.info("Starting up")
    logger.warning("Something odd happened")
    logger.error("Something broke", exc_info=True)
"""

import logging
import sys


def get_logger(name: str) -> logging.Logger:
    """Return a configured logger for the given name.

    Idempotent — calling multiple times with the same name returns the same
    logger without adding duplicate handlers.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            logging.Formatter(
                fmt="%(asctime)s [%(name)s] %(levelname)s %(message)s",
                datefmt="%H:%M:%S",
            )
        )
        logger.addHandler(handler)
        logger.setLevel(logging.DEBUG)
    return logger
