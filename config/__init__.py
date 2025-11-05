"""
Configuration module for MSP detection system.

This module provides centralized configuration management for all hyperparameters,
paths, and experimental settings.
"""

from .config import get_default_config

__all__ = ['get_default_config']
