"""
Feature extraction module for MSP detection.

Provides functions for extracting statistical and geometric features from
model predictions for meta-classifier training.
"""

from .extraction import extract_heatmap_features

__all__ = ['extract_heatmap_features']
