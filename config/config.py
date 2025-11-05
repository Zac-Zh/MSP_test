"""
Configuration system for MSP Detection.

Provides a centralized configuration dictionary containing all hyperparameters,
data paths, model settings, and training parameters required for reproducible
experiments in midsagittal plane detection.
"""

import os
import torch


def get_default_config():
    """
    Returns the default configuration for MSP detection experiments.

    This function initializes all hyperparameters, file paths, and experimental
    settings with sensible defaults. Users should modify these values according
    to their specific dataset and computational resources.

    Returns:
        Dictionary containing all configuration parameters

    Configuration Categories:
        - Data paths: Directories for images, labels, outputs, and cache
        - Model parameters: Architecture settings, input dimensions
        - Training parameters: Learning rates, epochs, batch sizes
        - Evaluation parameters: Thresholds, metrics, validation settings
        - Preprocessing: Normalization, augmentation settings

    Example:
        >>> config = get_default_config()
        >>> config['NUM_EPOCHS'] = 300
        >>> config['BATCH_SIZE'] = 16
    """
    # Verify PyTorch version
    try:
        if torch.__version__ < "1.13.0":
            print(f"Warning: PyTorch version {torch.__version__} may be older than recommended (>=1.13.0).")
    except ImportError:
        raise ImportError("PyTorch is required but not installed.")

    config = {
        # ===== Data Paths =====
        "IMAGE_DIR": "/root/autodl-tmp/data",
        "LABEL_DIR": "/root/autodl-tmp/label",
        "OUTPUT_DIR": "/root/autodl-tmp/results_refactored",
        "CACHE_DIR": "/root/autodl-tmp/cache_heatmap_refactored",

        # ===== Anatomical Structure Labels =====
        "HEATMAP_LABEL_MAP": [2, 3, 6, 7],
        "MSP_REQUIRED_LABELS": [2, 3, 6, 7],
        "KP_REQUIRED_LABELS": [4, 5],
        "KP_TO_PARENT_MAPPING": {4: 2, 5: 3},
        "STRUCTURE_LABELS": [2, 3, 6, 7],
        "EVAL_MAX_SAMPLES_PER_CLASS": 100,

        # ===== Gating Configuration =====
        "ENABLE_STRUCTURE_GATE": False,
        "AND_GATE_THRESHOLD": 0.35,
        "GATE_DEBUG": True,

        # ===== Model Architecture =====
        "IMAGE_SIZE": (512, 512),
        "SAGITTAL_AXIS": 2,
        "IN_CHANNELS": 1,

        # ===== Training Hyperparameters =====
        "NUM_EPOCHS": 400,
        "STAGE1_EPOCHS": 200,
        "STAGE2_EPOCHS": 200,
        "EARLY_STOPPING_PATIENCE": 50,
        "KFOLD_SPLITS": 5,
        "BATCH_SIZE": 8,
        "LEARNING_RATE": 2e-4,
        "WEIGHT_DECAY": 1e-5,
        "DEVICE": "cuda" if torch.cuda.is_available() else "cpu",
        "NUM_WORKERS": min(os.cpu_count() or 4, 8),

        # ===== Loss Function Weights =====
        "LAMBDA_REG": 1.5,
        "LAMBDA_CLS": 1.5,
        "LAMBDA_REG_S2": 0.3,
        "LAMBDA_CLS_S2": 1.0,
        "WEIGHT_DECAY_CLS": 1e-5,
        "OUTSIDE_BRAIN_PENALTY": 0.3,
        "FOCAL_GAMMA_REG": 2.5,
        "FOCAL_ALPHA_REG": 0.3,

        # ===== Thresholds =====
        "SLICE_CLS_THRESHOLD": 0.5,
        "MSP_CONFIDENCE_THRESHOLD": 0.5,
        "FEATURE_THRESHOLDS": [0.3, 0.5, 0.7, 0.9],
        "SLICE_THRESHOLD": 0.3,
        "CASE_THRESHOLD_DEFAULT": 0.5,
        "CASE_THRESHOLD_OPTIMAL": None,
        "MSP_SLICE_TOLERANCE": 2,
        "SAMPLES_PER_CASE": 3,

        # ===== Data Preprocessing =====
        "PERCENTILE_CLIP_LOW": 1,
        "PERCENTILE_CLIP_HIGH": 99,
        "BRAIN_MASK_INTENSITY_THRESHOLD": 0.01,
        "GENERATE_BRAIN_MASK_FROM_IMAGE": True,

        # ===== Data Splitting =====
        "TRAIN_RATIO": 0.8,
        "SPLIT_SEED": 42,

        # ===== Threshold Optimization =====
        "THRESHOLD_SPEC_WEIGHT": 0.5,
        "THRESHOLD_SPEC_MIN": 0.6,
        "THRESHOLD_SENS_MIN": 0.6,
        "THRESHOLD_METRIC": "youden",
        "CASE_SENS_MIN": 0.50,
        "AND_GATE_SENS_FLOOR": 0.70,

        # ===== Evaluation Parameters =====
        "EVAL_SPEC_WEIGHT": 0.7,
        "EVAL_SPEC_MIN": 0.6,
        "EVAL_SENS_MIN": 0.6,

        # ===== Advanced Features =====
        "USE_ATTENTION_AGG": True,
        "CONF_THRESHOLD_MC": 0.7,
        "GATE_LEARNABLE": True,
        "LAMBDA_CONSISTENCY": 0.2,
        "LAMBDA_GATE": 0.1,
        "ATTN_AGG_LR": 5e-4,
        "ATTN_AGG_DROPOUT": 0.1,
    }

    # Create label-to-channel mapping
    config["LABEL_TO_CHANNEL"] = {
        label: ch for ch, label in enumerate(config["HEATMAP_LABEL_MAP"])
    }

    return config
