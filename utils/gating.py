"""
Structure gating utilities for MSP detection.

Provides gating logic to filter slices based on structure presence.
"""

import numpy as np


def four_structure_and_gate_check(pred_heatmaps_probs, config, threshold=None):
    """
    Four-structure AND gate check, consistent between training and inference.

    Checks if all four required MSP structures meet a minimum confidence threshold.
    This prevents false positives from slices that only partially show the structures.

    Args:
        pred_heatmaps_probs: Predicted heatmap probabilities array (C, H, W)
        config: Configuration dictionary with structure mapping
        threshold: Optional threshold override (default from config["AND_GATE_THRESHOLD"])

    Returns:
        bool: True if all structures pass the threshold, False otherwise
    """
    # Use the threshold from config if not provided
    if threshold is None:
        threshold = config.get("AND_GATE_THRESHOLD", 0.25)

    # Check if gating is enabled
    if not config.get("ENABLE_STRUCTURE_GATE", True):
        if config.get("GATE_DEBUG", False):
            print("  Four-structure AND gate is disabled, passing directly")
        return True

    label_to_channel = config["LABEL_TO_CHANNEL"]
    msp_required_labels = config["MSP_REQUIRED_LABELS"]

    if config.get("GATE_DEBUG", False):
        print(f"DEBUG: AND gate check started (threshold={threshold})")
        print(f"  Prediction heatmap shape: {pred_heatmaps_probs.shape}")
        print(f"  Required labels: {msp_required_labels}")

    structure_max_probs = []

    for label in msp_required_labels:
        # Ensure label is an integer
        label_int = int(label)

        if label_int in label_to_channel:
            channel_idx = label_to_channel[label_int]
            if channel_idx < pred_heatmaps_probs.shape[0]:
                max_prob = np.max(pred_heatmaps_probs[channel_idx])
                structure_max_probs.append(max_prob)
                if config.get("GATE_DEBUG", False):
                    print(f"  Label {label_int} -> Channel {channel_idx}: Max probability = {max_prob:.4f}")
            else:
                if config.get("GATE_DEBUG", False):
                    print(f"  Warning: Label {label_int} corresponds to channel {channel_idx}, which is out of range")
                structure_max_probs.append(0.0)
        else:
            if config.get("GATE_DEBUG", False):
                print(f"  Warning: Label {label_int} not found in mapping")
            structure_max_probs.append(0.0)

    # AND gate logic: all structures must meet the threshold
    min_prob = min(structure_max_probs) if structure_max_probs else 0.0
    passed = min_prob >= threshold

    if config.get("GATE_DEBUG", False):
        print(f"  Structure probabilities: {structure_max_probs}")
        print(f"  Minimum probability: {min_prob:.4f}, Threshold: {threshold}")
        print(f"  AND gate result: {'Passed' if passed else 'Failed'}")

    return passed
