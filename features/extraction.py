"""
Feature extraction from heatmap predictions.

Extracts comprehensive statistical and geometric features from model-predicted
heatmaps for use in meta-classifier training.
"""

import math
import numpy as np
import cv2
import scipy.ndimage as ndi
from typing import List


def extract_heatmap_features(heatmap_logits: np.ndarray,
                             brain_mask: np.ndarray = None,
                             thresholds: List[float] = None,
                             expected_channels: int = 4,
                             config: dict = None) -> np.ndarray:
    """
    Enhanced feature extraction with robust error handling.

    IMPROVEMENTS:
    1. Better handling of edge cases
    2. Consistent dimension output
    3. More robust NaN handling
    """
    if thresholds is None:
        thresholds = [0.1, 0.3, 0.5, 0.7]

    # Calculate expected dimension
    if config is not None:
        expected_channels = len(config["HEATMAP_LABEL_MAP"])
        heatmap_label_map = config["HEATMAP_LABEL_MAP"]
    else:
        heatmap_label_map = list(range(expected_channels))

    n_corr_feats = expected_channels * (expected_channels - 1) // 2 if expected_channels > 1 else 0
    expected_dim = 2 + expected_channels * (6 + len(thresholds) + 2) + n_corr_feats + 2 + 2

    # Validate input
    if not isinstance(heatmap_logits, np.ndarray) or heatmap_logits.ndim != 3:
        return np.zeros(expected_dim, dtype=np.float32)

    num_input_channels, H, W = heatmap_logits.shape

    # Adjust channels if needed
    if num_input_channels != expected_channels:
        adjusted_logits = np.zeros((expected_channels, H, W), dtype=heatmap_logits.dtype)
        copy_count = min(num_input_channels, expected_channels)
        adjusted_logits[:copy_count] = heatmap_logits[:copy_count]
        heatmap_logits_to_process = adjusted_logits
    else:
        heatmap_logits_to_process = heatmap_logits

    # Convert to probabilities
    probs = 1 / (1 + np.exp(-heatmap_logits_to_process))

    features = []

    # Prepare brain mask
    if brain_mask is not None:
        if brain_mask.shape != (H, W):
            brain_mask_resized = cv2.resize(brain_mask.astype(np.float32), (W, H),
                                           interpolation=cv2.INTER_NEAREST)
            brain_mask_bool = brain_mask_resized.astype(bool)
        else:
            brain_mask_bool = brain_mask.astype(bool)

        # CRITICAL: Check brain mask validity
        if brain_mask_bool.sum() < 10:
            # Return zeros if brain mask is too small
            return np.zeros(expected_dim, dtype=np.float32)
    else:
        brain_mask_bool = np.ones((H, W), dtype=bool)

    # === Foreground features (2 features) ===
    if brain_mask is not None:
        brain_ratio = brain_mask_bool.mean()
        features.append(float(brain_ratio))

        total_activation = probs.sum()
        brain_activation = (probs * brain_mask_bool).sum()
        activation_ratio = brain_activation / (total_activation + 1e-9)
        features.append(float(activation_ratio))
    else:
        features.extend([1.0, 1.0])

    # Collect centroids for alignment calculation
    centroid_x_list = []
    centroid_y_list = []

    # === Per-channel features ===
    for c in range(expected_channels):
        channel_probs = probs[c]
        masked_probs = channel_probs[brain_mask_bool]

        if masked_probs.size == 0 or masked_probs.max() == 0:
            # Add (6 + len(thresholds) + 2) zero features
            features.extend([0.0] * (6 + len(thresholds) + 2))
            centroid_x_list.append(np.nan)
            centroid_y_list.append(np.nan)
            continue

        # Basic stats (6 features)
        features.append(float(masked_probs.max()))
        features.append(float(masked_probs.mean()))
        features.append(float(np.median(masked_probs)))
        features.append(float(masked_probs.std()))
        features.append(float(np.percentile(masked_probs, 75)))
        features.append(float(np.percentile(masked_probs, 90)))

        # Threshold features
        for thresh_val in thresholds:
            above_thresh_ratio = (masked_probs > thresh_val).mean()
            features.append(float(above_thresh_ratio))

        # Geometric features (2 features)
        binary_mask = (channel_probs > 0.5) & brain_mask_bool
        area_ratio = binary_mask.sum() / brain_mask_bool.sum() if brain_mask_bool.sum() > 0 else 0.0
        features.append(float(area_ratio))

        # Compactness
        compactness = 0.0
        if binary_mask.sum() > 0:
            try:
                kernel = np.ones((3, 3), np.uint8)
                eroded = cv2.erode(binary_mask.astype(np.uint8), kernel, iterations=1)
                perimeter_mask = binary_mask.astype(np.uint8) ^ eroded
                perimeter_pixels = perimeter_mask.sum()

                if perimeter_pixels > 0:
                    area_pixels = binary_mask.sum()
                    compactness = (4 * math.pi * area_pixels) / (perimeter_pixels ** 2)
                    compactness = min(compactness, 1.0)
            except:
                compactness = 0.0
        features.append(float(compactness))

        # Calculate centroid
        if binary_mask.sum() > 0:
            labeled_array, num_features = ndi.label(binary_mask.astype(int))
            if num_features > 0:
                component_sizes = [np.sum(labeled_array == i) for i in range(1, num_features + 1)]
                largest_component_label = np.argmax(component_sizes) + 1
                largest_component_mask = (labeled_array == largest_component_label)

                if largest_component_mask.sum() > 0:
                    y_coords, x_coords = np.where(largest_component_mask)
                    cx = float(x_coords.mean()) / W
                    cy = float(y_coords.mean()) / H
                else:
                    cx = cy = np.nan
            else:
                cx = cy = np.nan
        else:
            cx = cy = np.nan

        centroid_x_list.append(cx)
        centroid_y_list.append(cy)

    # === Cross-channel correlation ===
    all_probs_flat = probs.reshape(expected_channels, -1)[:, brain_mask_bool.flatten()]

    if all_probs_flat.size > 0 and expected_channels > 1:
        try:
            corr_matrix = np.corrcoef(all_probs_flat)
            if corr_matrix.ndim == 2:
                triu_indices = np.triu_indices(expected_channels, k=1)
                correlations = corr_matrix[triu_indices]
                features.extend(np.nan_to_num(correlations, nan=0.0).tolist())
            else:
                features.extend([0.0] * n_corr_feats)
        except:
            features.extend([0.0] * n_corr_feats)

        features.append(float(all_probs_flat.max()))
        features.append(float(all_probs_flat.mean()))
    else:
        features.extend([0.0] * n_corr_feats)
        features.extend([0.0, 0.0])

    # === Global alignment (2 features) ===
    valid_cx = [cx for cx in centroid_x_list if not np.isnan(cx)]
    valid_cy = [cy for cy in centroid_y_list if not np.isnan(cy)]

    centroid_std_x = float(np.std(valid_cx)) if len(valid_cx) > 1 else 0.0
    centroid_std_y = float(np.std(valid_cy)) if len(valid_cy) > 1 else 0.0

    features.append(centroid_std_x)
    features.append(centroid_std_y)

    # === Final validation ===
    features_array = np.array(features, dtype=np.float32)

    if len(features_array) != expected_dim:
        # Pad or truncate to match expected dimension
        padded_features = np.zeros(expected_dim, dtype=np.float32)
        copy_len = min(len(features_array), expected_dim)
        padded_features[:copy_len] = features_array[:copy_len]
        return padded_features

    # Replace any remaining NaN/Inf
    features_array = np.nan_to_num(features_array, nan=0.0, posinf=0.0, neginf=0.0)

    return features_array
