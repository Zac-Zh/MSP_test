"""
Keypoint detection functions for MSP analysis.

Provides functions for detecting anatomical keypoints from heatmap predictions.
"""

import numpy as np
from scipy.ndimage import distance_transform_edt, binary_dilation


def detect_two_keypoints(pred_hmaps: np.ndarray,
                         gt_slice: np.ndarray | None,
                         img_slice_orig: np.ndarray,
                         spacing_xy: tuple[float, float],
                         heatmap_size: tuple[int, int],
                         orig_size: tuple[int, int],
                         config: dict = None) -> dict:
    """
    Detects two keypoints (4 and 5) with corrected label mapping and coordinate transformations.

    This function:
    1. Maps keypoints 4,5 to their parent structures 2,3 using config
    2. Finds peaks on heatmap and optionally snaps them to structure masks
    3. Transforms coordinates: heatmap -> original image pixels -> millimeters
    4. Calculates inter-keypoint distance in mm
    5. Processes ground truth if available

    Args:
        pred_hmaps: Predicted heatmaps array (C, H, W)
        gt_slice: Ground truth label slice (H, W) or None
        img_slice_orig: Original image slice for reference
        spacing_xy: Pixel spacing in mm (sx, sy)
        heatmap_size: Model output size (H_model, W_model)
        orig_size: Original image size (H_orig, W_orig)
        config: Configuration dictionary with label mappings

    Returns:
        dict: Keypoint detection results with:
            - predicted_keypoints, ground_truth_keypoints: Lists of (x,y) tuples
            - pred_xy4, pred_xy5, pred_d_mm: Predicted keypoint coordinates and distance
            - gt_xy4, gt_xy5, gt_d_mm: Ground truth keypoint coordinates and distance
            - spacing_used, scale_factors: Transformation metadata
            - debug_info: Detailed debugging information
    """
    H_model, W_model = heatmap_size
    H_orig, W_orig = orig_size
    sx, sy = spacing_xy

    print(f"DEBUG: Keypoint detection started")
    print(f"  Heatmap shape: {pred_hmaps.shape}")
    print(f"  Heatmap size: {heatmap_size}")
    print(f"  Original size: {orig_size}")
    print(f"  Pixel spacing: {spacing_xy}")

    # Scaling factors: from heatmap coordinates to original image coordinates
    scale_x = W_orig / W_model
    scale_y = H_orig / H_model
    print(f"  Scaling factors: ({scale_x:.3f}, {scale_y:.3f})")

    THR_MASK = 0.45

    def _snap_to_mask(y, x, mask):
        """If (y,x) is outside the mask, project it back to the nearest pixel."""
        if mask.shape[0] <= y or mask.shape[1] <= x or y < 0 or x < 0:
            return np.nan, np.nan
        if mask[y, x]:
            return y, x
        dist, inds = distance_transform_edt(~mask, return_indices=True)
        if dist.size == 0:
            return np.nan, np.nan
        yy, xx = inds[:, y, x]
        return int(yy), int(xx)

    def _mm2px(pt_mm):
        """Converts millimeter coordinates back to pixel coordinates at the model's resolution."""
        if np.isnan(pt_mm[0]) or np.isnan(pt_mm[1]):
            return (np.nan, np.nan)
        return (pt_mm[0] / (sx * scale_x) if sx and scale_x else np.nan,
                pt_mm[1] / (sy * scale_y) if sy and scale_y else np.nan)

    # Correctly handle keypoint label mapping
    if config is None:
        print("  Warning: No config provided, using fallback logic")
        # Fallback: assume the first two channels are structures 2,3 (this might be incorrect)
        if pred_hmaps.shape[0] >= 2:
            p2, p3 = pred_hmaps[0], pred_hmaps[1]
            mask2 = mask3 = None
        else:
            p2 = p3 = None
            mask2 = mask3 = None
        # Keypoint detection: since 4,5 are not in the heatmap, we infer from parent structures
        p4 = p2  # keypoint 4 belongs to structure 2
        p5 = p3  # keypoint 5 belongs to structure 3
    else:
        # Use the correct mapping from config
        label_to_channel = config.get("LABEL_TO_CHANNEL", {})
        kp_to_parent = config.get("KP_TO_PARENT_MAPPING", {4: 2, 5: 3})

        print(f"  Label to channel map: {label_to_channel}")
        print(f"  Keypoint to parent map: {kp_to_parent}")

        p2 = p3 = p4 = p5 = None
        mask2 = mask3 = None

        # Get the heatmaps of the parent structures
        for parent_label in [2, 3]:
            parent_label_int = int(parent_label)
            if parent_label_int in label_to_channel:
                channel_idx = label_to_channel[parent_label_int]
                if channel_idx < pred_hmaps.shape[0]:
                    parent_hmap = pred_hmaps[channel_idx]
                    if parent_label_int == 2:
                        p2 = parent_hmap
                        mask2 = binary_dilation(parent_hmap > THR_MASK, iterations=1)
                        print(
                            f"  Structure 2 -> Channel {channel_idx}, Mask pixels: {mask2.sum() if mask2 is not None else 0}")
                    elif parent_label_int == 3:
                        p3 = parent_hmap
                        mask3 = binary_dilation(parent_hmap > THR_MASK, iterations=1)
                        print(
                            f"  Structure 3 -> Channel {channel_idx}, Mask pixels: {mask3.sum() if mask3 is not None else 0}")
            else:
                print(f"  Warning: Parent structure {parent_label_int} not found in map")

        # Keypoint 4 uses structure 2's heatmap, keypoint 5 uses structure 3's
        p4 = p2
        p5 = p3

    # Check if heatmaps were successfully obtained
    if p4 is None or p5 is None:
        print(f"  Error: Keypoint heatmap acquisition failed (p4: {p4 is not None}, p5: {p5 is not None})")
        return {
            'predicted_keypoints': [(np.nan, np.nan)] * 5,
            'ground_truth_keypoints': [(np.nan, np.nan)] * 5,
            'pred_xy4': (np.nan, np.nan), 'pred_xy5': (np.nan, np.nan), 'pred_d_mm': np.nan,
            'gt_xy4': (np.nan, np.nan), 'gt_xy5': (np.nan, np.nan), 'gt_d_mm': np.nan,
            'spacing_used': spacing_xy,
            'scale_factors': (scale_x, scale_y),
            'error': 'Failed to get keypoint heatmaps'
        }

    # Find peaks on the heatmap
    y4, x4 = np.unravel_index(np.argmax(p4), p4.shape)
    y5, x5 = np.unravel_index(np.argmax(p5), p5.shape)

    print(f"  Original peak locations: p4=({y4}, {x4}), p5=({y5}, {x5})")

    # Constrain to the corresponding structure mask if available
    if mask2 is not None and mask2.sum() > 0:
        y4_snapped, x4_snapped = _snap_to_mask(y4, x4, mask2)
        if not np.isnan(y4_snapped):
            y4, x4 = y4_snapped, x4_snapped
            print(f"  p4 location after constraint: ({y4}, {x4})")

    if mask3 is not None and mask3.sum() > 0:
        y5_snapped, x5_snapped = _snap_to_mask(y5, x5, mask3)
        if not np.isnan(y5_snapped):
            y5, x5 = y5_snapped, x5_snapped
            print(f"  p5 location after constraint: ({y5}, {x5})")

    # Correct coordinate transformation
    # Step 1: Heatmap coords -> Original image coords (pixels)
    x4_orig = x4 * scale_x
    y4_orig = y4 * scale_y
    x5_orig = x5 * scale_x
    y5_orig = y5 * scale_y

    print(f"  Original image coordinates: p4=({x4_orig:.1f}, {y4_orig:.1f}), p5=({x5_orig:.1f}, {y5_orig:.1f})")

    # Step 2: Pixel coords -> Millimeter coords
    pred4_mm = (x4_orig * sx, y4_orig * sy)
    pred5_mm = (x5_orig * sx, y5_orig * sy)

    print(
        f"  Millimeter coordinates: p4=({pred4_mm[0]:.2f}, {pred4_mm[1]:.2f}), p5=({pred5_mm[0]:.2f}, {pred5_mm[1]:.2f})")

    # Calculate predicted distance
    pred_d = float(np.linalg.norm(np.subtract(pred4_mm, pred5_mm)))
    print(f"  Predicted distance: {pred_d:.2f} mm")

    # Process GT labels
    if gt_slice is not None:
        gt4_coords = np.column_stack(np.where(gt_slice == 4))
        gt5_coords = np.column_stack(np.where(gt_slice == 5))

        if len(gt4_coords) > 0 and len(gt5_coords) > 0:
            gt4_center = gt4_coords.mean(axis=0)  # (y, x)
            gt5_center = gt5_coords.mean(axis=0)  # (y, x)

            # GT coordinate transformation: original image pixel -> mm
            gt4_mm = (gt4_center[1] * sx, gt4_center[0] * sy)  # (x, y) in mm
            gt5_mm = (gt5_center[1] * sx, gt5_center[0] * sy)  # (x, y) in mm
            gt_d = float(np.linalg.norm(np.subtract(gt4_mm, gt5_mm)))

            print(
                f"  GT millimeter coordinates: gt4=({gt4_mm[0]:.2f}, {gt4_mm[1]:.2f}), gt5=({gt5_mm[0]:.2f}, {gt5_mm[1]:.2f})")
            print(f"  GT distance: {gt_d:.2f} mm")
        else:
            gt4_mm = gt5_mm = (np.nan, np.nan)
            gt_d = np.nan
            print(f"  GT label missing")
    else:
        gt4_mm = gt5_mm = (np.nan, np.nan)
        gt_d = np.nan
        print(f"  No GT slice")

    # Assemble pixel-space keypoint arrays (for visualization)
    predicted_keypoints = []
    ground_truth_keypoints = []

    # Add 5 keypoints (to be compatible with draw_keypoints_overlay expectations)
    for i in range(5):
        if i == 3:  # 4th point (0-indexed)
            predicted_keypoints.append(_mm2px(pred4_mm))
            ground_truth_keypoints.append(_mm2px(gt4_mm))
        elif i == 4:  # 5th point
            predicted_keypoints.append(_mm2px(pred5_mm))
            ground_truth_keypoints.append(_mm2px(gt5_mm))
        else:
            predicted_keypoints.append((np.nan, np.nan))
            ground_truth_keypoints.append((np.nan, np.nan))

    print(f"  Final results:")
    print(f"    Predicted distance: {pred_d:.2f} mm")
    print(f"    GT distance: {gt_d:.2f} mm" if not np.isnan(gt_d) else "    GT distance: N/A")

    return {
        'predicted_keypoints': predicted_keypoints,
        'ground_truth_keypoints': ground_truth_keypoints,
        'pred_xy4': pred4_mm, 'pred_xy5': pred5_mm, 'pred_d_mm': pred_d,
        'gt_xy4': gt4_mm, 'gt_xy5': gt5_mm, 'gt_d_mm': gt_d,
        'spacing_used': spacing_xy,
        'scale_factors': (scale_x, scale_y),
        'mask_applied': mask2 is not None and mask3 is not None,
        'labels_used': config.get("KP_REQUIRED_LABELS", [4, 5]) if config else [4, 5],
        'parent_structures_used': [2, 3],
        'debug_info': {
            'heatmap_peaks': {'p4': (y4, x4), 'p5': (y5, x5)},
            'orig_coords': {'p4': (x4_orig, y4_orig), 'p5': (x5_orig, y5_orig)},
            'mm_coords': {'p4': pred4_mm, 'p5': pred5_mm}
        }
    }
