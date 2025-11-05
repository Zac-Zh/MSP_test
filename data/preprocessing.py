"""
Image preprocessing utilities for MRI data.
"""

import numpy as np
import cv2
import hashlib
import nibabel as nib
import scipy.ndimage as ndi
import albumentations as A
from albumentations.pytorch import ToTensorV2
from pathlib import Path
from typing import Optional, Tuple, List
from utils.logging_utils import log_message
from utils.io_utils import get_cache_path



def extract_slice(volume_data, slice_idx, axis=2):
    """Extracts a 2D slice from a 3D volume."""
    if volume_data is None or volume_data.ndim != 3:
        return None
    try:
        if axis == 0:
            slice_2d = volume_data[slice_idx, :, :]
        elif axis == 1:
            slice_2d = volume_data[:, slice_idx, :]
        elif axis == 2:
            slice_2d = volume_data[:, :, slice_idx]
        else:  # Default to axis 2 if invalid axis provided
            slice_2d = volume_data[:, :, slice_idx]
        return slice_2d.copy()  # Return a copy to avoid modifying the original volume
    except IndexError:
        return None
    except Exception:  # Generic exception for other unforeseen errors
        return None



def normalize_slice(slice_data, config):
    """Normalizes a slice."""
    if slice_data is None or slice_data.size == 0:
        H, W = config["IMAGE_SIZE"]
        return np.zeros((H, W), dtype=np.float32)

    slice_data_float = slice_data.astype(np.float32)

    # Using all pixels for percentile calculation initially, can be refined if only foreground needed
    pixels_for_percentile = slice_data_float.flatten()
    if pixels_for_percentile.size == 0:  # Should be caught by slice_data.size == 0
        return np.full(slice_data_float.shape, 0.0, dtype=np.float32)

    min_val_overall, max_val_overall = np.min(pixels_for_percentile), np.max(pixels_for_percentile)
    if min_val_overall == max_val_overall:  # Handle constant images
        return np.zeros_like(slice_data_float, dtype=np.float32)

    p_low, p_high = config.get("PERCENTILE_CLIP_LOW", 1), config.get("PERCENTILE_CLIP_HIGH", 99)
    p_low_val, p_high_val = np.percentile(pixels_for_percentile, [p_low, p_high])

    # Ensure p_high_val > p_low_val for robust clipping
    if p_high_val <= p_low_val:
        p_low_val = min_val_overall
        p_high_val = max_val_overall
        if p_high_val <= p_low_val:  # If still equal (e.g. after rounding), add epsilon
            p_high_val = p_low_val + 1e-6

    slice_clipped = np.clip(slice_data_float, p_low_val, p_high_val)

    # Normalize based on the clipped range
    min_clip, max_clip = np.min(slice_clipped), np.max(slice_clipped)

    if min_clip == max_clip:  # If clipping results in a constant image
        normalized = np.zeros_like(slice_clipped, dtype=np.float32)
    else:
        normalized = (slice_clipped - min_clip) / (max_clip - min_clip)

    return np.nan_to_num(normalized, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)



def generate_brain_mask_from_image(image_slice, config):
    """Generates a brain mask from an image slice."""
    if image_slice is None:
        H, W = config["IMAGE_SIZE"]
        # Return a mask of ones if image_slice is None, assuming full area is relevant
        return np.ones((H, W), dtype=np.float32)

        # Assuming image_slice is already normalized [0, 1]
    intensity_threshold = config.get("BRAIN_MASK_INTENSITY_THRESHOLD", 0.01)  # Default was 0.08

    # Create mask based on intensity threshold
    intensity_mask = (image_slice > intensity_threshold).astype(np.float32)

    if intensity_mask.sum() > 0:
        # Morphological operations to clean up the mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        # Closing: Dilate then Erode - fills small holes
        closed_mask = cv2.morphologyEx(intensity_mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
        # Opening: Erode then Dilate - removes small objects/noise
        opened_mask = cv2.morphologyEx(closed_mask, cv2.MORPH_OPEN, kernel)
        final_mask = opened_mask.astype(np.float32)
    else:
        # If no pixels are above threshold, return an empty mask (or ones, depending on desired behavior)
        final_mask = np.zeros_like(image_slice, dtype=np.float32)

    return final_mask


# ===== Heatmap Generation Functions =====

def mask_to_distancemap(bin_mask, max_dist=None):
    """Converts a binary mask to a distance transform map."""
    # import scipy.ndimage as ndi # Moved to top-level imports
    dt_map = ndi.distance_transform_edt(bin_mask)
    if max_dist is None:
        max_dist = dt_map.max() if dt_map.max() > 0 else 1.0
    return (dt_map / max_dist).astype(np.float32)



def create_target_heatmap_with_distance_transform(
        label_slice: np.ndarray,
        structure_labels: list[int],
        output_shape_hw: tuple[int, int],
        brain_mask: np.ndarray = None,
        min_structure_size: int = 20
) -> np.ndarray:
    """
    Creates the target heatmap using a distance transform.
    This logic ensures that holes from labels 4 and 5 are properly filled.
    """
    H, W = output_shape_hw
    C = len(structure_labels)
    target = np.zeros((C, H, W), dtype=np.float32)

    if label_slice is None:
        return target

    # === Key step: Process labels 4,5 and fill holes ===
    label_slice_clean = label_slice.copy().astype(np.float32)

    # Step 1: Set labels 4 and 5 to 0
    mask_45 = (label_slice_clean == 4) | (label_slice_clean == 5)
    label_slice_clean[mask_45] = 0

    # Step 2: Fill holes in major structures that often contain smaller ones
    for target_label in [2, 3]:  # These structures typically contain 4,5
        if target_label in structure_labels:
            struct_mask = (label_slice_clean == target_label).astype(np.uint8)
            if struct_mask.sum() > 0:
                # Use morphological closing to fill small holes
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
                filled_mask = cv2.morphologyEx(struct_mask, cv2.MORPH_CLOSE, kernel)
                # Reassign the filled area to the structure's label
                label_slice_clean[filled_mask > 0] = target_label

    # Resize the processed labels
    lbl_resized = cv2.resize(label_slice_clean, (W, H), interpolation=cv2.INTER_NEAREST)

    # Brain mask processing
    if brain_mask is not None:
        if brain_mask.shape != (H, W):
            brain_mask_resized = cv2.resize(brain_mask.astype(np.float32), (W, H),
                                            interpolation=cv2.INTER_NEAREST)
        else:
            brain_mask_resized = brain_mask.astype(np.float32)
    else:
        brain_mask_resized = np.ones((H, W), dtype=np.float32)

    for ch, code in enumerate(structure_labels):
        region_mask = (lbl_resized == code).astype(np.float32)
        region_mask = region_mask * brain_mask_resized

        # Verify structure size
        structure_size = region_mask.sum()
        if structure_size < min_structure_size:
            continue

        if region_mask.any():
            # Calculate the structure's compactness
            coords = np.where(region_mask > 0)
            if len(coords[0]) > 0:
                y_min, y_max = coords[0].min(), coords[0].max()
                x_min, x_max = coords[1].min(), coords[1].max()
                bbox_area = (y_max - y_min + 1) * (x_max - x_min + 1)
                compactness = structure_size / bbox_area

                if compactness < 0.2:
                    continue

            # Use distance transform, ensuring no internal holes remain
            bin_mask = (region_mask > 0.5).astype(np.uint8)
            if bin_mask.sum() > 0:
                # Final fill for any remaining small holes
                kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
                bin_mask_filled = cv2.morphologyEx(bin_mask, cv2.MORPH_CLOSE, kernel_small)

                distance_map = mask_to_distancemap(bin_mask_filled)
                distance_map = distance_map * brain_mask_resized
                target[ch] = distance_map

    return target



def preprocess_and_cache(img_path_str, cache_dir_str, config, log_file=None):
    """Preprocesses and caches image data."""
    config_hash_str = ""  # Currently not using config for hash, can be extended
    cache_file = get_cache_path(img_path_str, cache_dir_str, config_hash_str)

    if cache_file.exists():
        try:
            cached_data = np.load(cache_file, allow_pickle=False)
            img_vol_cached = cached_data["img"]
            # To maintain consistency with original return, load ref (though not always used by caller)
            img_nii_ref = nib.load(img_path_str)
            return img_vol_cached, img_nii_ref
        except Exception as e:
            if log_file:
                log_message(f"CACHE ERROR: Failed to load {cache_file}, rebuilding. Error: {e}", log_file)
            try:
                cache_file.unlink(missing_ok=True)
            except OSError:
                pass

    img_nii_orig = nib.load(img_path_str)
    img_data_raw = img_nii_orig.get_fdata(dtype=np.float32)
    if img_data_raw is None:
        raise ValueError(f"img get_fdata is None for {img_path_str}")

    img_vol_processed = np.squeeze(img_data_raw)
    if img_vol_processed is None:  # Should not happen if img_data_raw is not None
        raise ValueError(f"Squeeze for image volume returned None for {img_path_str}")

    Path(cache_dir_str).mkdir(parents=True, exist_ok=True)
    try:
        np.savez_compressed(cache_file, img=img_vol_processed.astype(np.float32))
    except Exception as e_save:
        if log_file:
            log_message(f"CACHE SAVE ERROR: Failed to save {cache_file}. Error: {e_save}", log_file)

    return img_vol_processed, img_nii_orig


def get_transforms(config, is_train=True):
    H, W = config["IMAGE_SIZE"]

    if is_train:
        transform = A.Compose([
            A.Resize(H, W, interpolation=cv2.INTER_LINEAR),
            A.HorizontalFlip(p=0.5),
            # 修复：移除了不被识别的 'value' 参数
            A.Rotate(limit=15, border_mode=cv2.BORDER_CONSTANT, p=0.3),
            # 修复：移除了已废弃的 'alpha_affine' 参数
            A.ElasticTransform(
                alpha=100, sigma=10, p=0.2,
                interpolation=cv2.INTER_LINEAR, border_mode=cv2.BORDER_REFLECT_101
            ),
            A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.3),
            # 修复：移除了不被识别的 'var_limit' 参数，使用默认方差范围
            A.GaussNoise(p=0.2),
            ToTensorV2(),
        ], additional_targets={'target': 'mask', 'brain_mask': 'mask'})
    else:
        # 修复：为验证/测试集也添加 additional_targets，确保 image, target, brain_mask 尺寸对齐
        transform = A.Compose([
            A.Resize(H, W, interpolation=cv2.INTER_LINEAR),
            ToTensorV2(),
        ], additional_targets={'target': 'mask', 'brain_mask': 'mask'})

    return transform


def remap_small_structures_to_parent(label_slice):
    """
    Remaps small structures to their parent structures.
    This is like telling the system that small islands are part of the mainland.
    """
    if label_slice is None:
        return None

    remapped_slice = label_slice.copy()
    # Hierarchical relationship in medical images: 4 belongs to 2, 5 belongs to 3
    remapped_slice[remapped_slice == 4] = 2  # Merge small structure 4 into large structure 2
    remapped_slice[remapped_slice == 5] = 3  # Merge small structure 5 into large structure 3

    return remapped_slice
