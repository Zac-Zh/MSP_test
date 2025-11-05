"""
Input/Output utilities for file management and caching.

Provides functions for cache path generation, file pairing, and
data discovery in medical imaging datasets.
"""

import hashlib
import re
from pathlib import Path
from typing import Optional, List, Dict
from tqdm import tqdm


def get_cache_path(img_path_str: str, cache_dir_str: str, config_str_for_hash: str = "") -> Path:
    """
    Generates a unique cache file path using MD5 hashing.

    Creates deterministic cache filenames based on original file path
    and configuration parameters, enabling efficient reuse of preprocessed data.

    Args:
        img_path_str: Original image file path
        cache_dir_str: Directory for cache storage
        config_str_for_hash: Configuration string to include in hash computation

    Returns:
        Path object pointing to cache file with .npz extension

    Example:
        >>> cache_path = get_cache_path("/data/scan001.nii.gz", "/cache", "")
        >>> print(cache_path)
        /cache/scan001_a1b2c3d4e5f6g7h8.npz
    """
    cache_key_input = f"{img_path_str}_{config_str_for_hash}"
    cache_key = hashlib.md5(cache_key_input.encode()).hexdigest()[:16]
    img_stem = Path(img_path_str).stem.replace(".nii", "")
    return Path(cache_dir_str) / f"{img_stem}_{cache_key}.npz"


def find_nifti_pairs(img_dir: str, ann_dir: str, log_file: Optional[str] = None, config=None) -> List[Dict]:
    """
    Pairs NIfTI image files with their corresponding annotation files.

    This function implements robust file pairing strategies to handle various
    naming conventions commonly found in medical imaging datasets. It supports:
    - Exact filename matching
    - Common suffix patterns (_label, _seg, _labels, etc.)
    - Marker removal (removing _t1, _t2, _image, etc.)
    - Numeric identifier matching

    Args:
        img_dir: Directory containing image volumes
        ann_dir: Directory containing annotation/label volumes
        log_file: Optional log file path
        config: Configuration dictionary (uses default if None)

    Returns:
        List of dictionaries, each containing:
            - 'image': Path to image file
            - 'label': Path to label file (or None)
            - 'mask': Path to mask file (currently None)
            - 'id': Case identifier
            - 'has_msp': Boolean indicating if MSP is present
            - 'msp_slice_idx': Index of MSP slice (-1 if absent)

    Example:
        >>> pairs = find_nifti_pairs("/data/images", "/data/labels")
        >>> print(f"Found {len(pairs)} image-label pairs")
    """
    from utils.logging_utils import log_message
    from utils.msp_utils import get_msp_index
    from data.loaders import load_nifti_data_cached
    from data.preprocessing import preprocess_and_cache

    if config is None:
        from config import get_default_config
        config = get_default_config()

    pairs = []
    img_dir_path = Path(img_dir)
    ann_dir_path = Path(ann_dir)

    # Find all NIfTI files
    all_img_niftis = list(img_dir_path.rglob("*.nii")) + list(img_dir_path.rglob("*.nii.gz"))
    all_ann_niftis = list(ann_dir_path.rglob("*.nii")) + list(ann_dir_path.rglob("*.nii.gz"))

    if log_file:
        log_message(f"[DEBUG] Image files found: {len(all_img_niftis)}", log_file)
        log_message(f"[DEBUG] Label files found: {len(all_ann_niftis)}", log_file)

    # Create annotation file map
    ann_files_map = {}
    for ann_file in all_ann_niftis:
        if ann_file.name.endswith('.nii.gz'):
            base_name = ann_file.name[:-7]
        elif ann_file.name.endswith('.nii'):
            base_name = ann_file.name[:-4]
        else:
            continue
        ann_files_map[base_name.lower()] = ann_file

    # Exclude annotation files from image list
    annotation_suffixes = ["_labels", "_label", "_seg", "_segmentation", "_anno", "_mask", "_gt"]
    potential_image_files = [
        fp for fp in all_img_niftis
        if not any(fp.stem.lower().endswith(s) for s in annotation_suffixes)
    ]

    used_labels = set()
    matched_count = 0
    non_msp_count = 0
    strategy_stats = {"exact": 0, "with_suffix": 0, "cleaned": 0, "exact_numeric": 0, "non_msp": 0, "other": 0}

    for img_path in tqdm(potential_image_files, desc="Pairing files"):
        if img_path.name.endswith('.nii.gz'):
            img_base = img_path.name[:-7]
        elif img_path.name.endswith('.nii'):
            img_base = img_path.name[:-4]
        else:
            continue

        found_label = None
        strategy_used = None

        # Strategy 1: Exact match
        exact_candidate_key = img_base.lower()
        if exact_candidate_key in ann_files_map and str(ann_files_map[exact_candidate_key]) not in used_labels:
            found_label = ann_files_map[exact_candidate_key]
            strategy_used = "exact"

        # Strategy 2: Image base + common suffix
        if not found_label:
            for suffix in annotation_suffixes:
                candidate_key = f"{img_base}{suffix}".lower()
                if candidate_key in ann_files_map and str(ann_files_map[candidate_key]) not in used_labels:
                    found_label = ann_files_map[candidate_key]
                    strategy_used = "with_suffix"
                    break

        # Strategy 3: Cleaned base name
        if not found_label:
            img_markers = ["_t1", "_t2", "_flair", "_image", "_img", "_brain", "_0000", "_0001"]
            img_base_lower = img_base.lower()
            for marker in img_markers:
                if marker in img_base_lower:
                    clean_base = img_base_lower.replace(marker, "")
                    if clean_base in ann_files_map and str(ann_files_map[clean_base]) not in used_labels:
                        found_label = ann_files_map[clean_base]
                        strategy_used = "cleaned"
                        break
                    for suffix in annotation_suffixes:
                        candidate_key = f"{clean_base}{suffix}"
                        if candidate_key in ann_files_map and str(ann_files_map[candidate_key]) not in used_labels:
                            found_label = ann_files_map[candidate_key]
                            strategy_used = "cleaned"
                            break
                    if found_label:
                        break

        # Strategy 4: Numeric matching
        if not found_label:
            img_numeric_parts = re.findall(r'\d+(?:[_\-.]\d+)*', img_base)
            if img_numeric_parts:
                main_numeric_img = img_numeric_parts[0]
                for ann_base_name_map, ann_path in ann_files_map.items():
                    if str(ann_path) in used_labels:
                        continue

                    ann_numeric_parts = re.findall(r'\d+(?:[_\-.]\d+)*', ann_base_name_map)
                    if ann_numeric_parts and ann_numeric_parts[0] == main_numeric_img:
                        ann_stripped = ann_base_name_map
                        for suffix in annotation_suffixes:
                            if ann_base_name_map.endswith(suffix):
                                ann_stripped = ann_base_name_map[:-len(suffix)]
                                break

                        if img_base.lower() == ann_stripped:
                            found_label = ann_path
                            strategy_used = "exact_numeric"
                            break

        # Process paired data
        if found_label:
            try:
                label_vol = load_nifti_data_cached(str(found_label), is_label=True)
                if label_vol is not None:
                    filtered_label_vol = label_vol.copy()
                    filtered_label_vol[(filtered_label_vol == 4) | (filtered_label_vol == 5)] = 0

                    msp_idx = get_msp_index(
                        filtered_label_vol,
                        config["SAGITTAL_AXIS"],
                        config["STRUCTURE_LABELS"]
                    )
                    has_msp = (msp_idx >= 0)
                else:
                    has_msp = False
                    msp_idx = -1

                pairs.append({
                    "image": str(img_path),
                    "label": str(found_label),
                    "mask": None,
                    "id": img_base,
                    "has_msp": has_msp,
                    "msp_slice_idx": msp_idx
                })
                used_labels.add(str(found_label))
                matched_count += 1
                if strategy_used:
                    strategy_stats[strategy_used] += 1
            except Exception as e:
                if log_file:
                    log_message(f"    Error processing label for {img_base}: {e}", log_file)
                continue
        else:
            # Cases without labels
            try:
                img_vol, _ = preprocess_and_cache(str(img_path), config["CACHE_DIR"], config, log_file)
                if img_vol is not None and img_vol.size > 0:
                    pairs.append({
                        "image": str(img_path),
                        "label": None,
                        "mask": None,
                        "id": img_base,
                        "has_msp": False,
                        "msp_slice_idx": -1
                    })
                    non_msp_count += 1
                    strategy_stats["non_msp"] += 1
            except Exception as e:
                if log_file:
                    log_message(f"    Error validating image for non-MSP case {img_base}: {e}", log_file)
                continue

    if log_file:
        log_message(
            f"[DEBUG] Successfully paired: {matched_count} with labels, {non_msp_count} without labels",
            log_file
        )
        log_message(f"[DEBUG] Total pairs: {len(pairs)}", log_file)
        if len(potential_image_files) > 0:
            log_message(f"[DEBUG] Pairing rate: {len(pairs) / len(potential_image_files) * 100:.1f}%", log_file)
        log_message(f"[DEBUG] Pairing strategy stats: {strategy_stats}", log_file)

    return pairs


def create_dir_with_permissions(path_obj: Path, mode: int = 0o755) -> Path:
    """
    Creates a directory with specific Unix permissions.

    Args:
        path_obj: Path to create
        mode: Unix permissions (default: 755)

    Returns:
        Created Path object
    """
    import os

    if not path_obj.exists():
        path_obj.mkdir(parents=True, exist_ok=False)
        try:
            os.chmod(path_obj, mode)
        except Exception as e:
            print(f"Warning: Could not chmod {path_obj} to {oct(mode)}: {e}")
    elif not path_obj.is_dir():
        raise FileExistsError(f"Path {path_obj} exists but is not a directory.")
    return path_obj
