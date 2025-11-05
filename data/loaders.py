"""
NIfTI data loading utilities.

Provides functions for loading medical imaging data in NIfTI format with
caching support for improved performance during training.
"""

import numpy as np
import nibabel as nib
from pathlib import Path
from functools import lru_cache
from typing import Optional


def load_nifti_data(path: Path, *, is_label: bool = False) -> Optional[np.ndarray]:
    """
    Loads NIfTI medical imaging data from disk.
    
    Args:
        path: Path to the NIfTI file (.nii or .nii.gz)
        is_label: If True, rounds values to integers for segmentation labels
    
    Returns:
        3D numpy array containing the volume data, or None if loading fails
    """
    try:
        img = nib.load(str(path))
        data = img.get_fdata(dtype=np.float32)
        if is_label:
            data = np.rint(data).astype(np.int16)
        return data
    except Exception as e:
        return None


@lru_cache(maxsize=128)
def load_nifti_data_cached(path_str: str, is_label: bool = False) -> Optional[np.ndarray]:
    """
    Cached version of NIfTI data loading for improved performance.
    
    Args:
        path_str: String path to the NIfTI file
        is_label: If True, treats as segmentation label
    
    Returns:
        3D numpy array or None if loading fails
    """
    return load_nifti_data(Path(path_str), is_label=is_label)
