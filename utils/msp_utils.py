"""
MSP (Midsagittal Plane) computation utilities.

Provides functions for determining the ground-truth MSP slice index
from anatomical label volumes.
"""

import numpy as np
from typing import Tuple, Optional


def get_msp_index(
    label_data_3d: np.ndarray,
    sagittal_axis: int = 2,
    structure_labels: Tuple[int, ...] = (1, 2, 5, 6),
    tolerance: float = 0.01
) -> int:
    """
    Computes the midsagittal plane (MSP) slice index from a 3D label volume.

    The MSP is identified as the slice where all specified anatomical structures
    are present simultaneously. This typically corresponds to the midline plane
    that divides the brain into left and right hemispheres.

    Algorithm:
        1. For each structure label, identify slices where it appears
        2. Find slices where ALL structures coexist
        3. Return median index among such slices
        4. If no slice contains all structures, return slice with maximum structure count

    Args:
        label_data_3d: 3D segmentation label volume (H, W, D)
        sagittal_axis: Axis along which sagittal slices are extracted (0, 1, or 2)
        structure_labels: Tuple of anatomical structure IDs to check
        tolerance: Numerical tolerance for label matching

    Returns:
        Integer index of the MSP slice, or -1 if no valid MSP is found

    Example:
        >>> label_vol = load_nifti_data("case001_labels.nii.gz", is_label=True)
        >>> msp_idx = get_msp_index(label_vol, sagittal_axis=2, structure_labels=(2, 3, 6, 7))
        >>> print(f"MSP slice: {msp_idx}")
    """
    if label_data_3d is None or label_data_3d.ndim != 3 or sagittal_axis >= 3:
        return -1

    try:
        # Determine axes to reduce (all except sagittal axis)
        axes_to_reduce = tuple(i for i in range(label_data_3d.ndim) if i != sagittal_axis)

        presence_mask_per_structure_per_slice = []
        for lab_code in structure_labels:
            # Check presence of label in each slice
            lab_present_on_slice = np.any(
                np.isclose(label_data_3d, lab_code, atol=tolerance),
                axis=axes_to_reduce
            )
            presence_mask_per_structure_per_slice.append(lab_present_on_slice)

        if not presence_mask_per_structure_per_slice:
            return -1

        # Stack presence masks: shape (num_structures, num_slices)
        combined_presence = np.stack(presence_mask_per_structure_per_slice, axis=0)

        # Find slices where ALL structures are present
        all_structures_present_on_slice = np.all(combined_presence, axis=0)
        msp_indices = np.where(all_structures_present_on_slice)[0]

        if msp_indices.size > 0:
            # Return median index of valid slices
            return int(np.median(msp_indices))
        else:
            # Fallback: find slice with maximum number of structures
            num_structures_per_slice = np.sum(combined_presence, axis=0)
            max_structures_found = np.max(num_structures_per_slice)

            if max_structures_found > 0:
                candidate_indices = np.where(num_structures_per_slice == max_structures_found)[0]
                return int(np.median(candidate_indices)) if candidate_indices.size > 0 else -1
            else:
                return -1
    except Exception:
        return -1
