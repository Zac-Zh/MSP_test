"""
Training helper functions for MSP detection.

Provides utilities for dataset preparation, model loading, and training support.
"""

import torch
from pathlib import Path
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from utils.logging_utils import log_message
from utils.io_utils import find_nifti_pairs
from models import UNetHeatmap, UNetWithCls, UNetWithDualHeads


def prepare_patient_grouped_datasets(config, log_file=None):
    """
    Prepares datasets grouped by patient, including non-MSP cases.
    """
    if log_file: log_message("Finding NIfTI pairs (including non-MSP cases)...", log_file)
    all_pairs = find_nifti_pairs(config["IMAGE_DIR"], config["LABEL_DIR"], log_file)

    if not all_pairs:
        raise ValueError("No NIfTI pairs found! Check IMAGE_DIR and LABEL_DIR.")

    # Separate cases with and without MSP
    msp_pairs = [p for p in all_pairs if p.get("has_msp", False)]
    non_msp_pairs = [p for p in all_pairs if not p.get("has_msp", False)]

    if log_file:
        log_message(f"Found {len(msp_pairs)} MSP pairs and {len(non_msp_pairs)} non-MSP pairs.", log_file)

    # Group pairs by patient ID
    patient_groups = {}
    for pair in all_pairs:
        patient_id = pair["id"]
        if patient_id not in patient_groups:
            patient_groups[patient_id] = []
        patient_groups[patient_id].append(pair)

    patient_ids_all = list(patient_groups.keys())
    if log_file: log_message(f"Found {len(patient_ids_all)} unique patient IDs.", log_file)
    if not patient_ids_all:
        raise ValueError("No patient groups formed. Check patient ID extraction logic.")

    # Split patient IDs for training and validation
    if len(patient_ids_all) < 2:
        raise ValueError(f"Not enough unique patients ({len(patient_ids_all)}) to perform a train/val split.")

    test_ratio = 1.0 - config["TRAIN_RATIO"]
    if not (0 < test_ratio < 1):
        if test_ratio == 0 and len(patient_ids_all) >= 1:
            train_patient_ids = patient_ids_all
            val_patient_ids = []
            log_message("Warning: TRAIN_RATIO is 1.0. Using all patients for training, 0 for validation.", log_file)
        elif test_ratio == 1 and len(patient_ids_all) >= 1:
            train_patient_ids = []
            val_patient_ids = patient_ids_all
            log_message("Warning: TRAIN_RATIO is 0.0. Using all patients for validation, 0 for training.", log_file)
        else:
            raise ValueError(
                f"Invalid TRAIN_RATIO ({config['TRAIN_RATIO']}) for splitting {len(patient_ids_all)} patients.")
    else:
        train_patient_ids, val_patient_ids = train_test_split(
            patient_ids_all,
            test_size=test_ratio,
            random_state=config["SPLIT_SEED"],
            shuffle=True
        )

    if log_file: log_message(f"Train patients: {len(train_patient_ids)}, Validation patients: {len(val_patient_ids)}",
                             log_file)

    # Build slice references (both MSP and non-MSP) for training and validation sets
    train_refs = []
    val_refs = []

    def _create_refs_for_patients(patient_id_list, group_dict, description):
        slice_references = []
        if log_file: log_message(f"Building slice references for {description}...", log_file)
        for patient_id in tqdm(patient_id_list, desc=f"Processing {description} patients"):
            if patient_id not in group_dict: continue
            for pair_info in group_dict[patient_id]:
                # Cases WITH label files are positive MSP cases
                if pair_info.get("label") is not None and pair_info.get("has_msp", False):
                    # Positive case: Extract the MSP slice with full annotations
                    msp_idx = pair_info.get("msp_slice_idx", -1)
                    if msp_idx >= 0:
                        slice_references.append({
                            "image_path": pair_info["image"],
                            "label_path": pair_info["label"],
                            "slice_idx": msp_idx,
                            "case_id": pair_info["id"],
                            "patient_id": patient_id,
                            "is_msp": True
                        })
                else:
                    # Cases WITHOUT label files or without MSP are negative cases
                    # Sample multiple slices from these volumes
                    num_non_msp_slices = config.get("NUM_NON_MSP_SLICES_PER_CASE", 3)

                    try:
                        import nibabel as nib
                        img_nii = nib.load(str(pair_info["image"]))
                        vol_shape = img_nii.shape
                        num_slices = vol_shape[config.get("SAGITTAL_AXIS", 0)]

                        # Sample from middle 60% of volume to avoid edge slices
                        start_idx = int(num_slices * 0.2)
                        end_idx = int(num_slices * 0.8)
                        step = max(1, (end_idx - start_idx) // num_non_msp_slices)

                        for slice_idx in range(start_idx, end_idx, step)[:num_non_msp_slices]:
                            slice_references.append({
                                "image_path": pair_info["image"],
                                "label_path": None,  # No label for negative cases
                                "slice_idx": slice_idx,
                                "case_id": pair_info["id"],
                                "patient_id": patient_id,
                                "is_msp": False
                            })
                    except Exception as e:
                        if log_file:
                            log_message(f"Error sampling non-MSP slices for {pair_info['id']}: {e}", log_file)
                        continue

        return slice_references

    train_refs = _create_refs_for_patients(train_patient_ids, patient_groups, "training")
    val_refs = _create_refs_for_patients(val_patient_ids, patient_groups, "validation")

    # Count MSP vs non-MSP
    train_msp_count = sum(1 for ref in train_refs if ref.get("is_msp", False))
    val_msp_count = sum(1 for ref in val_refs if ref.get("is_msp", False))

    if log_file:
        log_message(f"Training slices: {len(train_refs)} total ({train_msp_count} MSP, {len(train_refs)-train_msp_count} non-MSP)", log_file)
        log_message(f"Validation slices: {len(val_refs)} total ({val_msp_count} MSP, {len(val_refs)-val_msp_count} non-MSP)", log_file)

    return train_refs, val_refs, patient_groups


def load_model_with_correct_architecture(model_path, config, device, log_file=None):
    """Unified model loading function that ensures architecture consistency based on 'model_type' in the checkpoint."""
    if not Path(model_path).exists():
        error_msg = f"Model checkpoint file not found: {model_path}"
        if log_file: log_message(f"ERROR: {error_msg}", log_file)
        raise FileNotFoundError(error_msg)

    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    model_type_from_checkpoint = checkpoint.get('model_type', 'UNetHeatmap')

    # Add support for UNetWithDualHeads
    if model_type_from_checkpoint in ['UNetWithCls', 'UNetWithCls_Stage1']:
        model_instance = UNetWithCls(
            n_channels=config["IN_CHANNELS"],
            n_classes=len(config["STRUCTURE_LABELS"]),
            bilinear_unet=config.get("BILINEAR_UNET", True)
        ).to(device)
        if log_file: log_message(f"Loading {model_type_from_checkpoint} model structure from {model_path}", log_file)

    elif model_type_from_checkpoint in ['UNetWithDualHeads', 'UNetWithDualHeads_Stage1']:
        # Add UNetWithDualHeads support
        model_instance = UNetWithDualHeads(
            in_channels=config["IN_CHANNELS"],
            feat_channels=64  # Use default value, should be consistent with training
        ).to(device)
        if log_file: log_message(f"Loading {model_type_from_checkpoint} model structure from {model_path}", log_file)

    elif model_type_from_checkpoint == 'UNetHeatmap':
        model_instance = UNetHeatmap(
            n_channels=config["IN_CHANNELS"],
            n_classes=len(config["STRUCTURE_LABELS"]),
            bilinear=config.get("BILINEAR_UNET", True)
        ).to(device)
        if log_file: log_message(f"Loading UNetHeatmap model structure from {model_path}", log_file)

    else:
        error_msg = f"Unknown model_type '{model_type_from_checkpoint}' in checkpoint {model_path}."
        if log_file: log_message(f"ERROR: {error_msg}", log_file)
        raise ValueError(error_msg)

    # Load the state dictionary
    try:
        model_instance.load_state_dict(checkpoint['model_state_dict'])
        if log_file: log_message(
            f"Successfully loaded model weights into {model_type_from_checkpoint} from {model_path}", log_file)
    except Exception as e:
        error_msg = f"Error loading model_state_dict from {model_path} for {model_type_from_checkpoint}: {e}"
        if log_file: log_message(f"ERROR: {error_msg}", log_file)
        raise RuntimeError(error_msg)

    return model_instance, model_type_from_checkpoint
