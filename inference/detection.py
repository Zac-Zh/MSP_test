"""
Case-level MSP detection functions.

Provides volume-level MSP detection with coverage awareness, slice processing,
and decision-making logic.
"""

import numpy as np
import torch
import cv2
import pickle
from pathlib import Path
from tqdm import tqdm
from typing import List

from data.preprocessing import (
    preprocess_and_cache, extract_slice, normalize_slice,
    generate_brain_mask_from_image
)
from train.helpers import load_model_with_correct_architecture
from train.meta_classifier import extract_heatmap_features
from inference.tta import apply_tta_horizontal_flip
from utils.gating import four_structure_and_gate_check
from utils.logging_utils import log_message
from utils.msp_utils import get_msp_index
from data import load_nifti_data_cached


def combine_slice_probability(cov_logits: torch.Tensor, alpha: float = 0.5) -> torch.Tensor:
    """
    Combines coverage classification logits into a single slice probability.

    Formula: slice_prob = P(full) + alpha * P(partial)

    Args:
        cov_logits: Coverage logits tensor (batch, 3) for [no_structure, partial, full]
        alpha: Weight for partial coverage (default 0.5)

    Returns:
        torch.Tensor: Combined slice probability
    """
    probs = torch.softmax(cov_logits, dim=1)
    return probs[:, 2] + alpha * probs[:, 1]


def process_slice_with_coverage_constraints(volume_data, slice_idx, heatmap_net, config, device, log_file=None):
    """
    Processes a single slice for coverage-aware models with gating constraints.

    This function:
    1. Extracts and normalizes the slice
    2. Filters empty/low-quality slices
    3. Runs model inference with TTA
    4. Applies four-structure AND gate check
    5. Extracts heatmap features for meta-classifier

    Args:
        volume_data: Full volume numpy array
        slice_idx: Index of the slice to process
        heatmap_net: Trained heatmap model
        config: Configuration dictionary
        device: PyTorch device
        log_file: Optional log file path

    Returns:
        dict: Dictionary with msp_head_prob, coverage_prob, heatmap_features, pred_heatmaps_probs
              Returns dict with zeros if slice fails quality/gating checks
    """
    sagittal_axis = config["SAGITTAL_AXIS"]
    H_model, W_model = config["IMAGE_SIZE"]

    # Extract slice
    img_slice = extract_slice(volume_data, slice_idx, sagittal_axis)
    if img_slice is None:
        return {
            'msp_head_prob': 0.0,
            'coverage_prob': 0.0,
            'heatmap_features': None,
            'pred_heatmaps_probs': None
        }

    # Normalize
    img_slice_norm = normalize_slice(img_slice, config)

    # Filter empty skulls
    foreground_ratio = (img_slice_norm > 0.01).mean()
    if foreground_ratio < 0.05:
        return {
            'msp_head_prob': 0.0,
            'coverage_prob': 0.0,
            'heatmap_features': None,
            'pred_heatmaps_probs': None
        }

    # Generate brain mask
    brain_mask = generate_brain_mask_from_image(img_slice_norm, config) if config[
        "GENERATE_BRAIN_MASK_FROM_IMAGE"] else np.ones_like(img_slice_norm)

    # Resize and convert to tensor
    img_resized = cv2.resize(img_slice_norm, (W_model, H_model), interpolation=cv2.INTER_LINEAR)
    brain_mask_resized = cv2.resize(brain_mask, (W_model, H_model), interpolation=cv2.INTER_NEAREST)
    img_tensor = torch.from_numpy(img_resized).unsqueeze(0).unsqueeze(0).to(device)

    # Model inference
    with torch.no_grad():
        # Use the same TTA path as validation/ROC collection
        model_outputs = apply_tta_horizontal_flip(img_tensor, heatmap_net)
        # Support UNetWithDualHeads's three-output format
        if isinstance(model_outputs, (tuple, list)) and len(model_outputs) == 3:  # UNetWithDualHeads
            pred_heatmaps_logits, cls_logit, cov_logits = model_outputs
            msp_head_out = torch.sigmoid(cls_logit).squeeze().cpu().item()
            # Use coverage information to calculate slice probability
            coverage_slice_prob = combine_slice_probability(cov_logits, alpha=0.5)
            coverage_slice_prob = coverage_slice_prob.squeeze().cpu().item()

        elif isinstance(model_outputs, (tuple, list)) and len(model_outputs) == 2:  # UNetWithCls
            pred_heatmaps_logits, cls_logit = model_outputs
            msp_head_out = torch.sigmoid(cls_logit).squeeze().cpu().item()
            coverage_slice_prob = msp_head_out  # fallback

        else:  # UNetHeatmap only
            pred_heatmaps_logits = model_outputs
            msp_head_out = 0.5
            coverage_slice_prob = 0.5  # fallback

    # Get probability maps
    pred_heatmaps_probs = torch.sigmoid(pred_heatmaps_logits).cpu().numpy()[0]  # (C,H,W)

    # Use configurable four-structure AND gate check
    if config.get("ENABLE_STRUCTURE_GATE", True):
        if not four_structure_and_gate_check(pred_heatmaps_probs, config):
            if config.get("GATE_DEBUG", False):
                print(f"  Slice {slice_idx}: Did not pass four-structure AND gate check")
            return {
                'msp_head_prob': 0.0,
                'coverage_prob': 0.0,
                'heatmap_features': None,
                'pred_heatmaps_probs': None
            }
        else:
            if config.get("GATE_DEBUG", False):
                print(f"  Slice {slice_idx}: Passed four-structure AND gate check")

    # Extract features for the meta-classifier
    heatmap_features = extract_heatmap_features(
        pred_heatmaps_logits.cpu().numpy()[0],
        brain_mask_resized,
        config["FEATURE_THRESHOLDS"],
        expected_channels=len(config["HEATMAP_LABEL_MAP"]),
        config=config
    )

    # Return multiple probabilities for decision-making
    return {
        'msp_head_prob': msp_head_out,
        'coverage_prob': coverage_slice_prob,
        'heatmap_features': heatmap_features,
        'pred_heatmaps_probs': pred_heatmaps_probs
    }


def evaluate_case_level(volume_slice_probs: List[float], case_threshold: float = 0.5):
    """
    Makes case-level MSP decision based on the highest slice probability strategy.

    Args:
        volume_slice_probs: List of per-slice MSP probabilities
        case_threshold: Threshold for case-level decision

    Returns:
        dict: Case-level decision with has_msp, case_prob, predicted_msp_slice, decision_details
    """
    if not volume_slice_probs or len(volume_slice_probs) == 0:
        return {
            'has_msp': False, 'case_prob': 0.0, 'predicted_msp_slice': -1,
            'max_slice_prob': 0.0, 'decision_details': {'reason': 'empty_slice_probabilities'}
        }

    probs_np = np.array(volume_slice_probs, dtype=np.float64)

    # Use the highest slice probability
    case_prob = float(np.max(probs_np))

    case_has_msp = case_prob >= case_threshold
    predicted_msp_slice_idx = int(np.argmax(probs_np)) if case_has_msp else -1

    decision_details = {
        'slice_count': len(probs_np),
        'mean_slice_prob': float(np.mean(probs_np)),
        'std_slice_prob': float(np.std(probs_np)),
        'max_slice_prob_in_volume': case_prob,
        'case_decision_threshold_used': case_threshold,
        'strategy': 'max_slice_probability'
    }

    return {
        'has_msp': case_has_msp,
        'case_prob': case_prob,
        'predicted_msp_slice': predicted_msp_slice_idx,
        'max_slice_prob_in_volume': case_prob,
        'decision_details': decision_details
    }


def detect_msp_case_level_with_coverage(heatmap_model_path, meta_clf_path, volume_image_path, config,
                                        case_decision_threshold=None, log_file=None, keypoints_output_dir=None):
    """
    Performs case-level MSP detection with coverage-aware models.

    This is the main volume-level detection function that:
    1. Loads models (heatmap model + meta-classifier)
    2. Processes all slices in the volume with coverage constraints
    3. Applies meta-classifier to get per-slice probabilities
    4. Makes case-level decision using max slice probability
    5. Optionally detects keypoints on high-confidence slices

    Args:
        heatmap_model_path: Path to trained heatmap model
        meta_clf_path: Path to meta-classifier pickle file
        volume_image_path: Path to input NIfTI volume
        config: Configuration dictionary
        case_decision_threshold: Optional case-level threshold (auto-determined if None)
        log_file: Optional log file path
        keypoints_output_dir: Optional directory for keypoint visualization

    Returns:
        dict: Comprehensive detection results including:
            - volume_path: Input volume path
            - slice_probs_*: Per-slice probabilities from different sources
            - case_level_decision: Final case-level decision dict
            - num_slices_processed, constraint_pass_rate: Processing statistics
            - slice_channel_max_probs: Channel-wise max probabilities
            - true_msp_slice, true_case_has_msp: Ground truth if available
            - primary_model_type, constraints_applied: Metadata
    """
    if log_file:
        log_message(f"Starting coverage-aware case-level MSP detection for: {volume_image_path}", log_file)
    device = torch.device(config["DEVICE"])

    # Load models
    full_model_instance, model_type_loaded = load_model_with_correct_architecture(
        heatmap_model_path, config, device, log_file
    )
    full_model_instance.eval()

    # Determine the feature extractor
    if model_type_loaded == 'UNetWithDualHeads':
        heatmap_producer_net = full_model_instance
        if log_file:
            log_message(f"  Using UNetWithDualHeads model for coverage-aware detection", log_file)
    elif model_type_loaded in ['UNetWithCls', 'UNetWithCls_Stage1']:
        heatmap_producer_net = full_model_instance
        if log_file:
            log_message(f"  Using {model_type_loaded} model", log_file)
    else:
        heatmap_producer_net = full_model_instance
        if log_file:
            log_message(f"  Using {model_type_loaded} model", log_file)

    # Load meta-classifier
    if not Path(meta_clf_path).exists():
        error_msg = f"Meta-classifier file not found: {meta_clf_path}"
        if log_file:
            log_message(f"ERROR: {error_msg}", log_file)
        raise FileNotFoundError(error_msg)

    with open(meta_clf_path, 'rb') as f:
        meta_clf_data = pickle.load(f)
    meta_logistic_regressor = meta_clf_data['classifier']

    if case_decision_threshold is None:
        case_decision_threshold = meta_clf_data.get('calibrated_case_threshold')
        if case_decision_threshold is None:
            case_decision_threshold = config.get("CASE_THRESHOLD_OPTIMAL")
        if case_decision_threshold is None:
            case_decision_threshold = config.get("CASE_THRESHOLD_DEFAULT", 0.5)

    # Load and preprocess image
    img_vol_data, nii_ref = preprocess_and_cache(str(volume_image_path), config["CACHE_DIR"], config, log_file)
    if img_vol_data is None:
        error_msg = f"Failed to load or preprocess volume: {volume_image_path}"
        if log_file:
            log_message(f"ERROR: {error_msg}", log_file)
        raise ValueError(error_msg)

    # Get spacing information
    try:
        real_zooms = nii_ref.header.get_zooms()
        sagittal_axis = config["SAGITTAL_AXIS"]

        if len(real_zooms) >= 3:
            if sagittal_axis == 0:
                spacing_x_effective, spacing_y_effective = real_zooms[1], real_zooms[2]
            elif sagittal_axis == 1:
                spacing_x_effective, spacing_y_effective = real_zooms[0], real_zooms[2]
            else:
                spacing_x_effective, spacing_y_effective = real_zooms[0], real_zooms[1]
        else:
            spacing_x_effective, spacing_y_effective = 1.0, 1.0

        if sagittal_axis == 0:
            H_orig, W_orig = img_vol_data.shape[1], img_vol_data.shape[2]
        elif sagittal_axis == 1:
            H_orig, W_orig = img_vol_data.shape[0], img_vol_data.shape[2]
        else:
            H_orig, W_orig = img_vol_data.shape[0], img_vol_data.shape[1]

        H_model, W_model = config["IMAGE_SIZE"]

    except Exception as e_spacing:
        if log_file:
            log_message(f"  Warning: Could not get real spacing: {e_spacing}, using defaults", log_file)
        spacing_x_effective, spacing_y_effective = 1.0, 1.0
        H_orig, W_orig = 512, 512
        H_model, W_model = config["IMAGE_SIZE"]

    # Find the corresponding label file for ground truth
    true_msp_slice_idx = -1
    true_case_has_msp = None
    volume_path_obj = Path(volume_image_path)

    # Try to find label file
    possible_label_paths = [
        volume_path_obj.parent / f"{volume_path_obj.stem}_label.nii.gz",
        volume_path_obj.parent / "labels" / f"{volume_path_obj.stem}.nii.gz",
        Path(str(volume_path_obj).replace("images", "labels"))
    ]

    for label_path in possible_label_paths:
        if label_path.exists():
            try:
                label_vol = load_nifti_data_cached(str(label_path), is_label=True)
                if label_vol is not None:
                    true_msp_slice_idx = get_msp_index(label_vol, config["SAGITTAL_AXIS"], config["STRUCTURE_LABELS"])
                    true_case_has_msp = (true_msp_slice_idx >= 0)
                    break
            except:
                pass

    num_slices_in_volume = img_vol_data.shape[config["SAGITTAL_AXIS"]]
    per_slice_msp_probs_meta_clf = []
    per_slice_direct_cls_probs = []
    per_slice_coverage_probs = []
    per_slice_channel_max_probs = []

    # Keypoint detection data collection
    keypoint_detections = []
    case_id = Path(volume_image_path).stem

    # Gating statistics
    slice_passed_constraints = 0
    total_slices_processed = 0

    if log_file and config.get("ENABLE_STRUCTURE_GATE", True):
        log_message(f"  Gating enabled, threshold: {config.get('AND_GATE_THRESHOLD', 0.25)}", log_file)
    elif log_file:
        log_message("  Gating disabled", log_file)

    # Process each slice with constraints
    for slice_idx in tqdm(range(num_slices_in_volume),
                          desc=f"Coverage-aware processing for {Path(volume_image_path).name}"):
        try:
            # Use the constrained slice processing function
            slice_result = process_slice_with_coverage_constraints(
                img_vol_data, slice_idx, heatmap_producer_net, config, device, log_file
            )

            if slice_result['heatmap_features'] is None:  # Slice was filtered
                per_slice_msp_probs_meta_clf.append(0.0)
                per_slice_direct_cls_probs.append(0.0)
                per_slice_coverage_probs.append(0.0)
                per_slice_channel_max_probs.append(np.zeros(len(config["HEATMAP_LABEL_MAP"]), dtype=np.float32))
                continue

            total_slices_processed += 1
            slice_passed_constraints += 1

            # Extract results
            msp_head_prob = slice_result['msp_head_prob']
            coverage_prob = slice_result['coverage_prob']
            heatmap_features = slice_result['heatmap_features']
            pred_heatmaps_probs = slice_result['pred_heatmaps_probs']

            # Extract channel max probabilities
            C = pred_heatmaps_probs.shape[0]
            channel_max_probs = pred_heatmaps_probs.reshape(C, -1).max(axis=1)
            per_slice_channel_max_probs.append(channel_max_probs.astype(np.float32))

            # Meta-classifier prediction
            meta_clf_slice_prob = meta_logistic_regressor.predict_proba(heatmap_features.reshape(1, -1))[0, 1]

            per_slice_msp_probs_meta_clf.append(meta_clf_slice_prob)
            per_slice_direct_cls_probs.append(msp_head_prob)
            per_slice_coverage_probs.append(coverage_prob)

        except Exception as e_slice:
            if log_file:
                log_message(f"Error processing slice {slice_idx}: {e_slice}", log_file)
            per_slice_msp_probs_meta_clf.append(0.0)
            per_slice_direct_cls_probs.append(0.0)
            per_slice_coverage_probs.append(0.0)
            per_slice_channel_max_probs.append(np.zeros(len(config["HEATMAP_LABEL_MAP"]), dtype=np.float32))

    # Case-level Decision
    # Combine multiple probabilities for decision-making
    combined_slice_probs = []
    for meta_prob, coverage_prob in zip(per_slice_msp_probs_meta_clf, per_slice_coverage_probs):
        # Simple weighted combination
        combined_prob = 0.7 * meta_prob + 0.3 * coverage_prob
        combined_slice_probs.append(combined_prob)

    # Aggregate for case-level decision
    case_level_decision_output = evaluate_case_level(combined_slice_probs, case_decision_threshold)

    # Compile results with coverage information
    final_results = {
        'volume_path': str(volume_image_path),
        'slice_probs_from_meta_classifier': per_slice_msp_probs_meta_clf,
        'slice_probs_from_coverage': per_slice_coverage_probs,
        'slice_probs_combined': combined_slice_probs,
        'slice_probs_from_direct_classifier': per_slice_direct_cls_probs if model_type_loaded in ['UNetWithCls',
                                                                                                  'UNetWithCls_Stage1',
                                                                                                  'UNetWithDualHeads'] else None,
        'case_level_decision': case_level_decision_output,
        'num_slices_processed': num_slices_in_volume,
        'num_slices_passed_constraints': slice_passed_constraints,
        'constraint_pass_rate': slice_passed_constraints / total_slices_processed if total_slices_processed > 0 else 0.0,
        'slice_channel_max_probs': np.vstack(per_slice_channel_max_probs) if per_slice_channel_max_probs else None,
        'true_msp_slice': true_msp_slice_idx,
        'true_case_has_msp': true_case_has_msp,
        'case_decision_threshold_used': case_decision_threshold,
        'primary_model_type': model_type_loaded,
        'heatmap_model_path': str(heatmap_model_path),
        'meta_classifier_path': str(meta_clf_path),
        'tta_enabled': True,
        'coverage_aware': True,
        'constraints_applied': {
            'four_structure_and_gate': config.get("ENABLE_STRUCTURE_GATE", True),
            'gate_threshold': config.get("AND_GATE_THRESHOLD", 0.25),
            'empty_skull_filter': True,
            'keypoint_mask_snap': True,
            'coverage_classification': True,
            'explicit_label_mapping': True
        },
        'keypoints_detected': len(keypoint_detections),
        'effective_spacing_mm': (spacing_x_effective, spacing_y_effective),
        'label_mapping_used': {
            'heatmap_label_map': config["HEATMAP_LABEL_MAP"],
            'msp_required_labels': config["MSP_REQUIRED_LABELS"],
            'kp_required_labels': config["KP_REQUIRED_LABELS"]
        }
    }

    if log_file:
        log_message(f"Coverage-aware case-level detection for {Path(volume_image_path).name}:", log_file)
        log_message(f"  Decision: {'HAS MSP' if case_level_decision_output['has_msp'] else 'NO MSP'}", log_file)
        log_message(f"  Combined Case Probability: {case_level_decision_output['case_prob']:.4f}", log_file)
        log_message(
            f"  GT: {'HAS MSP' if true_case_has_msp else 'NO MSP' if true_case_has_msp is not None else 'Unknown'}",
            log_file)
        log_message(
            f"  Constraint pass rate: {final_results['constraint_pass_rate']:.3f} ({slice_passed_constraints}/{total_slices_processed})",
            log_file)
        log_message(f"  Coverage-aware features: enabled", log_file)

    return final_results
