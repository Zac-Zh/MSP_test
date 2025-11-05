"""
Case-level evaluation functions for MSP detection.

Provides functions for testing models at both slice-level and case-level,
including volume-level MSP detection with coverage constraints.
"""

import numpy as np
import torch
import cv2
import pickle
import datetime
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import (
    roc_auc_score, accuracy_score, confusion_matrix, f1_score
)

from data.preprocessing import (
    preprocess_and_cache, extract_slice, normalize_slice,
    generate_brain_mask_from_image
)
from inference.tta import apply_tta_horizontal_flip
from inference.detection import detect_msp_case_level_with_coverage
from train.helpers import load_model_with_correct_architecture
from train.meta_classifier import extract_heatmap_features
from utils.gating import four_structure_and_gate_check
from utils.logging_utils import log_message
from visualization.evaluation_plots import save_final_roc_pr


def test_fold_model(heatmap_model_path, meta_clf_path, test_slice_refs, config, device,
                    test_patient_id_or_fold_id: str, fold_num_for_output: int, paths, log_file):
    """
    Tests a trained model on a fold or patient's dataset.

    Core functionality:
    1. Load heatmap model and meta-classifier
    2. Use calibrated slice classification threshold (Sens>=0.70 constraint)
    3. Process each test slice with TTA (Test-Time Augmentation)
    4. Extract features and apply meta-classifier for secondary prediction
    5. Save ROC curve data per fold for subsequent analysis
    6. Calculate and return detailed performance metrics including AUC, accuracy,
       sensitivity, and specificity

    Key improvement: Forces use of meta_data.get('slice_threshold_calibrated', ...)
    to ensure the threshold calibrated during training (for high sensitivity) is
    strictly enforced during inference.

    Args:
        heatmap_model_path: Path to trained heatmap model checkpoint
        meta_clf_path: Path to meta-classifier pickle file
        test_slice_refs: List of slice reference dicts for testing
        config: Configuration dictionary
        device: PyTorch device
        test_patient_id_or_fold_id: Patient ID or fold ID string
        fold_num_for_output: Fold number for output naming
        paths: Dictionary with output paths
        log_file: Log file path

    Returns:
        dict: Performance metrics including AUC, accuracy, sensitivity, specificity
    """
    if log_file:
        log_message(
            f"Testing model for identifier: {test_patient_id_or_fold_id} (output fold_num: {fold_num_for_output})...",
            log_file)

    # 1. Load main model and determine its architecture type
    full_model, model_type = load_model_with_correct_architecture(heatmap_model_path, config, device, log_file)
    full_model.eval()

    # Determine which part of the model to use for feature extraction
    if model_type in ['UNetWithCls', 'UNetWithCls_Stage1', 'UNetWithDualHeads', 'UNetWithDualHeads_Stage1']:
        heatmap_net_for_features = full_model.unet
    else:  # UNetHeatmap
        heatmap_net_for_features = full_model

    # 2. Load meta-classifier
    if not Path(meta_clf_path).exists():
        if log_file:
            log_message(
                f"ERROR: Meta-classifier not found at {meta_clf_path} for {test_patient_id_or_fold_id}. Returning empty metrics.",
                log_file)
        return {"patient_id": test_patient_id_or_fold_id, "fold": fold_num_for_output, "auc": 0.0,
                "error": "Meta-classifier missing"}

    with open(meta_clf_path, 'rb') as f:
        meta_data = pickle.load(f)

    # Unified slice binary classification threshold: prioritize calibrated threshold from training
    slice_decision_threshold = float(
        meta_data.get('slice_threshold_calibrated', config.get('SLICE_CLS_THRESHOLD', 0.5))
    )
    if log_file:
        src = "meta" if 'slice_threshold_calibrated' in meta_data else "config"
        log_message(f"  Using slice_decision_threshold={slice_decision_threshold:.3f} (source={src})", log_file)

    meta_lr_clf = meta_data['classifier']
    stored_dim = meta_data.get('feature_dim')  # Still load stored dimension for runtime checking

    # 3. Process test slices (evaluation set no longer filtered by gating)
    slice_true_labels_is_msp = []
    slice_pred_probs_meta_clf = []
    slice_pred_probs_direct_cls = []
    slice_gate_flags = []  # Record whether each slice passes gating (only for final binary classification)

    total_processed = 0
    passed_gate = 0
    H_model, W_model = config["IMAGE_SIZE"]
    expected_heatmap_channels = len(config["STRUCTURE_LABELS"])
    thresholds = config.get("FEATURE_THRESHOLDS", [0.1, 0.3, 0.5, 0.7])

    for ref in tqdm(test_slice_refs, desc=f"Testing slices for {test_patient_id_or_fold_id}"):
        try:
            img_vol, _ = preprocess_and_cache(str(ref["image_path"]), config["CACHE_DIR"], config, log_file)
            if img_vol is None:
                continue

            img_slice = extract_slice(img_vol, ref["slice_idx"], config["SAGITTAL_AXIS"])
            if img_slice is None:
                continue

            img_slice_norm = normalize_slice(img_slice, config)
            brain_mask_orig = generate_brain_mask_from_image(img_slice_norm, config) if config[
                "GENERATE_BRAIN_MASK_FROM_IMAGE"] else np.ones_like(img_slice_norm)

            img_resized = cv2.resize(img_slice_norm, (W_model, H_model), interpolation=cv2.INTER_LINEAR)
            brain_mask_resized = cv2.resize(brain_mask_orig, (W_model, H_model), interpolation=cv2.INTER_NEAREST)
            img_tensor = torch.from_numpy(img_resized).unsqueeze(0).unsqueeze(0).to(device)

            total_processed += 1

            with torch.no_grad():
                # Use TTA during testing for consistency
                if model_type in ['UNetWithCls', 'UNetWithCls_Stage1']:
                    current_heatmap_logits_tensor, direct_cls_logit = apply_tta_horizontal_flip(img_tensor, full_model)
                    direct_cls_prob_val = torch.sigmoid(direct_cls_logit).cpu().item()
                elif model_type in ['UNetWithDualHeads', 'UNetWithDualHeads_Stage1']:
                    model_outputs = apply_tta_horizontal_flip(img_tensor, full_model)
                    current_heatmap_logits_tensor, direct_cls_logit, _ = model_outputs
                    direct_cls_prob_val = torch.sigmoid(direct_cls_logit).cpu().item()
                else:  # UNetHeatmap
                    current_heatmap_logits_tensor = apply_tta_horizontal_flip(img_tensor, full_model)
                    direct_cls_prob_val = None

            heatmap_logits_np = current_heatmap_logits_tensor.cpu().numpy()[0]

            # Apply structure gating logic (no longer filters samples; only records gate_pass for final binary classification)
            gate_pass = True
            if config.get("ENABLE_STRUCTURE_GATE", True):
                pred_heatmaps_probs = 1.0 / (1.0 + np.exp(-heatmap_logits_np))
                gate_pass = four_structure_and_gate_check(pred_heatmaps_probs, config)
                if gate_pass:
                    passed_gate += 1
            else:
                # If gating not enabled, also consider as passed
                gate_pass = True
                passed_gate += 1

            # Extract features from TTA-augmented heatmaps
            features_for_meta = extract_heatmap_features(
                heatmap_logits_np,  # Pass logits, function will apply sigmoid internally
                brain_mask_resized, thresholds,
                expected_channels=expected_heatmap_channels, config=config
            )
            feat_row = features_for_meta.reshape(1, -1).astype(np.float32)

            # Robust runtime check: if dimension mismatch, use fallback probability
            if stored_dim is not None and feat_row.shape[1] != int(stored_dim):
                fallback_prob = 0.5 if (direct_cls_prob_val is None) else float(np.clip(direct_cls_prob_val, 0.0, 1.0))
                if log_file:
                    log_message(
                        f"  [WARN] Feature dim mismatch (got {feat_row.shape[1]}, expect {stored_dim}); "
                        f"use fallback prob={fallback_prob:.3f} for slice {ref.get('case_id', '?')}_{ref.get('slice_idx', '?')}",
                        log_file)
                meta_clf_prob = fallback_prob
            else:
                meta_clf_prob = float(meta_lr_clf.predict_proba(feat_row)[0, 1])

            slice_true_labels_is_msp.append(1 if ref.get("is_msp", False) else 0)

            # Evaluation score: continuous probability (after one sigmoid + TTA aggregation), no thresholding/gating
            slice_pred_probs_meta_clf.append(float(np.clip(meta_clf_prob, 0.0, 1.0)))
            direct_prob = 0.5 if (direct_cls_prob_val is None) else float(np.clip(direct_cls_prob_val, 0.0, 1.0))
            slice_pred_probs_direct_cls.append(direct_prob)

            # Gate flag only for final binary classification
            slice_gate_flags.append(bool(gate_pass))

        except Exception as e_slice_test:
            if log_file:
                log_message(
                    f"  Error testing slice {ref.get('case_id', 'unknown')}_{ref.get('slice_idx', 'unk')} for {test_patient_id_or_fold_id}: {e_slice_test}",
                    log_file)
            continue

    # Save ROC data to pickle file
    if slice_true_labels_is_msp and slice_pred_probs_meta_clf:
        roc_data = {
            'y_true': slice_true_labels_is_msp,
            'y_pred_proba': slice_pred_probs_meta_clf,
            'gate_flags': slice_gate_flags,
            'fold': test_patient_id_or_fold_id,
            'fold_num': fold_num_for_output,
            'timestamp': datetime.datetime.now().isoformat(),
            'model_type': model_type
        }

        roc_data_filename = f"roc_data_{test_patient_id_or_fold_id}.pkl"
        roc_data_path = paths["checkpoint_dir"] / roc_data_filename

        try:
            with open(roc_data_path, 'wb') as f:
                pickle.dump(roc_data, f)
            if log_file:
                log_message(f"  ✅ ROC data saved: {roc_data_path} ({len(slice_true_labels_is_msp)} samples)", log_file)
        except Exception as e_roc:
            if log_file:
                log_message(f"  ⚠️ Failed to save ROC data: {e_roc}", log_file)
    else:
        if log_file:
            log_message(f"  ⚠️ No ROC data to save for {test_patient_id_or_fold_id} (empty predictions)", log_file)

    # 4. Calculate and return metrics
    if not slice_true_labels_is_msp:
        if log_file:
            log_message(f"  No slices processed for {test_patient_id_or_fold_id}. Returning zero metrics.", log_file)
        return {"patient_id": test_patient_id_or_fold_id, "fold": fold_num_for_output, "auc": 0.0, "n_positive": 0,
                "n_negative": 0, "error": "No slices processed"}

    y_true_np = np.array(slice_true_labels_is_msp)
    y_pred_probs_meta_clf_np = np.array(slice_pred_probs_meta_clf)
    gate_flags_np = np.array(slice_gate_flags, dtype=bool)

    assert len(y_true_np) == len(y_pred_probs_meta_clf_np) == len(gate_flags_np), \
        "Eval lists length mismatch; do NOT filter slices before ROC/PR."

    if log_file:
        log_message(f"  ROC/PR evaluated on ALL slices: N={len(y_true_np)} (no filtering).", log_file)

    metrics_results = {"patient_id": test_patient_id_or_fold_id, "fold": fold_num_for_output, "model_type": model_type}
    metrics_results["n_positive"] = int(y_true_np.sum())
    metrics_results["n_negative"] = int((y_true_np == 0).sum())
    metrics_results["gate_stats"] = {
        "total_processed": total_processed, "passed_gate": passed_gate,
        "gate_pass_rate": passed_gate / total_processed if total_processed > 0 else 0.0
    }

    if len(np.unique(y_true_np)) > 1:
        metrics_results["auc"] = roc_auc_score(y_true_np, y_pred_probs_meta_clf_np)

        if bool(config.get("SLICE_EVAL_IGNORE_GATE", True)):
            y_pred_binary = (y_pred_probs_meta_clf_np >= slice_decision_threshold).astype(int)
            if log_file:
                log_message(f"  Binary classification: threshold only (gate ignored for eval)", log_file)
        else:
            y_pred_binary = ((y_pred_probs_meta_clf_np >= slice_decision_threshold) & gate_flags_np).astype(int)
            if log_file:
                log_message(f"  Binary classification: threshold AND gate", log_file)

        metrics_results["accuracy"] = accuracy_score(y_true_np, y_pred_binary)
        cm = confusion_matrix(y_true_np, y_pred_binary, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        metrics_results["sensitivity"] = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        metrics_results["specificity"] = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        metrics_results["f1_score"] = f1_score(y_true_np, y_pred_binary, zero_division=0)
    else:
        metrics_results["auc"] = 0.0

        if bool(config.get("SLICE_EVAL_IGNORE_GATE", True)):
            y_pred_binary = (y_pred_probs_meta_clf_np >= slice_decision_threshold).astype(int)
        else:
            y_pred_binary = ((y_pred_probs_meta_clf_np >= slice_decision_threshold) & gate_flags_np).astype(int)

        metrics_results["accuracy"] = accuracy_score(y_true_np, y_pred_binary)
        cm = confusion_matrix(y_true_np, y_pred_binary, labels=[0, 1])
        if cm.size == 4:
            tn, fp, fn, tp = cm.ravel()
        else:
            tn, fp, fn, tp = 0, 0, 0, int(y_true_np.sum()) if y_true_np.size else 0
        metrics_results["sensitivity"] = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        metrics_results["specificity"] = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        metrics_results["f1_score"] = f1_score(y_true_np, y_pred_binary, zero_division=0)

    if log_file:
        log_message(
            f"  Metrics for {test_patient_id_or_fold_id}: AUC={metrics_results.get('auc', 0.0):.4f}, "
            f"Acc={metrics_results.get('accuracy', 0.0):.4f}",
            log_file)
        log_message(
            f"  Gate pass rate: {metrics_results['gate_stats']['gate_pass_rate']:.1%} ({passed_gate}/{total_processed})",
            log_file)

    if bool(config.get("SAVE_FINAL_SLICE_ROC", True)):
        out_png = paths["viz_dir"] / f"FINAL_slice_ROC_PR_{test_patient_id_or_fold_id}.png"
        save_final_roc_pr(y_true_np, y_pred_probs_meta_clf_np, out_png, title_prefix="Final Slice")
        if log_file: log_message(f"  Final slice ROC/PR saved: {out_png}", log_file)

    return metrics_results


def test_case_level(heatmap_model_path, meta_clf_path, volume_info: dict, config, device,
                    test_patient_id_for_output: str, fold_num_for_output: int, log_file,
                    precomputed_detection_output: dict = None,
                    case_decision_threshold: float = None):
    """
    Test case-level MSP detection on a single volume.

    Args:
        heatmap_model_path: Path to trained heatmap model
        meta_clf_path: Path to meta-classifier
        volume_info: Dictionary with volume information (volume_path, true labels, etc.)
        config: Configuration dictionary
        device: PyTorch device
        test_patient_id_for_output: Patient ID for output
        fold_num_for_output: Fold number for output
        log_file: Log file path
        precomputed_detection_output: Optional precomputed detection results
        case_decision_threshold: Optional case-level decision threshold

    Returns:
        dict: Case-level results including accuracy, predicted MSP slice, probabilities
    """
    # Add field checks and default values
    if volume_info is None:
        return {"patient_id": test_patient_id_for_output, "error": "volume_info is None"}

    # Ensure required fields exist
    required_fields = {
        'volume_path': None,
        'true_case_has_msp': False,
        'true_msp_slice_idx': -1,
        'patient_id': test_patient_id_for_output
    }

    for key, default_val in required_fields.items():
        if key not in volume_info:
            volume_info[key] = default_val

    if 'volume_path' not in volume_info:
        return {
            "patient_id": test_patient_id_for_output,
            "fold": fold_num_for_output,
            "volume_path": "N/A",
            "error": "volume_info missing volume_path"
        }

    try:
        if precomputed_detection_output:
            detection_output_for_volume = precomputed_detection_output
            if log_file:
                log_message(f"  Using precomputed detection output for {volume_info['volume_path']}", log_file)
        else:
            detection_output_for_volume = detect_msp_case_level_with_coverage(
                heatmap_model_path, meta_clf_path, volume_info['volume_path'],
                config,
                case_decision_threshold=None,
                log_file=log_file
            )

        # Extract true labels for this case from volume_info
        true_case_label_has_msp = volume_info['true_case_has_msp']
        true_msp_slice_actual_idx = volume_info.get('true_msp_slice_idx', -1)

        # Extract predictions from detection_output
        pred_case_info = detection_output_for_volume['case_level_decision']
        thr_used = pred_case_info.get('threshold_used', None)
        case_prob = float(pred_case_info.get('case_prob', pred_case_info.get('case_prob_predicted', 0.0)))

        if thr_used is not None:
            pred_case_label_has_msp = (case_prob >= float(thr_used))
        else:
            pred_case_label_has_msp = bool(pred_case_info.get('has_msp', False))

        pred_msp_slice_idx = pred_case_info['predicted_msp_slice']  # -1 if not pred_case_label_has_msp
        pred_case_probability = pred_case_info['case_prob']

        # Calculate performance metrics for this single case
        # Case-level accuracy (correctly classified as having MSP or not)
        case_acc = 1.0 if (true_case_label_has_msp == pred_case_label_has_msp) else 0.0

        # Slice localization accuracy (only if both true and pred indicate MSP)
        slice_loc_acc = 0.0
        if true_case_label_has_msp and pred_case_label_has_msp and true_msp_slice_actual_idx != -1:
            slice_idx_tolerance = config.get("MSP_SLICE_TOLERANCE", 2)
            slice_loc_acc = 1.0 if abs(true_msp_slice_actual_idx - pred_msp_slice_idx) <= slice_idx_tolerance else 0.0

        # Simplified "AUC" for a single case: probability if positive, 1-probability if negative
        # This is a proxy, true AUC needs multiple cases.
        single_case_auc_proxy = pred_case_probability if true_case_label_has_msp else (1.0 - pred_case_probability)

        # Sensitivity/Specificity components for this single case
        # These are binary (1 or 0) for a single case, will be averaged later.
        case_sens_component = 0.0
        case_spec_component = 0.0
        if true_case_label_has_msp:  # This is a "positive" case
            if pred_case_label_has_msp:
                case_sens_component = 1.0  # TP
            # else: FN (sens is 0)
        else:  # This is a "negative" case
            if not pred_case_label_has_msp:
                case_spec_component = 1.0  # TN
            # else: FP (spec is 0)

        result_dict = {
            "patient_id": test_patient_id_for_output,  # The patient this volume belongs to
            "fold": fold_num_for_output,  # Fold number if part of CV
            "volume_path": str(volume_info['volume_path']),
            "true_case_has_msp": true_case_label_has_msp,
            "pred_case_has_msp": pred_case_label_has_msp,
            "true_msp_slice_idx": true_msp_slice_actual_idx,
            "pred_msp_slice_idx": pred_msp_slice_idx,
            "case_prob_predicted": pred_case_probability,
            "case_prob": pred_case_probability,
            "case_accuracy": case_acc,  # 0 or 1 for this case
            "slice_accuracy": slice_loc_acc,  # 0 or 1 for this case if applicable
            "case_auc": single_case_auc_proxy,  # Proxy for AUC contribution
            "case_sensitivity": case_sens_component,  # 0 or 1 for this case
            "case_specificity": case_spec_component,  # 0 or 1 for this case
            "model_type_used": detection_output_for_volume.get('primary_model_type', 'Unknown'),
            "all_slice_probs": detection_output_for_volume.get('slice_probs_from_meta_classifier', [])  # Store for viz
        }

        # Log message depends on the context of how test_case_level is called.
        if log_file and not precomputed_detection_output:
            log_message(
                f"  Case Test (detection run inside): Vol='{Path(volume_info['volume_path']).name}', TrueHasMSP={true_case_label_has_msp}, "
                f"PredHasMSP={pred_case_label_has_msp} (Prob={pred_case_probability:.3f}), CaseAcc={case_acc:.1f}",
                log_file)
        elif log_file and precomputed_detection_output:
            log_message(
                f"  Case Test (used precomputed detection): Vol='{Path(volume_info['volume_path']).name}', TrueHasMSP={true_case_label_has_msp}, "
                f"PredHasMSP={pred_case_label_has_msp} (Prob={pred_case_probability:.3f}), CaseAcc={case_acc:.1f}",
                log_file)
        return result_dict

    except Exception as e_test_case:
        import traceback
        if log_file:
            log_message(
                f"ERROR in test_case_level for patient {test_patient_id_for_output}, vol {Path(volume_info.get('volume_path', 'N/A')).name}: {e_test_case}\n{traceback.format_exc()}",
                log_file)
        return {
            "patient_id": test_patient_id_for_output, "fold": fold_num_for_output,
            "volume_path": str(volume_info.get('volume_path', 'N/A')), "error": str(e_test_case)
        }
