"""
Validation pipelines for MSP detection.

Provides complete 5-fold cross-validation and baseline validation pipelines
exactly as implemented in the original research code.
"""

import numpy as np
import pandas as pd
import torch
import json
from pathlib import Path
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score, confusion_matrix

from train import (
    prepare_patient_grouped_datasets,
    generate_negative_samples,
    train_heatmap_model_with_coverage_aware_training,
    train_meta_classifier_full,
)
from eval import test_fold_model, test_case_level, find_optimal_case_threshold
from inference import detect_msp_case_level_with_coverage
from data import load_nifti_data_cached
from utils import (
    log_message,
    get_msp_index,
    build_lopo_results_df,
    tune_and_gate_threshold_min_on_val,
)
from visualization import (
    create_5fold_visualizations,
    create_case_level_visualizations,
)




def run_5fold_validation_with_case_level(config, paths):
    """
    [FIXED v6.2] 5-fold with PER-FOLD threshold optimization.
    Includes robustness patches for paths, variable scopes, and prevention of
    within-fold validation data leakage into meta-classifier training.
    This version incorporates patches to correctly separate threshold scanning
    from final evaluation, preventing data contamination and fixing logic errors.
    """
    log_file = paths["log_file"]
    device = torch.device(config["DEVICE"])
    avg_fold_threshold = float(config.get("CASE_THRESHOLD_DEFAULT", 0.5))
    std_fold_threshold = 0.0
    # Ensure output directory exists and unify JSON path exposure
    paths["run_dir"].mkdir(parents=True, exist_ok=True)
    paths["per_fold_thresholds_json"] = paths["run_dir"] / "per_fold_thresholds.json"

    if log_file:
        log_message("Starting 5-fold CV (GroupKFold) with PER-FOLD threshold optimization...", log_file)

    # 1) Prepare all patient data
    _, _, all_patient_groups = prepare_patient_grouped_datasets(config, log_file)
    all_patient_ids = list(all_patient_groups.keys())
    if not all_patient_ids:
        raise ValueError("No patients found to perform 5-Fold CV.")

    n_cv_splits = config.get("KFOLD_SPLITS", 5)
    if len(all_patient_ids) < n_cv_splits:
        n_cv_splits = len(all_patient_ids)
        if log_file: log_message(f"Warning: only {len(all_patient_ids)} patients -> use {n_cv_splits}-fold", log_file)
    if n_cv_splits < 2:
        raise ValueError(f"Not enough patients ({len(all_patient_ids)}) for CV.")

    gkf = GroupKFold(n_splits=n_cv_splits)
    patients_np = np.array(all_patient_ids)

    # Result containers
    per_fold_slice_metrics = []
    per_fold_case_results_list = []
    fold_models_paths_info = []
    per_fold_optimal_thresholds = []
    per_fold_thr_map = {}

    if log_file: log_message(f"Total unique patients for CV: {len(patients_np)} | using {n_cv_splits}-fold", log_file)

    for fold_idx, (tr_patient_indices, va_patient_indices) in enumerate(gkf.split(patients_np, groups=patients_np)):
        fold_num = fold_idx + 1
        fold_id = f"fold_{fold_num}"
        if log_file: log_message(f"\n--- Processing {fold_id}/{n_cv_splits} ---", log_file)

        tr_pids = patients_np[tr_patient_indices].tolist()
        va_pids = patients_np[va_patient_indices].tolist()
        
        # PATCH C: Assert no patient intersection (GroupKFold should guarantee this)
        assert set(tr_pids).isdisjoint(va_pids), "Leakage: train/val patients overlap within a fold."
        
        if log_file: log_message(f"  Train patients: {len(tr_pids)}, Val patients: {len(va_pids)}", log_file)

        # 3) Build training and validation slice lists
        tr_msp_refs = []
        for pid in tr_pids:
            for case_data in all_patient_groups.get(pid, []):
                if case_data.get("label"):
                    label_vol = load_nifti_data_cached(str(case_data["label"]), is_label=True)
                    if label_vol is None: continue
                    msp_idx = get_msp_index(label_vol, config["SAGITTAL_AXIS"], config["STRUCTURE_LABELS"])
                    if msp_idx >= 0:
                        tr_msp_refs.append({
                            "image_path": case_data["image"], "label_path": case_data["label"],
                            "slice_idx": msp_idx, "case_id": case_data["id"],
                            "patient_id": pid, "is_msp": True
                        })
        
        neg_tr_slices = generate_negative_samples(tr_msp_refs, all_patient_groups, config, target_patient_ids=tr_pids, log_file=log_file)
        tr_all_slices = tr_msp_refs + neg_tr_slices
        
        seed = int(config.get("SPLIT_SEED", 42))
        rng = np.random.RandomState(seed + fold_num)
        rng.shuffle(tr_all_slices)

        va_msp_refs = []
        for pid in va_pids:
            for case_data in all_patient_groups.get(pid, []):
                if case_data.get("label"):
                    label_vol = load_nifti_data_cached(str(case_data["label"]), is_label=True)
                    if label_vol is None: continue
                    msp_idx = get_msp_index(label_vol, config["SAGITTAL_AXIS"], config["STRUCTURE_LABELS"])
                    if msp_idx >= 0:
                        va_msp_refs.append({
                            "image_path": case_data["image"], "label_path": case_data["label"],
                            "slice_idx": msp_idx, "case_id": case_data["id"],
                            "patient_id": pid, "is_msp": True
                        })

        neg_va_slices = generate_negative_samples(va_msp_refs, all_patient_groups, config, target_patient_ids=va_pids, log_file=log_file)
        va_all_slices = va_msp_refs + neg_va_slices
        rng.shuffle(va_all_slices)

        if log_file:
            log_message(f"  Train Slices: {len(tr_msp_refs)} MSP + {len(neg_tr_slices)} non-MSP = {len(tr_all_slices)}", log_file)
            log_message(f"  Valid Slices: {len(va_msp_refs)} MSP + {len(neg_va_slices)} non-MSP = {len(va_all_slices)}", log_file)
        
        if not tr_all_slices:
            if log_file: log_message(f"  {fold_id} has an empty training slice set -> skipping fold.", log_file)
            continue

        fold_ckpt = paths["checkpoint_dir"] / fold_id
        fold_ckpt.mkdir(parents=True, exist_ok=True)
        fold_paths = paths.copy()
        fold_paths["checkpoint_dir"] = fold_ckpt
        
        fold_heatmap = train_heatmap_model_with_coverage_aware_training(tr_all_slices, va_all_slices, config, fold_paths)
        
        old_gate_threshold = config.get("AND_GATE_THRESHOLD")
        best_gate_thr = tune_and_gate_threshold_min_on_val(fold_heatmap, va_all_slices, config, device, log_file=log_file)
        
        _config_gate_backup = old_gate_threshold
        config["AND_GATE_THRESHOLD"] = float(best_gate_thr)
        
        # Train meta-classifier ONLY on training data to prevent leakage
        fold_meta = train_meta_classifier_full(fold_heatmap, tr_all_slices, [], config, fold_paths, fold=fold_id)

        fold_models_paths_info.append({"fold": fold_id, "heatmap_model": fold_heatmap, "meta_classifier": fold_meta})
        
        if va_all_slices:
            m_slice = test_fold_model(fold_heatmap, fold_meta, va_all_slices, config, device,
                                      test_patient_id_or_fold_id=fold_id, fold_num_for_output=fold_num,
                                      paths=fold_paths, log_file=log_file)
            per_fold_slice_metrics.append(m_slice)

        # Prepare case-level validation data for this fold
        fold_val_cases = []
        for pid in va_pids:
            for case_data in all_patient_groups.get(pid, []):
                label_path = case_data.get("label")
                has_msp = False
                msp_idx = -1
                if label_path:
                    lbl = load_nifti_data_cached(str(label_path), is_label=True)
                    if lbl is not None:
                        msp_idx = get_msp_index(lbl, config["SAGITTAL_AXIS"], config["STRUCTURE_LABELS"])
                        has_msp = (msp_idx >= 0)

                fold_val_cases.append({
                    "case_id": case_data["id"],
                    "volume_path": case_data["image"],
                    "label_path": label_path,
                    "true_case_has_msp": bool(has_msp),
                    "true_msp_slice_idx": int(msp_idx),
                    "patient_id": pid
                })

        # PATCH A: Create cache before scanning; scanning phase only collects scores
        det_cache = {}  # case_id -> detection output

        # SCAN: collect case_prob on this fold's VAL cases (no thresholding, no keypoints dump)
        fold_case_probs = []
        fold_case_labels = []

        if log_file: log_message(f"  Optimizing case threshold for {fold_id} on {len(fold_val_cases)} validation cases...", log_file)

        for vinfo in fold_val_cases:
            try:
                det = detect_msp_case_level_with_coverage(
                    fold_heatmap, fold_meta, vinfo['volume_path'],
                    config, log_file=None, keypoints_output_dir=None  # No keypoints during scanning
                )
                det_cache[vinfo['case_id']] = {'detection': det} 
                case_prob = det.get('case_level_decision', {}).get('case_prob', 0.0)
                fold_case_probs.append(float(case_prob))
                fold_case_labels.append(int(vinfo['true_case_has_msp']))
            except Exception as e:
                if log_file: log_message(f"    Error during threshold opt for {vinfo['case_id']}: {e}", log_file)
                continue

        # Determine optimal threshold for this fold
        if len(fold_case_probs) >= 2 and len(set(fold_case_labels)) > 1:
            threshold_result = find_optimal_case_threshold(
                fold_case_probs,
                [bool(l) for l in fold_case_labels],
                sens_min=config.get("CASE_SENS_MIN", 0.70),
                metric='f1'
            )
            if 'best_threshold' in threshold_result:
                fold_optimal_threshold = float(threshold_result['best_threshold'])
                if log_file:
                    log_message(f"  ✅ {fold_id} fold-optimal threshold: {fold_optimal_threshold:.4f} "
                              f"(F1={threshold_result.get('f1_score', 0):.3f}, "
                              f"Sens={threshold_result.get('sensitivity', 0):.3f})", log_file)
            else:
                fold_optimal_threshold = config.get("CASE_THRESHOLD_DEFAULT", 0.5)
                if log_file: log_message(f"  ⚠️ {fold_id} using default threshold: {fold_optimal_threshold:.4f}", log_file)
        else:
            fold_optimal_threshold = config.get("CASE_THRESHOLD_DEFAULT", 0.5)
            if log_file: log_message(f"  ⚠️ {fold_id} insufficient data, using default threshold: {fold_optimal_threshold:.4f}", log_file)
        
        per_fold_optimal_thresholds.append(fold_optimal_threshold)
        per_fold_thr_map[str(fold_num)] = float(fold_optimal_threshold)

        # PATCH B: Evaluate after determining the threshold, reusing cached detections
        # EVAL: apply fold_optimal_threshold on the same VAL cases; reuse cached detections
        key_dir = fold_paths["checkpoint_dir"] / "keypoints"
        key_dir.mkdir(parents=True, exist_ok=True)

        for vinfo in fold_val_cases:
            try:
                cached = det_cache.get(vinfo['case_id'])
                
                # Compatible with old cache (directly det) and new cache ({'detection': det, 'threshold': ...})
                if isinstance(cached, dict) and 'detection' in cached:
                    det = cached['detection']
                    thr_to_use = float(cached.get('threshold', fold_optimal_threshold))
                else:
                    det = cached
                    thr_to_use = float(fold_optimal_threshold)
                
                if det is None:
                    # Cache miss: re-run detection (this time will generate keypoint files)
                    if log_file:
                        log_message(f"    Warning: Cache miss for {vinfo['case_id']}, re-running detection", log_file)
                    det = detect_msp_case_level_with_coverage(
                        fold_heatmap, fold_meta, vinfo['volume_path'],
                        config, log_file=None, keypoints_output_dir=str(key_dir)
                    )
                    thr_to_use = float(fold_optimal_threshold)
                
                # Unify key names and sources: use 'has_msp' + 'threshold_used', and ensure 'case_prob' exists
                if det and 'case_level_decision' in det:
                    cld = det['case_level_decision']
                    case_prob = float(cld.get('case_prob', cld.get('case_prob_predicted', 0.0)))
                    cld['has_msp'] = (case_prob >= thr_to_use)
                    cld['threshold_used'] = thr_to_use
                    # Optional: sync a consistent field, prevent subsequent functions/visualization from reading old names
                    cld['case_prob_predicted'] = case_prob
                
                # Unify write-back cache structure (with threshold), ensure subsequent consistency
                det_cache[vinfo['case_id']] = {
                    'detection': det,
                    'threshold': thr_to_use
                }
                
                result = test_case_level(
                    fold_heatmap, fold_meta,
                    volume_info=vinfo, config=config, device=device,
                    test_patient_id_for_output=vinfo["patient_id"], fold_num_for_output=fold_num,
                    log_file=log_file, precomputed_detection_output=det
                )
                result['fold_optimal_threshold'] = float(thr_to_use)
                per_fold_case_results_list.append(result)

            except Exception as e:
                if log_file: log_message(f"    Case-level error pid={vinfo['patient_id']}, case={vinfo['case_id']}: {e}", log_file)
                per_fold_case_results_list.append({
                    "patient_id": vinfo["patient_id"],
                    "case_id": vinfo["case_id"],
                    "fold": fold_num,
                    "fold_optimal_threshold": fold_optimal_threshold,
                    "error": str(e)
                })

        # Restore original AND gate threshold
        if _config_gate_backup is not None:
            config["AND_GATE_THRESHOLD"] = _config_gate_backup

    # ===== Aggregate Results =====
    if log_file: log_message("\n--- 5-fold CV complete, aggregating... ---", log_file)

    # Define avg/std at aggregation start for scope safety
    avg_fold_threshold = float(config.get("CASE_THRESHOLD_DEFAULT", 0.5))
    std_fold_threshold = 0.0
    if per_fold_optimal_thresholds:
        avg_fold_threshold = float(np.mean(per_fold_optimal_thresholds))
        std_fold_threshold = float(np.std(per_fold_optimal_thresholds))
    config['CASE_THRESHOLD_OPTIMAL'] = avg_fold_threshold
    
    if log_file:
        if per_fold_optimal_thresholds:
            log_message(f"[PER-FOLD THRESHOLDS] Average: {avg_fold_threshold:.4f} ± {std_fold_threshold:.4f}", log_file)
            log_message(f"  Individual fold thresholds: {[f'{t:.3f}' for t in per_fold_optimal_thresholds]}", log_file)
        else:
            log_message("[WARNING] No fold thresholds computed, using default", log_file)
    np.save(paths["run_dir"] / "global_threshold_from_cv.npy", np.array(avg_fold_threshold, dtype=np.float32))
    paths["global_threshold_from_cv"] = paths["run_dir"] / "global_threshold_from_cv.npy"

    slice_df = build_lopo_results_df(pd.DataFrame(per_fold_slice_metrics)) if per_fold_slice_metrics else pd.DataFrame()
    case_df = pd.DataFrame(per_fold_case_results_list) if per_fold_case_results_list else pd.DataFrame()
    
    # Calculate summary metrics
    slice_summary = {}
    if not slice_df.empty:
        for m in ['auc', 'accuracy', 'sensitivity', 'specificity']:
            if m in slice_df.columns:
                slice_summary[f'mean_slice_{m}'] = float(slice_df[m].mean())
                slice_summary[f'std_slice_{m}'] = float(slice_df[m].std())

    case_summary = {}
    if not case_df.empty and ('true_case_has_msp' in case_df.columns) and ('case_prob_predicted' in case_df.columns):
        vc = case_df.copy()
        if 'error' in vc.columns:
            vc = vc[vc['error'].isna()]
        vc = vc.dropna(subset=['true_case_has_msp', 'case_prob_predicted'])
        
        if not vc.empty:
            y_true = vc['true_case_has_msp'].astype(str).str.strip().str.lower().map(
                {'1': True, '0': False, 'true': True, 'false': False, 't': True, 'f': False, 'yes': True, 'no': False}
            ).fillna(False).to_numpy(dtype=bool)
            y_prob = vc['case_prob_predicted'].astype(float).to_numpy()
            
            if np.unique(y_true).size > 1:
                case_summary['mean_case_auc'] = float(roc_auc_score(y_true, y_prob))
            
            y_pred = []
            for idx, row in vc.iterrows():
                case_thresh = row.get('fold_optimal_threshold', avg_fold_threshold)
                y_pred.append(row['case_prob_predicted'] >= case_thresh)
            
            y_pred = np.array(y_pred, dtype=bool)
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[False, True]).ravel()
            total = tp + tn + fp + fn
            case_summary['mean_case_accuracy'] = float((tp + tn) / total) if total > 0 else 0.0
            case_summary['case_sensitivity'] = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
            case_summary['case_specificity'] = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0

    # Save results
    slice_csv_path = None
    case_csv_path = None
    if not slice_df.empty:
        slice_csv_path = paths["run_dir"] / "5fold_slice_level_results_per_fold.csv"
        slice_df.to_csv(slice_csv_path, index=False, float_format='%.4f')

    if not case_df.empty:
        case_csv_path = paths["run_dir"] / "5fold_case_level_results_per_volume.csv"
        if 'true_case_has_msp' in case_df.columns:
            case_df['true_msp_label'] = case_df['true_case_has_msp'].astype(int)
        case_df.to_csv(case_csv_path, index=False, float_format='%.4f')

    # Persist per-fold thresholds using the unified path
    with open(paths["per_fold_thresholds_json"], "w") as f:
        json.dump(per_fold_thr_map, f, indent=2)

    final_combined_summary = {
        'mean_auc': slice_summary.get('mean_slice_auc', 0.0),
        'std_auc': slice_summary.get('std_slice_auc', 0.0),
        'mean_accuracy': slice_summary.get('mean_slice_accuracy', 0.0),
        'std_accuracy': slice_summary.get('std_slice_accuracy', 0.0),
        'mean_case_auc': case_summary.get('mean_case_auc', 0.0),
        'mean_case_accuracy': case_summary.get('mean_case_accuracy', 0.0),
        'case_sensitivity': case_summary.get('case_sensitivity', 0.0),
        'case_specificity': case_summary.get('case_specificity', 0.0),
        'per_fold_thresholds': per_fold_optimal_thresholds,
        'avg_fold_threshold': config['CASE_THRESHOLD_OPTIMAL'],
        'std_fold_threshold': std_fold_threshold
    }

    with open(paths["run_dir"] / "5fold_combined_summary.json", 'w') as f:
        json.dump(final_combined_summary, f, indent=2)

    # Visualization
    try:
        if not slice_df.empty: create_5fold_visualizations(slice_df, paths, log_file)
        if not case_df.empty: create_case_level_visualizations(case_df.to_dict('records'), paths, config, log_file)
    except Exception as e:
        if log_file: log_message(f"Error creating visualizations: {e}", log_file)

    if log_file: 
        log_message(f"5-fold cross-validation complete. Results saved in {paths['run_dir']}", log_file)
        log_message(f"✅ NO DATA LEAKAGE: Each fold used its own validation set for all optimizations.", log_file)
    
    return slice_csv_path, case_csv_path, final_combined_summary
def run_baseline_validation(config, paths):
    """Runs baseline validation with soft labels."""
    log_file = paths["log_file"]
    if log_file: log_message("Starting soft-label baseline validation with patient-level split...", log_file)

    # 1. Prepare MSP references
    train_msp_refs, val_msp_refs, all_patient_groups_info = prepare_patient_grouped_datasets(config, log_file)

    if not train_msp_refs and not val_msp_refs:
        if log_file: log_message(
            "ERROR: No MSP slice references found for training or validation in baseline. Aborting.", log_file)
        raise ValueError("Baseline validation cannot proceed without data.")

    # 2. Generate soft-label negative samples
    train_patient_ids_set = set(ref["patient_id"] for ref in train_msp_refs)
    val_patient_ids_set = set(ref["patient_id"] for ref in val_msp_refs)

    if log_file: log_message("  Generating soft-label negative samples for training set...", log_file)
    negative_train_refs = generate_negative_samples(
        positive_msp_refs=train_msp_refs,
        all_patient_groups=all_patient_groups_info,
        config=config,
        target_patient_ids=list(train_patient_ids_set),
        log_file=log_file
    )

    if log_file: log_message("  Generating soft-label negative samples for validation set...", log_file)
    negative_val_refs = generate_negative_samples(
        positive_msp_refs=val_msp_refs,
        all_patient_groups=all_patient_groups_info,
        config=config,
        target_patient_ids=list(val_patient_ids_set),
        samples_per_case_override=config.get("SAMPLES_PER_CASE_VAL", config["SAMPLES_PER_CASE"]),
        log_file=log_file
    )

    # 3. Combine samples
    combined_train_refs = train_msp_refs + negative_train_refs
    combined_val_refs = val_msp_refs + negative_val_refs
    np.random.shuffle(combined_train_refs)

    if log_file:
        log_message(
            f"  Training set: {len(train_msp_refs)} MSP + {len(negative_train_refs)} soft-label non-MSP = {len(combined_train_refs)} total slices.",
            log_file)
        log_message(
            f"  Validation set: {len(val_msp_refs)} MSP + {len(negative_val_refs)} soft-label non-MSP = {len(combined_val_refs)} total slices.",
            log_file)

    if not combined_train_refs:
        if log_file: log_message("ERROR: Training set is empty after adding negative samples. Aborting baseline.",
                                 log_file)
        raise ValueError("Empty training set for baseline validation.")

    # 4. Train model
    if log_file: log_message("  Training heatmap model with all constraints...", log_file)
    heatmap_model_output_path = train_heatmap_model_with_coverage_aware_training(
        combined_train_refs, combined_val_refs, config, paths
    )

    # 5. Train meta-classifier
    if log_file: log_message("  Training meta-classifier...", log_file)
    meta_classifier_output_path = train_meta_classifier_full(
        heatmap_model_output_path,
        combined_train_refs,
        combined_val_refs,
        config, paths,
        fold="baseline"
    )

    if log_file:
        log_message("Soft-label baseline validation process completed!", log_file)
        log_message(f"  Primary Heatmap Model: {heatmap_model_output_path}", log_file)
        log_message(f"  Meta-Classifier: {meta_classifier_output_path}", log_file)

    return heatmap_model_output_path, meta_classifier_output_path
