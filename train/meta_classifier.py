"""
Meta-classifier training for MSP detection.

Uses LightGBM (or LogisticRegression as fallback) to classify slices based on
heatmap features extracted from the trained UNet model.
"""

import numpy as np
import torch
import pickle
import datetime
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import roc_auc_score

from utils.logging_utils import log_message
from data import preprocess_and_cache, extract_slice, normalize_slice, generate_brain_mask_from_image
from features import extract_heatmap_features
from inference import apply_tta_horizontal_flip
from eval import scan_slice_threshold_youden
import cv2


def collect_features_and_labels(heatmap_producing_model, data_refs, config, device, log_file=None):
    """
    Collect heatmap features and labels for meta-classifier training.

    Key improvements:
    1. Temporarily disable gating during training (avoid imbalanced training set)
    2. Explicit float32 type conversion
    3. Logits → prob conversion (semantic alignment)
    4. Dynamic feature dimension detection
    5. Quality filtering with per-label statistics
    6. Brain mask binarization

    Args:
        heatmap_producing_model: Trained UNet model
        data_refs: List of slice references
        config: Configuration dictionary
        device: torch device
        log_file: Optional log file path

    Returns:
        X: Feature matrix (N, D)
        y: Labels (N,)
    """
    if log_file:
        log_message("Collecting features with FULL diversity (gate disabled during training)...", log_file)

    heatmap_producing_model.eval()

    features_list = []
    labels_list = []

    # Statistics counters
    total_processed = 0
    passed_gate = 0
    failed_gate = 0
    failed_quality = 0
    dimension_mismatch = 0
    dropped_by_quality_pos = 0
    dropped_by_quality_neg = 0
    missing_label_count = 0

    expected_channels_for_heatmap = len(config["HEATMAP_LABEL_MAP"])
    thresholds = config.get("FEATURE_THRESHOLDS", [0.1, 0.3, 0.5, 0.7])

    # Dynamic feature dimension detection
    expected_feature_dim = None

    # KEY FIX 1: Temporarily disable gating
    original_gate_setting = config.get("ENABLE_STRUCTURE_GATE", True)
    original_gate_threshold = config.get("AND_GATE_THRESHOLD", 0.25)

    config["ENABLE_STRUCTURE_GATE"] = False  # ← Training mode: gate disabled

    if log_file:
        log_message(f"  [TRAINING MODE] Gate temporarily DISABLED", log_file)
        log_message(f"  Expected heatmap channels: {expected_channels_for_heatmap}", log_file)

    H_model, W_model = config["IMAGE_SIZE"]

    try:
        for ref_idx, ref in enumerate(tqdm(data_refs, desc="Collecting features")):
            try:
                total_processed += 1

                # KEY FIX 6: Label field robustness check
                if "is_msp" not in ref:
                    missing_label_count += 1
                    if log_file and missing_label_count <= 5:
                        log_message(f"  [WARN] Missing 'is_msp' field in ref {ref.get('case_id', '?')}, skipping", log_file)
                    continue

                current_label = 1 if ref.get("is_msp", False) else 0

                # Load and preprocess
                img_vol, _ = preprocess_and_cache(
                    str(ref["image_path"]), config["CACHE_DIR"], config, log_file
                )
                if img_vol is None:
                    failed_quality += 1
                    if current_label == 1:
                        dropped_by_quality_pos += 1
                    else:
                        dropped_by_quality_neg += 1
                    continue

                img_slice = extract_slice(img_vol, ref["slice_idx"], config["SAGITTAL_AXIS"])
                if img_slice is None:
                    failed_quality += 1
                    if current_label == 1:
                        dropped_by_quality_pos += 1
                    else:
                        dropped_by_quality_neg += 1
                    continue

                img_slice_norm = normalize_slice(img_slice, config)

                # Quality check (relaxed threshold)
                foreground_ratio = (img_slice_norm > 0.01).mean()
                if foreground_ratio < 0.02:  # Lowered from 0.05 to 0.02
                    failed_quality += 1
                    if current_label == 1:
                        dropped_by_quality_pos += 1
                    else:
                        dropped_by_quality_neg += 1
                    continue

                # Generate brain mask
                if config["GENERATE_BRAIN_MASK_FROM_IMAGE"]:
                    brain_mask_orig_res = generate_brain_mask_from_image(img_slice_norm, config)
                else:
                    brain_mask_orig_res = np.ones_like(img_slice_norm, dtype=np.float32)

                # KEY FIX 2: Explicit float32 + binarization
                img_resized = cv2.resize(
                    img_slice_norm, (W_model, H_model),
                    interpolation=cv2.INTER_LINEAR
                ).astype(np.float32)

                brain_mask_resized = cv2.resize(
                    brain_mask_orig_res, (W_model, H_model),
                    interpolation=cv2.INTER_NEAREST
                ).astype(np.float32)

                # KEY FIX 5: Binarization prevents interpolation contamination
                brain_mask_resized = (brain_mask_resized > 0.5).astype(np.float32)

                # Quality check: brain mask area
                if brain_mask_resized.sum() < 50:  # Lowered from 100 to 50
                    failed_quality += 1
                    if current_label == 1:
                        dropped_by_quality_pos += 1
                    else:
                        dropped_by_quality_neg += 1
                    continue

                img_tensor = torch.from_numpy(img_resized).unsqueeze(0).unsqueeze(0).float().to(device)

                # Model inference (with TTA)
                with torch.no_grad():
                    pred_heatmap_logits_tensor = apply_tta_horizontal_flip(
                        img_tensor,
                        heatmap_producing_model
                    )

                # KEY FIX 3: Smart logits → prob conversion
                model_output_type = config.get("MODEL_OUTPUT_ACTIVATION", "logits")

                if model_output_type == "logits":
                    pred_heatmap_probs_tensor = torch.sigmoid(pred_heatmap_logits_tensor)
                    activation_used = "sigmoid"
                elif model_output_type == "probs":
                    pred_heatmap_probs_tensor = pred_heatmap_logits_tensor.clamp(0.0, 1.0)
                    activation_used = "clamp"
                else:
                    # Heuristic detection (fallback)
                    t = pred_heatmap_logits_tensor
                    t_min, t_max = t.min().item(), t.max().item()

                    if (t_min < -0.1) or (t_max > 1.1):
                        pred_heatmap_probs_tensor = torch.sigmoid(t)
                        activation_used = "sigmoid(auto-detected)"
                        if log_file and ref_idx == 0:
                            log_message(
                                f"  [AUTO-DETECT] Model output range [{t_min:.3f}, {t_max:.3f}] detected as logits, applying sigmoid",
                                log_file
                            )
                    else:
                        pred_heatmap_probs_tensor = t.clamp(0.0, 1.0)
                        activation_used = "clamp(auto-detected)"
                        if log_file and ref_idx == 0:
                            log_message(
                                f"  [AUTO-DETECT] Model output range [{t_min:.3f}, {t_max:.3f}] detected as probs, skipping sigmoid",
                                log_file
                            )

                # First execution: log activation strategy
                if ref_idx == 0 and log_file:
                    log_message(f"  [ACTIVATION] Strategy: {activation_used}", log_file)
                    p_min, p_max = pred_heatmap_probs_tensor.min().item(), pred_heatmap_probs_tensor.max().item()
                    log_message(f"  [ACTIVATION] Output prob range: [{p_min:.4f}, {p_max:.4f}]", log_file)

                    # Double sigmoid detection
                    if 0.45 <= p_min and p_max <= 0.55:
                        log_message(
                            f"  ⚠️  WARNING: Probs highly concentrated around 0.5 - possible double sigmoid!",
                            log_file
                        )

                pred_heatmap_probs_np = pred_heatmap_probs_tensor.cpu().numpy()[0]  # (C, H, W)

                # Record gating status (don't filter)
                from eval import four_structure_and_gate_check
                gate_pass_check = four_structure_and_gate_check(
                    pred_heatmap_probs_np,
                    {**config, "ENABLE_STRUCTURE_GATE": True, "AND_GATE_THRESHOLD": original_gate_threshold}
                )

                if gate_pass_check:
                    passed_gate += 1
                else:
                    failed_gate += 1
                # ✅ Note: Don't continue, keep all samples

                # KEY FIX 4: Dynamic feature dimension detection (first time)
                if expected_feature_dim is None:
                    probe_features = extract_heatmap_features(
                        pred_heatmap_probs_np,
                        brain_mask_resized,
                        thresholds
                    )
                    expected_feature_dim = len(probe_features)
                    if log_file:
                        log_message(f"  [DYNAMIC] Detected feature dimension: {expected_feature_dim}", log_file)

                # Extract features
                feat_vec = extract_heatmap_features(
                    pred_heatmap_probs_np,
                    brain_mask_resized,
                    thresholds
                )

                if len(feat_vec) != expected_feature_dim:
                    dimension_mismatch += 1
                    if log_file and dimension_mismatch <= 3:
                        log_message(
                            f"  [WARN] Feature dimension mismatch: expected {expected_feature_dim}, got {len(feat_vec)}. Skipping.",
                            log_file
                        )
                    continue

                features_list.append(feat_vec)
                labels_list.append(current_label)

            except Exception as e:
                if log_file and total_processed <= 5:
                    log_message(f"  Error processing ref {ref.get('case_id', '?')}: {e}", log_file)
                continue

    finally:
        # Restore gating settings
        config["ENABLE_STRUCTURE_GATE"] = original_gate_setting
        config["AND_GATE_THRESHOLD"] = original_gate_threshold

    if log_file:
        log_message(f"\n  Feature collection complete:", log_file)
        log_message(f"    Total processed: {total_processed}", log_file)
        log_message(f"    Features collected: {len(features_list)}", log_file)
        log_message(f"    Gate pass/fail (recorded only): {passed_gate}/{failed_gate}", log_file)
        log_message(f"    Failed quality: {failed_quality} (pos={dropped_by_quality_pos}, neg={dropped_by_quality_neg})", log_file)
        log_message(f"    Dimension mismatch: {dimension_mismatch}", log_file)
        if missing_label_count > 0:
            log_message(f"    Missing label field: {missing_label_count}", log_file)

    if len(features_list) == 0:
        if log_file:
            log_message("  ERROR: No features collected!", log_file)
        return np.array([]), np.array([])

    X = np.array(features_list, dtype=np.float32)
    y = np.array(labels_list, dtype=np.int64)

    return X, y


def train_meta_classifier_full(heatmap_model_path, train_refs, val_refs, config, paths, fold="combined"):
    """
    Train meta-classifier for slice-level classification.

    Uses LightGBM (or LogisticRegression as fallback) on heatmap features.
    Optimizes threshold using Youden's J statistic.

    Args:
        heatmap_model_path: Path to trained heatmap model
        train_refs: Training slice references
        val_refs: Validation slice references
        config: Configuration dictionary
        paths: Dictionary with checkpoint_dir and log_file
        fold: Fold identifier

    Returns:
        Path to saved meta-classifier pickle file
    """
    log_file = paths["log_file"]
    if log_file:
        log_message(f"Training meta-classifier for run/fold: {fold}", log_file)

    device = torch.device(config["DEVICE"])

    # Load model with correct architecture
    from train import load_model_with_correct_architecture
    full_model_instance, model_type_loaded = load_model_with_correct_architecture(
        heatmap_model_path, config, device, log_file
    )

    # Determine the heatmap producing part
    if model_type_loaded in ['UNetWithCls', 'UNetWithCls_Stage1']:
        heatmap_producer_net = full_model_instance.unet
    elif model_type_loaded in ['UNetWithDualHeads', 'UNetWithDualHeads_Stage1']:
        heatmap_producer_net = full_model_instance.unet
    elif model_type_loaded == 'UNetHeatmap':
        heatmap_producer_net = full_model_instance
    else:
        raise ValueError(f"Unsupported model type {model_type_loaded}")

    heatmap_producer_net.eval()

    # Patch A: Temporarily disable gating for feature collection
    original_gate_setting = config.get("ENABLE_STRUCTURE_GATE", True)
    original_gate_threshold = config.get("AND_GATE_THRESHOLD", 0.25)

    config["ENABLE_STRUCTURE_GATE"] = False

    if log_file:
        log_message(f"  [TRAINING MODE] Gate temporarily DISABLED for feature collection", log_file)

    # Collect features (now without gating)
    X_train, y_train = collect_features_and_labels(
        heatmap_producer_net, train_refs, config, device, log_file
    )
    X_val, y_val = collect_features_and_labels(
        heatmap_producer_net, val_refs, config, device, log_file
    )

    # Restore gating settings
    config["ENABLE_STRUCTURE_GATE"] = original_gate_setting
    config["AND_GATE_THRESHOLD"] = original_gate_threshold

    if X_train.shape[0] == 0:
        if log_file:
            log_message("ERROR: No training features collected. Cannot train.", log_file)
        meta_clf_save_path = paths["checkpoint_dir"] / f"meta_classifier_{fold}_FAILED.pkl"
        with open(meta_clf_save_path, 'wb') as f:
            pickle.dump({'error': 'No training data'}, f)
        return str(meta_clf_save_path)

    # Ensure int dtype
    y_train = np.asarray(y_train, dtype=np.int64)
    y_val   = np.asarray(y_val,   dtype=np.int64)

    if log_file:
        log_message(f"  Collected {X_train.shape[0]} training samples, {X_val.shape[0]} validation samples", log_file)
        tr_hist = np.bincount(y_train, minlength=2)
        log_message(f"  Training labels: {tr_hist} (0s, 1s)", log_file)
        if y_val.size > 0:
            va_hist = np.bincount(y_val, minlength=2)
            log_message(f"  Validation labels: {va_hist} (0s, 1s)", log_file)

    # Train classifier
    try:
        import lightgbm as lgb

        neg_pos = (len(y_train) - y_train.sum()) / y_train.sum() if y_train.sum() > 0 else 1.0

        lgb_params = {
            'objective': 'binary',
            'metric': 'auc',
            'boosting_type': 'gbdt',
            'num_leaves': 15,
            'learning_rate': 0.05,
            'n_estimators': 100,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1,
            'random_state': config["SPLIT_SEED"],
            'scale_pos_weight': neg_pos,
            'min_child_samples': 5,
            'reg_alpha': 0.1,
            'reg_lambda': 0.1,
        }

        calibrated_clf = lgb.LGBMClassifier(**lgb_params)

        has_valid_val = (X_val.shape[0] > 0) and (len(np.unique(y_val)) > 1)
        if has_valid_val:
            calibrated_clf.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                eval_metric='auc',
                callbacks=[lgb.early_stopping(stopping_rounds=20, verbose=False)]
            )
        else:
            if log_file and X_val.shape[0] == 0:
                log_message("  No validation set – training without early stopping.", log_file)
            elif log_file:
                log_message("  Validation single-class – training without early stopping.", log_file)
            calibrated_clf.fit(X_train, y_train)

    except ImportError:
        if log_file:
            log_message("  LightGBM not available, using LogisticRegression", log_file)

        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import Pipeline

        calibrated_clf = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', LogisticRegression(
                solver='liblinear',
                max_iter=1000,
                class_weight='balanced',
                random_state=config["SPLIT_SEED"],
                C=0.1
            ))
        ])
        calibrated_clf.fit(X_train, y_train)

    # Patch B: Use Youden strategy to optimize threshold
    optimal_threshold = 0.5
    threshold_stats = {}

    if X_val.shape[0] > 0 and len(np.unique(y_val)) > 1:
        y_val_pred_proba = calibrated_clf.predict_proba(X_val)[:, 1]
        val_auc = roc_auc_score(y_val, y_val_pred_proba)

        if log_file:
            log_message(f"  Validation AUC: {val_auc:.4f}", log_file)

        # Save ROC data
        roc_data = {
            'y_true': y_val.tolist(),
            'y_pred_proba': y_val_pred_proba.tolist(),
            'fold': fold,
            'timestamp': datetime.datetime.now().isoformat()
        }

        roc_data_path = paths["checkpoint_dir"] / f"roc_data_{fold}.pkl"
        with open(roc_data_path, 'wb') as f:
            pickle.dump(roc_data, f)

        if log_file:
            log_message(f"  ROC data saved to: {roc_data_path}", log_file)

        # Use Youden strategy to scan threshold
        num_steps = int(config.get("SLICE_THRESHOLD_NUM_STEPS", 1001))
        optimal_threshold, threshold_results_df, best_idx = scan_slice_threshold_youden(
            y_val, y_val_pred_proba, num_steps=num_steps
        )

        # Extract statistics
        optimal_row = threshold_results_df.loc[best_idx]
        threshold_stats = {
            'optimal_threshold': float(optimal_threshold),
            'sensitivity': float(optimal_row['sensitivity']),
            'specificity': float(optimal_row['specificity']),
            'precision': float(optimal_row['precision']),
            'f1_score': float(optimal_row['f1']),
            'youden_j': float(optimal_row['youden']),
            'validation_auc': float(val_auc),
            'pos_ratio': float(np.mean(y_val)),
            'strategy': 'youden_maximization'
        }

        if log_file:
            log_message(f"  Optimal threshold (Youden's J): {optimal_threshold:.4f}", log_file)
            log_message(f"  Youden's J: {threshold_stats['youden_j']:.4f}", log_file)
            log_message(f"  Sensitivity: {threshold_stats['sensitivity']:.4f}", log_file)
            log_message(f"  Specificity: {threshold_stats['specificity']:.4f}", log_file)

    # Write back slice decision threshold
    config["SLICE_CLS_THRESHOLD"] = float(optimal_threshold)

    # Save meta-classifier
    meta_clf_save_path = paths["checkpoint_dir"] / f"meta_classifier_{fold}.pkl"

    meta_classifier_data = {
        'classifier': calibrated_clf,
        'slice_threshold_calibrated': float(optimal_threshold),
        'threshold_optimization_stats': threshold_stats,
        'feature_dim': X_train.shape[1],
        'config_at_save_time': config,
        'fold_identifier': fold,
        'model_type_of_feature_extractor': model_type_loaded,
        'source_heatmap_model_path': str(heatmap_model_path),
    }

    with open(meta_clf_save_path, 'wb') as f:
        pickle.dump(meta_classifier_data, f)

    if log_file:
        log_message(f"  Meta-classifier saved: {meta_clf_save_path}", log_file)

    return str(meta_clf_save_path)
