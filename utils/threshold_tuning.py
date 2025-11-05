"""
Threshold tuning utilities for gating and classification.

Provides functions for optimizing thresholds on validation data.
"""

import numpy as np
import torch
import cv2

from data.preprocessing import preprocess_and_cache, extract_slice, normalize_slice
from train.helpers import load_model_with_correct_architecture
from inference.tta import apply_tta_horizontal_flip
from utils.logging_utils import log_message


def tune_and_gate_threshold_min_on_val(heatmap_model_path, val_refs, config, device, log_file=None,
                                       sens_floor=None, metric="f1"):
    """
    Tunes the AND gate threshold using min aggregation on validation set.

    This function:
    1. Computes g = min(max_prob_per_required_label) for each validation slice
    2. Scans thresholds on g to maximize F1 score with sensitivity >= sens_floor constraint
    3. Returns optimal threshold for the AND gate

    Strategy:
    - For each slice, compute max probability for each of the 4 required structures
    - Take min of these 4 values as the gating score g
    - Scan all unique g values as candidate thresholds
    - Find threshold that maximizes F1 while maintaining sens >= sens_floor

    Args:
        heatmap_model_path: Path to trained heatmap model
        val_refs: List of validation slice references
        config: Configuration dictionary
        device: PyTorch device
        log_file: Optional log file path
        sens_floor: Minimum sensitivity constraint (default from config)
        metric: Metric to optimize ('f1' or 'youden')

    Returns:
        float: Optimal AND gate threshold
    """
    if sens_floor is None:
        sens_floor = float(config.get("AND_GATE_SENS_FLOOR", 0.70))

    # Load model
    full_model, model_type = load_model_with_correct_architecture(heatmap_model_path, config, device, log_file)
    full_model.eval()

    label_to_channel = config["LABEL_TO_CHANNEL"]
    required_labels = [int(x) for x in config["MSP_REQUIRED_LABELS"]]
    H_model, W_model = config["IMAGE_SIZE"]

    g_list, y_list = [], []

    for ref in val_refs:
        try:
            img_vol, _ = preprocess_and_cache(str(ref["image_path"]), config["CACHE_DIR"], config, log_file)
            if img_vol is None:
                continue
            img_slice = extract_slice(img_vol, ref["slice_idx"], config["SAGITTAL_AXIS"])
            if img_slice is None:
                continue

            # Preprocess
            img_slice_norm = normalize_slice(img_slice, config)
            img_resized = cv2.resize(img_slice_norm, (W_model, H_model), interpolation=cv2.INTER_LINEAR)
            img_tensor = torch.from_numpy(img_resized).unsqueeze(0).unsqueeze(0).to(device)

            # Forward pass (consistent with testing: TTA horizontal flip)
            with torch.no_grad():
                if model_type in ['UNetWithCls', 'UNetWithCls_Stage1']:
                    heatmap_logits, _ = apply_tta_horizontal_flip(img_tensor, full_model)
                elif model_type in ['UNetWithDualHeads', 'UNetWithDualHeads_Stage1']:
                    outs = apply_tta_horizontal_flip(img_tensor, full_model)
                    heatmap_logits = outs[0] if isinstance(outs, (list, tuple)) else outs
                else:
                    heatmap_logits = apply_tta_horizontal_flip(img_tensor, full_model)

            heatmap_logits_np = heatmap_logits.cpu().numpy()[0]  # (C,H,W)
            pred_heatmaps_probs = 1.0 / (1.0 + np.exp(-heatmap_logits_np))
            C = pred_heatmaps_probs.shape[0]

            # Compute max prob for 4 structures, then take min as aggregated g
            structure_max = []
            for lab in required_labels:
                ch = label_to_channel.get(lab, None)
                if ch is None or ch >= C:
                    structure_max.append(0.0)
                else:
                    structure_max.append(float(pred_heatmaps_probs[ch].max()))
            g = min(structure_max) if structure_max else 0.0

            g_list.append(g)
            y_list.append(1 if ref.get("is_msp", False) else 0)

        except Exception as e:
            if log_file:
                log_message(f"[AND-GATE TUNE] skip slice: {e}", log_file)
            continue

    if not g_list:
        if log_file:
            log_message("[AND-GATE TUNE] empty val refs, keep existing threshold.", log_file)
        return float(config.get("AND_GATE_THRESHOLD", 0.25))

    g_arr = np.array(g_list, dtype=np.float32)
    y_arr = np.array(y_list, dtype=np.int32)

    # Full scan on all unique g values
    candidates = np.unique(g_arr)
    best = {"thr": None, "score": -1.0, "sens": None, "spec": None, "prec": None, "rec": None}

    P = int((y_arr == 1).sum())
    N = int((y_arr == 0).sum())

    def eval_point(t):
        yhat = (g_arr >= t).astype(np.int32)
        TP = int(((yhat == 1) & (y_arr == 1)).sum())
        FP = int(((yhat == 1) & (y_arr == 0)).sum())
        FN = int(((yhat == 0) & (y_arr == 1)).sum())
        TN = int(((yhat == 0) & (y_arr == 0)).sum())
        sens = TP / P if P else 0.0
        spec = TN / N if N else 0.0
        prec = TP / (TP + FP) if (TP + FP) else 0.0
        rec = sens
        if sens < sens_floor:
            return -1.0, (sens, spec, prec, rec)
        if metric == "f1":
            score = (2 * prec * rec) / (prec + rec) if (prec + rec) else 0.0
        elif metric == "youden":
            score = sens + spec - 1.0
        else:
            score = (2 * prec * rec) / (prec + rec) if (prec + rec) else 0.0
        return score, (sens, spec, prec, rec)

    for t in candidates:
        score, stats = eval_point(t)
        if score > best["score"]:
            best.update({"thr": float(t), "score": float(score),
                         "sens": float(stats[0]), "spec": float(stats[1]),
                         "prec": float(stats[2]), "rec": float(stats[3])})

    # Fallback if no point meets sens floor: choose max sens, tie-break with F1
    if best["thr"] is None:
        if log_file:
            log_message(f"[AND-GATE TUNE] no t meets sens >= {sens_floor:.2f}, fallback to max-sens.", log_file)
        best_sens, best_f1, best_thr = -1.0, -1.0, None
        for t in candidates:
            yhat = (g_arr >= t).astype(np.int32)
            TP = int(((yhat == 1) & (y_arr == 1)).sum())
            FP = int(((yhat == 1) & (y_arr == 0)).sum())
            FN = int(((yhat == 0) & (y_arr == 1)).sum())
            TN = int(((yhat == 0) & (y_arr == 0)).sum())
            sens = TP / P if P else 0.0
            prec = TP / (TP + FP) if (TP + FP) else 0.0
            rec = sens
            f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) else 0.0
            if (sens > best_sens) or (abs(sens - best_sens) <= 1e-9 and f1 > best_f1):
                best_sens, best_f1, best_thr = sens, f1, float(t)
        best["thr"] = best_thr

    if log_file:
        log_message(f"[AND-GATE TUNE] best_thr={best['thr']:.4f} | metric={metric} | sens_floor={sens_floor}", log_file)

    return best["thr"]
