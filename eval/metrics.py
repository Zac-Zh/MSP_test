"""
Evaluation metrics and threshold optimization for MSP detection.

Provides functions for computing performance metrics, optimizing decision
thresholds, and evaluating predictions at both slice and case levels.
"""

import numpy as np
import pandas as pd
import torch
import pickle
import datetime
from typing import List
from sklearn.metrics import f1_score, confusion_matrix


def scan_slice_threshold_youden(y_true: np.ndarray, y_score: np.ndarray, num_steps: int = 1001):
    """
    Slice-level threshold scanning WITHOUT gate.
    Select threshold by maximizing Youden's J = Sensitivity + Specificity - 1.
    Returns: best_threshold, results_df, best_idx
    """
    thresholds = np.linspace(0.0, 1.0, num_steps)
    sens_list, spec_list, youden_list, prec_list, f1_list = [], [], [], [], []

    P = (y_true == 1).sum()
    N = (y_true == 0).sum()

    for t in thresholds:
        yb = (y_score >= t).astype(np.uint8)
        TP = int(((yb == 1) & (y_true == 1)).sum())
        TN = int(((yb == 0) & (y_true == 0)).sum())
        FP = int(((yb == 1) & (y_true == 0)).sum())
        FN = int(((yb == 0) & (y_true == 1)).sum())

        sens = TP / (TP + FN) if (TP + FN) > 0 else 0.0            # recall
        spec = TN / (TN + FP) if (TN + FP) > 0 else 0.0
        prec = TP / (TP + FP) if (TP + FP) > 0 else 0.0
        f1   = (2*prec*sens)/(prec+sens) if (prec+sens) > 0 else 0.0
        J    = sens + spec - 1.0

        sens_list.append(sens)
        spec_list.append(spec)
        prec_list.append(prec)
        f1_list.append(f1)
        youden_list.append(J)

    results_df = pd.DataFrame({
        "threshold": thresholds,
        "sensitivity": sens_list,
        "specificity": spec_list,
        "precision": prec_list,
        "f1": f1_list,
        "youden": youden_list
    })

    # 选 J 最大；若并列，优先更高的 sensitivity，其次更高的 specificity
    best_idx = int(np.argmax(results_df["youden"].values))
    maxJ = results_df.loc[best_idx, "youden"]
    ties = results_df.index[results_df["youden"].values >= (maxJ - 1e-9)].tolist()

    # break ties by higher sensitivity then specificity
    if len(ties) > 1:
        sub = results_df.loc[ties]
        best_idx = int(sub.sort_values(
            ["sensitivity", "specificity", "threshold"],
            ascending=[False, False, True]
        ).index[0])

    return float(results_df.loc[best_idx, "threshold"]), results_df, int(best_idx)


def collect_and_store_roc_data(model, test_loader, device, output_path, config):
    """Collect ROC data from model predictions on test loader."""
    from inference import apply_tta_horizontal_flip

    model.eval()
    all_predictions, all_true_labels = [], []
    label_to_channel = config["LABEL_TO_CHANNEL"]
    required = [int(x) for x in config["MSP_REQUIRED_LABELS"]]

    with torch.no_grad():
        for batch in test_loader:
            images = batch["image"].to(device)
            y = batch["is_msp_label"].cpu().numpy().astype(int).flatten()

            # 统一与验证/推理的 TTA 路径
            outs = apply_tta_horizontal_flip(images, model)
            if hasattr(model, 'cls_head'):
                # UNetWithCls / DualHeads
                if isinstance(outs, (tuple, list)):
                    heatmaps, cls_logit = outs[0], outs[1]
                else:
                    raise RuntimeError("Unexpected model outputs")
                scores = torch.sigmoid(cls_logit).cpu().numpy().flatten()
            else:
                # Heatmap-only: use min of per-structure max probabilities as score g
                if isinstance(outs, (tuple, list)):
                    heatmaps = outs[0]
                else:
                    heatmaps = outs
                probs = torch.sigmoid(heatmaps).cpu().numpy()   # [B,C,H,W]
                g_list = []
                for b in range(probs.shape[0]):
                    vals = []
                    for lab in required:
                        ch = label_to_channel.get(lab, None)
                        vals.append(probs[b, ch].max() if ch is not None and ch < probs.shape[1] else 0.0)
                    g_list.append(min(vals))
                scores = np.array(g_list, dtype=np.float32)

            all_predictions.extend(scores.tolist())
            all_true_labels.extend(y.tolist())

    roc_data = {'y_true': all_true_labels,
                'y_pred_proba': all_predictions,
                'timestamp': datetime.datetime.now().isoformat()}
    with open(output_path, 'wb') as f:
        pickle.dump(roc_data, f)
    return all_true_labels, all_predictions


def evaluate_case_level(volume_slice_probs: List[float], case_threshold: float = 0.5):
    """Case-level MSP decision based on the highest slice probability strategy."""
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


def compute_optimal_case_threshold(all_case_results, config, paths, log_file=None):
    """Computes the optimal case-level threshold using pure F1 maximization."""
    from utils.logging_utils import log_message

    if log_file: log_message("Computing optimal case-level threshold using pure F1 maximization...", log_file)

    case_probs = []
    true_labels = []

    for case_result in all_case_results:
        if 'error' in case_result:
            continue
        case_probs.append(case_result.get('case_prob_predicted', 0.0))
        true_labels.append(int(case_result.get('true_case_has_msp', False)))

    if len(case_probs) < 2:
        if log_file: log_message("ERROR: Insufficient data for threshold optimization", log_file)
        return config.get("CASE_THRESHOLD_DEFAULT", 0.5), {}

    case_probs = np.array(case_probs)
    true_labels = np.array(true_labels)

    # Scan all unique probability points directly to find the max F1
    thresholds = np.unique(case_probs)
    f1_scores = []

    for th in thresholds:
        pred = (case_probs >= th).astype(int)
        f1_scores.append(f1_score(true_labels, pred, zero_division=0))

    best_idx = int(np.argmax(f1_scores))
    optimal_threshold = float(thresholds[best_idx])
    best_f1_score = float(f1_scores[best_idx])

    # Calculate performance at the optimal threshold
    y_pred_optimal = (case_probs >= optimal_threshold).astype(int)
    cm = confusion_matrix(true_labels, y_pred_optimal)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)

    stats = {
        'optimal_threshold': optimal_threshold,
        'best_f1_score': best_f1_score,
        'accuracy': float((tp + tn) / (tp + tn + fp + fn)) if (tp + tn + fp + fn) > 0 else 0.0,
        'sensitivity': float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0,
        'specificity': float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0,
        'tp': int(tp), 'tn': int(tn), 'fp': int(fp), 'fn': int(fn)
    }

    if log_file:
        log_message(f"  Optimal threshold (F1-max): {optimal_threshold:.4f}", log_file)
        log_message(f"  Best F1 score: {best_f1_score:.4f}", log_file)
        log_message(
            f"  Accuracy: {stats['accuracy']:.3f}, Sensitivity: {stats['sensitivity']:.3f}, Specificity: {stats['specificity']:.3f}",
            log_file)

    return optimal_threshold, stats


def find_optimal_case_threshold(case_probs: List[float],
                                true_case_labels: List[bool],
                                spec_weight: float = None, # No longer used, but kept for signature compatibility
                                spec_min: float = 0.0,     # No longer used
                                sens_min: float = 0.0,
                                metric: str = 'f1') -> dict:
    """
    REVISED: Chooses the optimal threshold by maximizing F1-score, constrained by a minimum
    sensitivity (`sens_min`).

    KEY FIXES:
    1.  Primary strategy is now F1-maximization under a sensitivity floor.
    2.  If no threshold meets the floor, it gracefully falls back to a max-sensitivity strategy,
        with F1-score and then specificity as tie-breakers.
    3.  More robust calculation of metrics to avoid division by zero.
    """
    if not case_probs or not true_case_labels:
        return {'best_threshold': 0.5, 'error': 'Empty input data'}

    pred = np.asarray(case_probs, dtype=np.float64)
    label = np.asarray(true_case_labels, dtype=bool)

    if len(np.unique(label)) < 2:
        return {'best_threshold': 0.5, 'error': 'Only one class present in labels'}

    # Use all unique prediction values as candidate thresholds
    thresholds = np.unique(pred)
    eps = 1e-9

    P = np.sum(label)
    N = len(label) - P

    best_candidate = None

    # --- Primary Strategy: Find best F1 score for thresholds meeting the sensitivity floor ---
    candidates_meeting_floor = []
    for t in thresholds:
        y_hat = (pred >= t)
        tp = np.sum((y_hat == 1) & (label == 1))

        sens = tp / (P + eps)

        if sens >= sens_min:
            tn = np.sum((y_hat == 0) & (label == 0))
            fp = N - tn
            prec = tp / (tp + fp + eps)
            spec = tn / (N + eps)
            f1 = 2 * (prec * sens) / (prec + sens + eps)
            youden = sens + spec - 1.0
            candidates_meeting_floor.append({
                'threshold': t, 'f1_score': f1, 'sensitivity': sens,
                'specificity': spec, 'precision': prec, 'youden_j': youden
            })

    if candidates_meeting_floor:
        # Sort by F1-score (desc), then sensitivity (desc) as tie-breaker
        candidates_meeting_floor.sort(key=lambda x: (-x['f1_score'], -x['sensitivity']))
        best_candidate = candidates_meeting_floor[0]

    else:
        # --- Fallback Strategy: No threshold met the sensitivity floor ---
        # Find the threshold that maximizes sensitivity, with F1 and then specificity as tie-breakers.
        all_candidates = []
        for t in thresholds:
            y_hat = (pred >= t)
            tp = np.sum((y_hat == 1) & (label == 1))
            tn = np.sum((y_hat == 0) & (label == 0))
            fp = N - tn
            sens = tp / (P + eps)
            prec = tp / (tp + fp + eps)
            spec = tn / (N + eps)
            f1 = 2 * (prec * sens) / (prec + sens + eps)
            youden = sens + spec - 1.0
            all_candidates.append({
                'threshold': t, 'f1_score': f1, 'sensitivity': sens,
                'specificity': spec, 'precision': prec, 'youden_j': youden
            })

        if all_candidates:
            # Sort by sensitivity (desc), then F1 (desc), then specificity (desc)
            all_candidates.sort(key=lambda x: (-x['sensitivity'], -x['f1_score'], -x['specificity']))
            best_candidate = all_candidates[0]

    if best_candidate:
        return {
            'best_threshold': float(best_candidate['threshold']),
            'f1_score': float(best_candidate['f1_score']),
            'sensitivity': float(best_candidate['sensitivity']),
            'specificity': float(best_candidate['specificity']),
            'precision': float(best_candidate['precision']),
            'youden_j': float(best_candidate['youden_j']),
            'strategy_used': 'F1-max_with_sens_floor' if candidates_meeting_floor else 'Fallback_max_sensitivity'
        }

    return {'best_threshold': 0.5, 'error': 'Could not determine a valid threshold'}


def adaptive_threshold_search(case_probs: List[float],
                              true_case_labels: List[bool],
                              target_spec: float = None,
                              target_sens: float = None) -> dict:
    """
    Adaptive threshold search: dynamically adjusts strategy based on data distribution.

    Args:
        case_probs: Predicted probabilities
        true_case_labels: Ground truth labels
        target_spec: Target specificity (if specified)
        target_sens: Target sensitivity (if specified)
    """
    y = np.asarray(true_case_labels, dtype=bool)
    pos_ratio = np.mean(y)

    # Dynamically adjust strategy based on class imbalance
    if pos_ratio < 0.1:  # Extremely imbalanced: positive < 10%
        # Prioritize sensitivity with low positive rates
        spec_weight = 0.3
        spec_min = 0.1
        sens_min = 0.8  # High sensitivity requirement
        metric = 'weighted'
    elif pos_ratio > 0.9:  # Extremely imbalanced: positive > 90%
        # Prioritize specificity with high positive rates
        spec_weight = 0.9
        spec_min = 0.3
        sens_min = 0.6  # Relatively lower sensitivity requirement
        metric = 'weighted'
    else:  # Relatively balanced
        # Use Youden's index to balance sensitivity and specificity
        spec_weight = 0.5
        spec_min = 0.3
        sens_min = 0.7  # Standard sensitivity requirement
        metric = 'youden'

    # Adjust strategy if there are specific targets
    if target_spec is not None:
        spec_min = target_spec
        spec_weight = 0.8
    if target_sens is not None:
        # If sensitivity is required, slightly lower the specificity weight
        sens_min = target_sens
        spec_weight = max(0.3, spec_weight - 0.2)

    result = find_optimal_case_threshold(
        case_probs, true_case_labels,
        spec_weight=spec_weight,
        spec_min=spec_min,
        sens_min=sens_min,
        metric=metric
    )

    result['adaptive_strategy'] = {
        'pos_ratio': float(pos_ratio),
        'chosen_spec_weight': spec_weight,
        'chosen_spec_min': spec_min,
        'chosen_sens_min': sens_min,
        'chosen_metric': metric,
        'rationale': f'pos_ratio={pos_ratio:.3f}, strategy={"imbalanced" if pos_ratio < 0.1 or pos_ratio > 0.9 else "balanced"}'
    }

    return result
