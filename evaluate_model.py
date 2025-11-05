"""
Model Evaluation Script for MSP Detection

This script evaluates a trained model on the validation set and computes metrics.

Usage:
    python evaluate_model.py --model_path checkpoints/best_baseline_model.pth
"""

import torch
import numpy as np
import argparse
from pathlib import Path
from tqdm import tqdm

from config import get_default_config
from utils.logging_utils import log_message
from train import prepare_patient_grouped_datasets, load_model_with_correct_architecture
from data import HeatmapDataset, create_balanced_dataloader
from eval import scan_slice_threshold_youden, find_optimal_case_threshold, evaluate_case_level


def collect_predictions(model, data_loader, device):
    """Collect predictions from model on data loader."""
    model.eval()

    all_slice_probs = []
    all_slice_labels = []
    all_case_ids = []

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Collecting predictions"):
            images = batch["image"].to(device)
            is_msp_labels = batch["is_msp_label"].numpy()
            case_ids = batch.get("case_id", ["unknown"] * len(images))

            # Forward pass
            outputs = model(images)

            if isinstance(outputs, tuple):
                _, cls_logits = outputs
                probs = torch.sigmoid(cls_logits).cpu().numpy().flatten()
            else:
                # If model only outputs heatmaps, use max probability
                heatmaps = outputs
                probs = torch.sigmoid(heatmaps).cpu().numpy()
                probs = probs.max(axis=(1, 2, 3))  # Max over C, H, W

            all_slice_probs.extend(probs)
            all_slice_labels.extend(is_msp_labels)
            all_case_ids.extend(case_ids)

    return np.array(all_slice_probs), np.array(all_slice_labels), all_case_ids


def compute_metrics(y_true, y_score, threshold):
    """Compute classification metrics."""
    y_pred = (y_score >= threshold).astype(int)

    TP = np.sum((y_pred == 1) & (y_true == 1))
    TN = np.sum((y_pred == 0) & (y_true == 0))
    FP = np.sum((y_pred == 1) & (y_true == 0))
    FN = np.sum((y_pred == 0) & (y_true == 1))

    accuracy = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) > 0 else 0.0
    sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    specificity = TN / (TN + FP) if (TN + FP) > 0 else 0.0
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    f1 = 2 * precision * sensitivity / (precision + sensitivity) if (precision + sensitivity) > 0 else 0.0

    return {
        'accuracy': accuracy,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'precision': precision,
        'f1': f1,
        'TP': TP,
        'TN': TN,
        'FP': FP,
        'FN': FN
    }


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description='Evaluate MSP detection model')
    parser.add_argument('--model_path', type=str, default='checkpoints/best_baseline_model.pth',
                        help='Path to trained model checkpoint')
    parser.add_argument('--image_dir', type=str, default=None,
                        help='Path to image directory (overrides config)')
    parser.add_argument('--label_dir', type=str, default=None,
                        help='Path to label directory (overrides config)')
    args = parser.parse_args()

    # ===== CONFIGURATION =====
    config = get_default_config()

    if args.image_dir:
        config["IMAGE_DIR"] = args.image_dir
    if args.label_dir:
        config["LABEL_DIR"] = args.label_dir

    config["DEVICE"] = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(config["DEVICE"])

    print("=" * 60)
    print("MSP Detection - Model Evaluation")
    print("=" * 60)
    print(f"Model: {args.model_path}")
    print(f"Device: {config['DEVICE']}")
    print(f"Image Directory: {config['IMAGE_DIR']}")
    print(f"Label Directory: {config['LABEL_DIR']}")
    print("=" * 60)

    # ===== LOAD MODEL =====
    print("\nLoading model...")
    model, model_type = load_model_with_correct_architecture(
        args.model_path,
        config,
        device,
        log_file=None
    )
    print(f"✅ Loaded model type: {model_type}")

    # ===== PREPARE DATA =====
    print("\nPreparing validation data...")
    train_refs, val_refs, patient_groups = prepare_patient_grouped_datasets(config, log_file=None)

    if not val_refs:
        print("ERROR: No validation data found!")
        return

    print(f"Validation slices: {len(val_refs)}")

    # Create validation dataset and loader
    val_dataset = HeatmapDataset(val_refs, config, is_train=False)
    val_loader = create_balanced_dataloader(val_dataset, config, is_train=False)

    # ===== COLLECT PREDICTIONS =====
    print("\nCollecting predictions...")
    slice_probs, slice_labels, case_ids = collect_predictions(model, val_loader, device)

    print(f"✅ Collected {len(slice_probs)} predictions")
    print(f"   Positive slices: {slice_labels.sum()} ({100*slice_labels.mean():.1f}%)")
    print(f"   Negative slices: {len(slice_labels) - slice_labels.sum()} ({100*(1-slice_labels.mean()):.1f}%)")

    # ===== SLICE-LEVEL EVALUATION =====
    print("\n" + "=" * 60)
    print("Slice-Level Evaluation")
    print("=" * 60)

    # Find optimal threshold using Youden's J
    best_thresh, results_df, best_idx = scan_slice_threshold_youden(
        y_true=slice_labels,
        y_score=slice_probs
    )

    print(f"\nOptimal slice threshold (Youden's J): {best_thresh:.4f}")

    # Compute metrics at optimal threshold
    metrics = compute_metrics(slice_labels, slice_probs, best_thresh)

    print(f"\nSlice-Level Metrics (threshold={best_thresh:.4f}):")
    print(f"  Accuracy:    {metrics['accuracy']:.4f}")
    print(f"  Sensitivity: {metrics['sensitivity']:.4f}")
    print(f"  Specificity: {metrics['specificity']:.4f}")
    print(f"  Precision:   {metrics['precision']:.4f}")
    print(f"  F1 Score:    {metrics['f1']:.4f}")
    print(f"\nConfusion Matrix:")
    print(f"  TP: {metrics['TP']:4d}  |  FP: {metrics['FP']:4d}")
    print(f"  FN: {metrics['FN']:4d}  |  TN: {metrics['TN']:4d}")

    # ===== CASE-LEVEL EVALUATION =====
    print("\n" + "=" * 60)
    print("Case-Level Evaluation")
    print("=" * 60)

    # Group predictions by case
    unique_cases = list(set(case_ids))
    print(f"\nUnique cases: {len(unique_cases)}")

    case_probs = []
    case_labels = []

    for case_id in unique_cases:
        # Get all slices for this case
        case_mask = np.array([cid == case_id for cid in case_ids])
        case_slice_probs = slice_probs[case_mask]
        case_slice_labels = slice_labels[case_mask]

        # Case probability = max slice probability
        case_prob = float(case_slice_probs.max())
        # Case label = 1 if any slice is MSP
        case_label = int(case_slice_labels.max())

        case_probs.append(case_prob)
        case_labels.append(case_label)

    case_probs = np.array(case_probs)
    case_labels = np.array(case_labels)

    print(f"   MSP cases: {case_labels.sum()}")
    print(f"   Non-MSP cases: {len(case_labels) - case_labels.sum()}")

    # Find optimal case threshold
    optimal_case = find_optimal_case_threshold(
        case_probs=case_probs.tolist(),
        true_case_labels=case_labels.tolist(),
        sens_min=0.7,  # Minimum sensitivity requirement
        metric='f1'
    )

    case_thresh = optimal_case['best_threshold']
    print(f"\nOptimal case threshold (F1): {case_thresh:.4f}")

    # Compute case-level metrics
    case_metrics = compute_metrics(case_labels, case_probs, case_thresh)

    print(f"\nCase-Level Metrics (threshold={case_thresh:.4f}):")
    print(f"  Accuracy:    {case_metrics['accuracy']:.4f}")
    print(f"  Sensitivity: {case_metrics['sensitivity']:.4f}")
    print(f"  Specificity: {case_metrics['specificity']:.4f}")
    print(f"  Precision:   {case_metrics['precision']:.4f}")
    print(f"  F1 Score:    {case_metrics['f1']:.4f}")
    print(f"\nConfusion Matrix:")
    print(f"  TP: {case_metrics['TP']:4d}  |  FP: {case_metrics['FP']:4d}")
    print(f"  FN: {case_metrics['FN']:4d}  |  TN: {case_metrics['TN']:4d}")

    # ===== SUMMARY =====
    print("\n" + "=" * 60)
    print("Evaluation Summary")
    print("=" * 60)
    print(f"✅ Slice-level F1: {metrics['f1']:.4f} (threshold={best_thresh:.4f})")
    print(f"✅ Case-level F1:  {case_metrics['f1']:.4f} (threshold={case_thresh:.4f})")
    print("=" * 60)


if __name__ == "__main__":
    main()
