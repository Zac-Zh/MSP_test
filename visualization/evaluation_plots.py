"""
Evaluation visualization functions for MSP detection.

Provides ROC curves, PR curves, confusion matrices, and threshold analysis plots.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_recall_curve, confusion_matrix
from pathlib import Path


def create_comprehensive_evaluation_visualization(case_probs, true_labels, optimal_threshold, paths, log_file=None):
    """
    Creates a comprehensive evaluation visualization including negative samples.

    Features:
    - ROC/PR curves (black & white, no legend)
    - Threshold scan: maximize F1 with Sensitivity>=0.70 constraint
    - Confusion matrix using the chosen threshold

    Args:
        case_probs: Array of predicted probabilities
        true_labels: Array of true labels
        optimal_threshold: Optimal threshold value
        paths: Dictionary with 'viz_dir' path
        log_file: Optional log file path

    Returns:
        None (saves figure to disk)
    """
    from utils.logging_utils import log_message

    viz_dir = paths["viz_dir"]
    viz_dir.mkdir(parents=True, exist_ok=True)

    # Global B/W style
    plt.rcParams.update({
        "axes.edgecolor": "black",
        "axes.labelcolor": "black",
        "xtick.color": "black",
        "ytick.color": "black",
        "text.color": "black",
    })

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Comprehensive Evaluation: MSP Detection with Positive and Negative Cases', fontsize=16)

    # ROC Curve
    ax1 = axes[0, 0]
    fpr, tpr, _ = roc_curve(true_labels, case_probs)
    ax1.plot(fpr, tpr, 'k-', lw=2)
    ax1.plot([0, 1], [0, 1], 'k--', lw=1)
    ax1.set_xlabel('False Positive Rate')
    ax1.set_ylabel('True Positive Rate')
    ax1.set_title('ROC Curve')
    ax1.grid(True, alpha=0.3)

    # PR Curve
    ax2 = axes[0, 1]
    precision, recall, _ = precision_recall_curve(true_labels, case_probs)
    ax2.plot(recall, precision, 'k-', lw=2)
    ax2.set_xlabel('Recall (Sensitivity)')
    ax2.set_ylabel('Precision')
    ax2.set_title('Precision-Recall Curve')
    ax2.grid(True, alpha=0.3)

    # Threshold scan
    ax3 = axes[1, 0]

    y_true = np.asarray(true_labels).astype(int)
    y_score = np.asarray(case_probs)
    uni_thr = np.unique(np.clip(y_score, 0, 1))
    thr_grid = np.linspace(0.0, 1.0, 1001)
    thresholds_scan = np.unique(np.r_[0.0, 1.0, uni_thr, thr_grid])

    sens_floor = 0.70
    f1_list, acc_list, sens_list, spec_list = [], [], [], []

    best_thr = 0.5
    best_f1, best_acc = -1.0, 0.0

    for t in thresholds_scan:
        y_pred = (y_score >= t).astype(int)
        tp = np.sum((y_true == 1) & (y_pred == 1))
        tn = np.sum((y_true == 0) & (y_pred == 0))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        fn = np.sum((y_true == 1) & (y_pred == 0))

        sens = tp / (tp + fn + 1e-9)
        spec = tn / (tn + fp + 1e-9)
        prec = tp / (tp + fp + 1e-9)
        f1   = 2 * prec * sens / (prec + sens + 1e-9)
        acc  = (tp + tn) / (tp + tn + fp + fn + 1e-9)

        f1_list.append(f1)
        acc_list.append(acc)
        sens_list.append(sens)
        spec_list.append(spec)

        # Candidate selection under sensitivity floor
        if sens >= sens_floor:
            if (f1 > best_f1) or (abs(f1 - best_f1) < 1e-9 and (acc > best_acc or (abs(acc - best_acc) < 1e-9 and t > best_thr))):
                best_f1, best_acc, best_thr = f1, acc, t

    f1_arr   = np.asarray(f1_list)
    acc_arr  = np.asarray(acc_list)
    sens_arr = np.asarray(sens_list)

    # Fallback if no threshold meets Sens floor
    if best_f1 < 0:
        idx = int(np.argmax(f1_arr))
        tie_idx = np.where(np.abs(f1_arr - f1_arr[idx]) < 1e-12)[0]
        if tie_idx.size > 1:
            acc_sub = acc_arr[tie_idx]
            idx2 = tie_idx[np.argmax(acc_sub)]
            thr_candidates = thresholds_scan[np.where((np.abs(f1_arr - f1_arr[idx]) < 1e-12) &
                                                       (np.abs(acc_arr - acc_arr[idx2]) < 1e-12))[0]]
            best_thr = float(np.max(thr_candidates))
        else:
            best_thr = float(thresholds_scan[idx])
        if log_file:
            log_message(f"[WARN] No threshold meets Sens ≥ {sens_floor:.2f}. "
                        f"Fallback to global F1-opt threshold = {best_thr:.3f}.", log_file)

    # Plot threshold scan
    ax3.plot(thresholds_scan, f1_arr,  'k-', lw=2)
    ax3.plot(thresholds_scan, acc_arr, 'k--', lw=1)
    ax3.axvline(best_thr,    color='k', linestyle=':', lw=1)
    ax3.axhline(sens_floor,  color='k', linestyle=':', lw=1)
    ax3.set_xlim(0, 1)
    ax3.set_ylim(0, 1)
    ax3.set_xlabel('Decision Threshold')
    ax3.set_ylabel('Score')
    ax3.set_title('Threshold Sensitivity Analysis')
    ax3.grid(True, alpha=0.3)

    decision_threshold = float(best_thr)

    # Confusion Matrix
    ax4 = axes[1, 1]
    y_pred_opt = (y_score >= decision_threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred_opt, labels=[0, 1])

    im = ax4.imshow(cm, cmap='Blues')
    ax4.set_xticks([0, 1])
    ax4.set_yticks([0, 1])
    ax4.set_xticklabels(['Pred No MSP', 'Pred Has MSP'])
    ax4.set_yticklabels(['True No MSP', 'True Has MSP'])
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax4.text(j, i, str(cm[i, j]),
                     ha='center', va='center',
                     color=('white' if cm[i, j] > cm.max()*0.6 else 'black'),
                     fontsize=11)
    ax4.set_title(f'Confusion Matrix (Threshold: {decision_threshold:.3f})')
    fig.colorbar(im, ax=ax4, fraction=0.046, pad=0.04)

    plt.tight_layout()
    output_path = viz_dir / "Comprehensive_evaluation_visualization.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    if log_file:
        log_message(f"Comprehensive evaluation visualization saved | chosen_thr={decision_threshold:.3f} (Sens≥{sens_floor})", log_file)


def save_final_roc_pr(y_true, y_score, out_path, title_prefix="Final"):
    """
    Save final ROC and PR curves.

    Args:
        y_true: True labels
        y_score: Predicted scores
        out_path: Output path for saving figure
        title_prefix: Prefix for plot titles

    Returns:
        None (saves figure to disk)
    """
    from sklearn.metrics import roc_auc_score, average_precision_score

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # ROC
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = roc_auc_score(y_true, y_score)
    axes[0].plot(fpr, tpr, 'b-', lw=2, label=f'AUC = {roc_auc:.3f}')
    axes[0].plot([0, 1], [0, 1], 'k--', lw=1)
    axes[0].set_xlabel('False Positive Rate')
    axes[0].set_ylabel('True Positive Rate')
    axes[0].set_title(f'{title_prefix} ROC Curve')
    axes[0].legend(loc='lower right')
    axes[0].grid(True, alpha=0.3)

    # PR
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    avg_pr = average_precision_score(y_true, y_score)
    axes[1].plot(recall, precision, 'b-', lw=2, label=f'AP = {avg_pr:.3f}')
    axes[1].set_xlabel('Recall')
    axes[1].set_ylabel('Precision')
    axes[1].set_title(f'{title_prefix} Precision-Recall Curve')
    axes[1].legend(loc='best')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()


def create_5fold_visualizations(lopo_results_df, paths, log_file=None):
    """
    Creates visualizations for 5-fold/GroupKFold validation results.

    Args:
        lopo_results_df: DataFrame with fold results
        paths: Dictionary with visualization directory path
        log_file: Optional log file path

    Returns:
        None (saves figures to disk)
    """
    from utils.logging_utils import log_message
    import pandas as pd

    if log_file:
        log_message("Creating 5-Fold/GroupKFold performance visualizations...", log_file)

    viz_dir = paths.get("viz_dir", Path("visualizations"))
    viz_dir.mkdir(parents=True, exist_ok=True)

    # Extract per-fold metrics
    unique_folds = sorted(lopo_results_df['fold'].unique())
    metrics_dict = {}

    for fold_id in unique_folds:
        fold_data = lopo_results_df[lopo_results_df['fold'] == fold_id]
        metrics_dict[fold_id] = {
            'accuracy': fold_data.get('accuracy', [0]).mean() if 'accuracy' in fold_data.columns else 0,
            'f1_score': fold_data.get('f1_score', [0]).mean() if 'f1_score' in fold_data.columns else 0,
            'sensitivity': fold_data.get('sensitivity', [0]).mean() if 'sensitivity' in fold_data.columns else 0,
            'specificity': fold_data.get('specificity', [0]).mean() if 'specificity' in fold_data.columns else 0,
        }

    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("5-Fold/GroupKFold Performance Metric Distributions", fontsize=16, y=0.98)

    fold_names = [str(f) for f in unique_folds]

    # Accuracy
    acc_vals = [metrics_dict[f]['accuracy'] for f in unique_folds]
    axes[0, 0].bar(fold_names, acc_vals, color='steelblue', alpha=0.7)
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].set_title('Accuracy per Fold')
    axes[0, 0].set_ylim(0, 1)
    axes[0, 0].grid(axis='y', alpha=0.3)

    # F1 Score
    f1_vals = [metrics_dict[f]['f1_score'] for f in unique_folds]
    axes[0, 1].bar(fold_names, f1_vals, color='coral', alpha=0.7)
    axes[0, 1].set_ylabel('F1 Score')
    axes[0, 1].set_title('F1 Score per Fold')
    axes[0, 1].set_ylim(0, 1)
    axes[0, 1].grid(axis='y', alpha=0.3)

    # Sensitivity
    sens_vals = [metrics_dict[f]['sensitivity'] for f in unique_folds]
    axes[1, 0].bar(fold_names, sens_vals, color='lightgreen', alpha=0.7)
    axes[1, 0].set_ylabel('Sensitivity')
    axes[1, 0].set_title('Sensitivity per Fold')
    axes[1, 0].set_ylim(0, 1)
    axes[1, 0].grid(axis='y', alpha=0.3)

    # Specificity
    spec_vals = [metrics_dict[f]['specificity'] for f in unique_folds]
    axes[1, 1].bar(fold_names, spec_vals, color='plum', alpha=0.7)
    axes[1, 1].set_ylabel('Specificity')
    axes[1, 1].set_title('Specificity per Fold')
    axes[1, 1].set_ylim(0, 1)
    axes[1, 1].grid(axis='y', alpha=0.3)

    plt.tight_layout()
    output_path = viz_dir / "5fold_performance_metrics.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    if log_file:
        log_message("5-Fold/GroupKFold visualizations created successfully.", log_file)
