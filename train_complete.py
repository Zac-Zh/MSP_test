"""
Complete Training Pipeline for MSP Detection

Implements the full research pipeline:
- 5-fold patient-grouped cross-validation
- 2-stage training (Stage 1: heatmap, Stage 2: joint with coverage)
- Negative sample generation for balanced training
- UNetWithDualHeads architecture
- Coverage-aware loss functions

Usage:
    python train_complete.py
"""

import torch
import numpy as np
from pathlib import Path
from sklearn.model_selection import GroupKFold
from datetime import datetime

from config import get_default_config
from utils.logging_utils import log_message, setup_logging
from utils.io_utils import find_nifti_pairs
from train import (
    generate_negative_samples,
    train_stage1_heatmap,
    train_stage2_joint
)
from data import HeatmapDataset, create_balanced_dataloader
from models import UNetWithDualHeads


def run_5fold_cv_with_staged_training(config, paths):
    """
    Run 5-fold cross-validation with 2-stage training for each fold.

    This is the complete research pipeline matching the original main.py.

    Args:
        config: Configuration dictionary
        paths: Dictionary with run_dir, checkpoint_dir, log_file

    Returns:
        List of fold results with metrics
    """
    log_file = paths["log_file"]

    if log_file:
        log_message("="*80, log_file)
        log_message("COMPLETE 5-FOLD CV WITH 2-STAGE TRAINING", log_file)
        log_message("="*80, log_file)

    # Get all patient data
    all_pairs = find_nifti_pairs(config["IMAGE_DIR"], config["LABEL_DIR"], log_file)

    if not all_pairs:
        raise ValueError("No NIfTI pairs found! Check IMAGE_DIR and LABEL_DIR.")

    # Group by patient
    patient_groups = {}
    for pair in all_pairs:
        patient_id = pair["id"]
        if patient_id not in patient_groups:
            patient_groups[patient_id] = []
        patient_groups[patient_id].append(pair)

    all_patient_ids = list(patient_groups.keys())

    if not all_patient_ids:
        raise ValueError("No patients found!")

    if log_file:
        log_message(f"Total unique patients: {len(all_patient_ids)}", log_file)

    n_cv_splits = config.get("KFOLD_SPLITS", 5)
    if len(all_patient_ids) < n_cv_splits:
        n_cv_splits = len(all_patient_ids)
        if log_file:
            log_message(f"WARNING: Only {len(all_patient_ids)} patients, using {n_cv_splits}-fold", log_file)

    if n_cv_splits < 2:
        raise ValueError(f"Not enough patients ({len(all_patient_ids)}) for CV!")

    # Setup GroupKFold
    gkf = GroupKFold(n_splits=n_cv_splits)
    patients_np = np.array(all_patient_ids)

    fold_results = []
    device = torch.device(config["DEVICE"])

    # Iterate through folds
    for fold_idx, (train_indices, val_indices) in enumerate(gkf.split(patients_np, groups=patients_np)):
        fold_num = fold_idx + 1
        fold_id = f"fold_{fold_num}"

        train_patient_ids = patients_np[train_indices].tolist()
        val_patient_ids = patients_np[val_indices].tolist()

        # Verify no overlap
        assert set(train_patient_ids).isdisjoint(val_patient_ids), \
            "Patient overlap detected!"

        if log_file:
            log_message(f"\n{'='*80}", log_file)
            log_message(f"FOLD {fold_num}/{n_cv_splits}", log_file)
            log_message(f"{'='*80}", log_file)
            log_message(f"Train patients: {len(train_patient_ids)}, Val patients: {len(val_patient_ids)}", log_file)

        # Build MSP references (positive samples)
        train_msp_refs = []
        for pid in train_patient_ids:
            for pair_info in patient_groups.get(pid, []):
                if pair_info.get("label") and pair_info.get("has_msp", False):
                    msp_idx = pair_info.get("msp_slice_idx", -1)
                    if msp_idx >= 0:
                        train_msp_refs.append({
                            "image_path": pair_info["image"],
                            "label_path": pair_info["label"],
                            "slice_idx": msp_idx,
                            "case_id": pair_info["id"],
                            "patient_id": pid,
                            "is_msp": True
                        })

        val_msp_refs = []
        for pid in val_patient_ids:
            for pair_info in patient_groups.get(pid, []):
                if pair_info.get("label") and pair_info.get("has_msp", False):
                    msp_idx = pair_info.get("msp_slice_idx", -1)
                    if msp_idx >= 0:
                        val_msp_refs.append({
                            "image_path": pair_info["image"],
                            "label_path": pair_info["label"],
                            "slice_idx": msp_idx,
                            "case_id": pair_info["id"],
                            "patient_id": pid,
                            "is_msp": True
                        })

        # Generate negative samples
        neg_train_slices = generate_negative_samples(
            train_msp_refs, patient_groups, config,
            target_patient_ids=train_patient_ids,
            log_file=log_file
        )
        neg_val_slices = generate_negative_samples(
            val_msp_refs, patient_groups, config,
            target_patient_ids=val_patient_ids,
            log_file=log_file
        )

        # Combine positive and negative samples
        train_all_slices = train_msp_refs + neg_train_slices
        val_all_slices = val_msp_refs + neg_val_slices

        # Shuffle
        seed = int(config.get("SPLIT_SEED", 42))
        rng = np.random.RandomState(seed + fold_num)
        rng.shuffle(train_all_slices)
        rng.shuffle(val_all_slices)

        if not train_all_slices:
            if log_file:
                log_message(f"Fold {fold_num} has no training data, skipping!", log_file)
            continue

        if log_file:
            log_message(f"Train Slices: {len(train_msp_refs)} MSP + {len(neg_train_slices)} non-MSP = {len(train_all_slices)}", log_file)
            log_message(f"Valid Slices: {len(val_msp_refs)} MSP + {len(neg_val_slices)} non-MSP = {len(val_all_slices)}", log_file)

        # Create datasets
        train_dataset = HeatmapDataset(train_all_slices, config, is_train=True)
        val_dataset = HeatmapDataset(val_all_slices, config, is_train=False)

        # Create dataloaders
        train_loader = create_balanced_dataloader(train_dataset, config, is_train=True)
        val_loader = create_balanced_dataloader(val_dataset, config, is_train=False)

        if log_file:
            log_message(f"Training batches: {len(train_loader)}", log_file)
            log_message(f"Validation batches: {len(val_loader)}", log_file)

        # Initialize model
        model = UNetWithDualHeads(
            in_channels=config["IN_CHANNELS"],
            feat_channels=64
        ).to(device)

        # Stage 1: Heatmap regression training
        stage1_model_path = train_stage1_heatmap(
            model, train_loader, val_loader, config, paths, fold_id
        )

        # Stage 2: Joint training with coverage
        stage2_model_path = train_stage2_joint(
            model, train_loader, val_loader, config, paths, fold_id, stage1_model_path
        )

        fold_results.append({
            'fold': fold_num,
            'stage1_model': str(stage1_model_path),
            'stage2_model': str(stage2_model_path),
            'train_size': len(train_all_slices),
            'val_size': len(val_all_slices),
            'train_msp': len(train_msp_refs),
            'val_msp': len(val_msp_refs)
        })

    # Summary
    if log_file:
        log_message("\n" + "="*80, log_file)
        log_message("5-FOLD CROSS-VALIDATION SUMMARY", log_file)
        log_message("="*80, log_file)

        for result in fold_results:
            log_message(
                f"Fold {result['fold']}: "
                f"Train={result['train_size']} ({result['train_msp']} MSP), "
                f"Val={result['val_size']} ({result['val_msp']} MSP)",
                log_file
            )
            log_message(f"  Stage 1 model: {Path(result['stage1_model']).name}", log_file)
            log_message(f"  Stage 2 model: {Path(result['stage2_model']).name}", log_file)

        log_message(f"\nAll models saved in: {paths['checkpoint_dir']}", log_file)
        log_message("Training complete!", log_file)

    return fold_results


def main():
    """Main execution function."""

    print("="*80)
    print("🔥 COMPLETE MSP DETECTION TRAINING PIPELINE")
    print("   - 5-Fold Patient-Grouped Cross-Validation")
    print("   - 2-Stage Training (Heatmap → Joint with Coverage)")
    print("   - UNetWithDualHeads Architecture")
    print("="*80)

    # Load configuration
    config = get_default_config()

    # ===== UPDATE THESE PATHS =====
    config["IMAGE_DIR"] = "/path/to/your/images"  # ← CHANGE THIS
    config["LABEL_DIR"] = "/path/to/your/labels"  # ← CHANGE THIS
    # ==============================

    # Training parameters
    config["NUM_EPOCHS"] = 400  # Total epochs (will be split between stages)
    config["STAGE1_EPOCHS"] = 200  # Stage 1: Heatmap regression
    config["STAGE2_EPOCHS"] = 200  # Stage 2: Joint training
    config["BATCH_SIZE"] = 8
    config["LEARNING_RATE"] = 2e-4
    config["WEIGHT_DECAY"] = 1e-5
    config["EARLY_STOPPING_PATIENCE"] = 50
    config["KFOLD_SPLITS"] = 5
    config["SAMPLES_PER_CASE"] = 3  # Negative samples per case

    # Loss weights
    config["LAMBDA_REG_S2"] = 0.3  # Stage 2 regression weight
    config["LAMBDA_CLS_S2"] = 1.0  # Stage 2 classification weight
    config["LAMBDA_COV"] = 0.3  # Coverage loss weight
    config["CLS_POS_WEIGHT"] = 2.0  # Positive class weight for imbalanced data

    # Focal loss parameters
    config["FOCAL_GAMMA_REG"] = 2.0
    config["FOCAL_ALPHA_REG"] = 0.25

    print(f"🔧 Configuration loaded. Device: {config['DEVICE']}")
    print(f"📁 Image directory: {config['IMAGE_DIR']}")
    print(f"📁 Label directory: {config['LABEL_DIR']}")
    print(f"🎯 Training: {config['STAGE1_EPOCHS']} + {config['STAGE2_EPOCHS']} epochs")
    print("="*80)

    # Setup output directories
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = Path("results") / f"run_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)

    log_dir = run_dir / "logs"
    log_dir.mkdir(exist_ok=True)

    log_file = setup_logging(log_dir / "training.log")

    paths = {
        "run_dir": run_dir,
        "checkpoint_dir": checkpoint_dir,
        "log_file": log_file
    }

    config["log_file"] = log_file

    log_message("="*80, log_file)
    log_message("COMPLETE MSP DETECTION TRAINING PIPELINE", log_file)
    log_message("="*80, log_file)
    log_message(f"Run directory: {run_dir}", log_file)
    log_message(f"Device: {config['DEVICE']}", log_file)
    log_message(f"Stage 1 epochs: {config['STAGE1_EPOCHS']}", log_file)
    log_message(f"Stage 2 epochs: {config['STAGE2_EPOCHS']}", log_file)

    try:
        # Run complete pipeline
        fold_results = run_5fold_cv_with_staged_training(config, paths)

        print("\n" + "="*80)
        print("✅ Training completed successfully!")
        print(f"📊 Trained {len(fold_results)} folds")
        print(f"💾 Results saved in: {run_dir}")
        print("="*80)

        # Print model paths
        print("\n📦 Model Checkpoints:")
        for result in fold_results:
            print(f"\n  Fold {result['fold']}:")
            print(f"    Stage 1: {Path(result['stage1_model']).name}")
            print(f"    Stage 2: {Path(result['stage2_model']).name}")

    except Exception as e:
        error_msg = f"Error during training: {str(e)}"
        print(f"\n❌ {error_msg}")
        if log_file:
            log_message(f"ERROR: {error_msg}", log_file)
        raise


if __name__ == "__main__":
    main()
