"""
2-Stage Training Pipeline for MSP Detection

Stage 1: Heatmap regression training (freeze classification heads)
Stage 2: Joint training (fine-tune entire model with coverage awareness)
"""

import torch
import torch.optim as optim
from tqdm import tqdm
from pathlib import Path

from losses import compute_coverage_aware_combined_loss
from utils.logging_utils import log_message


def generate_negative_samples(
        positive_msp_refs,
        all_patient_groups,
        config,
        target_patient_ids=None,
        exclude_patient_ids=None,
        samples_per_case_override=None,
        log_file=None):
    """
    Generate negative (non-MSP) samples from volumes with MSP labels.

    For each volume with an MSP slice:
    - Sample non-MSP slices from the same volume
    - Apply quality filtering (foreground ratio check)
    - Return hard negative labels (is_msp=False)

    Args:
        positive_msp_refs: List of positive MSP slice references
        all_patient_groups: Dictionary mapping patient_id to case info
        config: Configuration dictionary
        target_patient_ids: Optional list of patient IDs to process
        exclude_patient_ids: Optional list of patient IDs to exclude
        samples_per_case_override: Override for samples_per_case config
        log_file: Log file path

    Returns:
        List of negative slice references
    """
    import numpy as np
    from data import preprocess_and_cache, load_nifti_data_cached, extract_slice, normalize_slice
    from utils.msp_utils import get_msp_index

    negative_refs = []
    sagittal_axis = config["SAGITTAL_AXIS"]
    structure_labels = config["STRUCTURE_LABELS"]
    samples_to_gen = samples_per_case_override if samples_per_case_override is not None else config.get("SAMPLES_PER_CASE", 3)

    if exclude_patient_ids is None:
        exclude_patient_ids = []

    patient_ids_to_process = []
    if target_patient_ids:
        patient_ids_to_process = [pid for pid in target_patient_ids if pid not in exclude_patient_ids]
    else:
        patient_ids_to_process = [pid for pid in all_patient_groups.keys() if pid not in exclude_patient_ids]

    if log_file:
        log_message(f"Generating negative samples for {len(patient_ids_to_process)} patients.", log_file)

    for patient_id in tqdm(patient_ids_to_process, desc="Generating negative samples"):
        if patient_id not in all_patient_groups:
            continue

        for case_info in all_patient_groups[patient_id]:
            img_vol, _ = preprocess_and_cache(str(case_info["image"]), config["CACHE_DIR"], config, log_file)
            if img_vol is None:
                continue

            label_vol = load_nifti_data_cached(str(case_info["label"]), is_label=True)
            if label_vol is None:
                continue

            num_slices_in_volume = label_vol.shape[sagittal_axis]
            if num_slices_in_volume <= 1:
                continue

            # Get the ground truth MSP index
            true_msp_idx_for_volume = get_msp_index(label_vol, sagittal_axis, structure_labels)
            if true_msp_idx_for_volume < 0:
                continue

            # Simplified sampling strategy: distinguish MSP vs non-MSP slices only
            sampled_slices = []

            for slice_idx in range(num_slices_in_volume):
                # Skip true MSP slice
                if slice_idx == true_msp_idx_for_volume:
                    continue

                # Quality check
                img_slice = extract_slice(img_vol, slice_idx, sagittal_axis)
                if img_slice is None:
                    continue

                img_slice_norm = normalize_slice(img_slice, config)
                foreground_ratio = (img_slice_norm > 0.01).mean()

                if foreground_ratio < 0.05:
                    continue

                sampled_slices.append(slice_idx)

            # Randomly sample specified number of negative samples
            if sampled_slices:
                n_to_sample = min(samples_to_gen, len(sampled_slices))
                sampled_indices = np.random.choice(sampled_slices, n_to_sample, replace=False)

                for slice_idx in sampled_indices:
                    negative_refs.append({
                        "image_path": case_info["image"],
                        "label_path": case_info["label"],
                        "slice_idx": slice_idx,
                        "case_id": case_info["id"],
                        "patient_id": patient_id,
                        "is_msp": False,  # Hard label: False
                        "true_msp_idx": true_msp_idx_for_volume
                    })

    if log_file:
        log_message(f"Generated {len(negative_refs)} negative samples (all hard label=0)", log_file)

    return negative_refs


def train_stage1_heatmap(model, train_loader, val_loader, config, paths, fold_id="fold_1"):
    """
    Stage 1: Train heatmap regression only.

    Freezes classification and coverage heads, only trains the UNet backbone.

    Args:
        model: UNetWithDualHeads model
        train_loader: Training data loader
        val_loader: Validation data loader
        config: Configuration dictionary
        paths: Dictionary with checkpoint_dir and log_file paths
        fold_id: Fold identifier string

    Returns:
        Path to best Stage 1 model checkpoint
    """
    log_file = paths["log_file"]
    device = torch.device(config["DEVICE"])

    if log_file:
        log_message(f"=== Stage 1: Heatmap Regression Training ({fold_id}) ===", log_file)

    # Freeze classification and coverage heads
    for param in model.cls_head.parameters():
        param.requires_grad = False
    for param in model.coverage_head.parameters():
        param.requires_grad = False

    # Only train UNet backbone
    optimizer = optim.AdamW(
        model.unet.parameters(),
        lr=config["LEARNING_RATE"],
        weight_decay=config.get("WEIGHT_DECAY", 1e-5)
    )

    stage1_epochs = config.get("STAGE1_EPOCHS", 200)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=stage1_epochs)

    best_val_loss = float('inf')
    patience_counter = 0
    patience = config.get("EARLY_STOPPING_PATIENCE", 50)

    stage1_model_path = paths["checkpoint_dir"] / f"{fold_id}_best_stage1_regression_model.pth"

    for epoch in range(stage1_epochs):
        model.train()
        train_losses = []

        pbar = tqdm(train_loader, desc=f"S1-Epoch {epoch+1}/{stage1_epochs}")
        for batch in pbar:
            images = batch["image"].to(device)
            target_heatmaps = batch["target_heatmap"].to(device)
            brain_masks = batch["brain_mask"].to(device)
            has_regression_targets = batch["has_regression_target"].to(device)
            soft_msp_labels = batch["is_msp_label"].to(device)
            cov_labels = batch["cov_label"].to(device)

            optimizer.zero_grad()

            pred_heatmaps_logits, cls_logit, cov_logits = model(images)

            # Stage 1: Only regression loss
            total_loss, _ = compute_coverage_aware_combined_loss(
                pred_heatmaps_logits, cls_logit, cov_logits,
                target_heatmaps, soft_msp_labels, cov_labels, has_regression_targets,
                brain_masks,
                lambda_reg=1.0,  # Only regression
                lambda_cls=0.0,  # No classification
                lambda_cov=0.0,  # No coverage
                config=config,
                current_epoch=epoch,
                total_epochs=stage1_epochs
            )

            total_loss.backward()
            optimizer.step()

            train_losses.append(total_loss.item())
            pbar.set_postfix(loss=f"{total_loss.item():.4f}")

        scheduler.step()

        # Validation
        model.eval()
        val_losses = []

        with torch.no_grad():
            for batch in val_loader:
                images = batch["image"].to(device)
                target_heatmaps = batch["target_heatmap"].to(device)
                brain_masks = batch["brain_mask"].to(device)
                has_regression_targets = batch["has_regression_target"].to(device)
                soft_msp_labels = batch["is_msp_label"].to(device)
                cov_labels = batch["cov_label"].to(device)

                pred_heatmaps_logits, cls_logit, cov_logits = model(images)

                val_loss, _ = compute_coverage_aware_combined_loss(
                    pred_heatmaps_logits, cls_logit, cov_logits,
                    target_heatmaps, soft_msp_labels, cov_labels, has_regression_targets,
                    brain_masks,
                    lambda_reg=1.0,
                    lambda_cls=0.0,
                    lambda_cov=0.0,
                    config=config,
                    current_epoch=epoch,
                    total_epochs=stage1_epochs
                )

                val_losses.append(val_loss.item())

        avg_train_loss = sum(train_losses) / len(train_losses)
        avg_val_loss = sum(val_losses) / len(val_losses)

        if log_file:
            log_message(f"S1 Epoch {epoch+1}/{stage1_epochs}: Train={avg_train_loss:.4f}, Val={avg_val_loss:.4f}",
                       log_file)

        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0

            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'model_type': 'UNetWithDualHeads_Stage1',
                'config': config
            }, stage1_model_path)

            if log_file:
                log_message(f"  -> Stage 1 best model saved (val_loss: {avg_val_loss:.4f})", log_file)
        else:
            patience_counter += 1

        if patience_counter >= patience:
            if log_file:
                log_message(f"Early stopping at epoch {epoch+1}", log_file)
            break

    # Unfreeze all parameters for Stage 2
    for param in model.cls_head.parameters():
        param.requires_grad = True
    for param in model.coverage_head.parameters():
        param.requires_grad = True

    return stage1_model_path


def train_stage2_joint(model, train_loader, val_loader, config, paths, fold_id="fold_1", stage1_model_path=None):
    """
    Stage 2: Joint training with coverage awareness.

    Trains the entire model (UNet + classification + coverage heads) jointly.

    Args:
        model: UNetWithDualHeads model
        train_loader: Training data loader
        val_loader: Validation data loader
        config: Configuration dictionary
        paths: Dictionary with checkpoint_dir and log_file paths
        fold_id: Fold identifier string
        stage1_model_path: Path to Stage 1 checkpoint (optional, for loading)

    Returns:
        Path to best Stage 2 model checkpoint
    """
    log_file = paths["log_file"]
    device = torch.device(config["DEVICE"])

    if log_file:
        log_message(f"=== Stage 2: Joint Training ({fold_id}) ===", log_file)

    # Load Stage 1 weights if available
    if stage1_model_path and Path(stage1_model_path).exists():
        try:
            checkpoint = torch.load(stage1_model_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            if log_file:
                log_message("Successfully loaded Stage 1 weights", log_file)
        except Exception as e:
            if log_file:
                log_message(f"Warning: Could not load Stage 1 model weights. Error: {e}", log_file)

    # Stage 2: Lower learning rate for UNet, normal for heads
    optimizer = optim.AdamW([
        {'params': model.unet.parameters(), 'lr': config["LEARNING_RATE"] * 0.1},
        {'params': list(model.cls_head.parameters()) + list(model.coverage_head.parameters()),
         'lr': config["LEARNING_RATE"]}
    ], weight_decay=config.get("WEIGHT_DECAY", 1e-5))

    stage2_epochs = config.get("STAGE2_EPOCHS", 200)
    stage1_epochs = config.get("STAGE1_EPOCHS", 200)
    total_epochs = stage1_epochs + stage2_epochs

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=stage2_epochs)

    best_val_loss = float('inf')
    patience_counter = 0
    patience = config.get("EARLY_STOPPING_PATIENCE", 50)

    stage2_model_path = paths["checkpoint_dir"] / f"{fold_id}_best_coverage_aware_heatmap_model.pth"

    # Calculate positive class weight for balanced training
    # This helps with imbalanced MSP/non-MSP ratios
    pos_weight = config.get("CLS_POS_WEIGHT", 2.0)

    for epoch in range(stage2_epochs):
        model.train()
        train_losses = []

        pbar = tqdm(train_loader, desc=f"S2-Epoch {epoch+1}/{stage2_epochs}")
        for batch in pbar:
            images = batch["image"].to(device)
            target_heatmaps = batch["target_heatmap"].to(device)
            brain_masks = batch["brain_mask"].to(device)
            has_regression_targets = batch["has_regression_target"].to(device)
            soft_msp_labels = batch["is_msp_label"].to(device)
            cov_labels = batch["cov_label"].to(device)

            optimizer.zero_grad()

            pred_heatmaps_logits, cls_logit, cov_logits = model(images)

            # Stage 2: Full loss with all components
            total_loss, _ = compute_coverage_aware_combined_loss(
                pred_heatmaps_logits, cls_logit, cov_logits,
                target_heatmaps, soft_msp_labels, cov_labels, has_regression_targets,
                brain_masks,
                lambda_reg=config.get("LAMBDA_REG_S2", 0.3),
                lambda_cls=config.get("LAMBDA_CLS_S2", 1.0),
                lambda_cov=config.get("LAMBDA_COV", 0.3),
                cls_pos_weight=pos_weight,
                config=config,
                current_epoch=stage1_epochs + epoch,
                total_epochs=total_epochs
            )

            total_loss.backward()
            optimizer.step()

            train_losses.append(total_loss.item())
            pbar.set_postfix(loss=f"{total_loss.item():.4f}")

        scheduler.step()

        # Validation
        model.eval()
        val_losses = []

        with torch.no_grad():
            for batch in val_loader:
                images = batch["image"].to(device)
                target_heatmaps = batch["target_heatmap"].to(device)
                brain_masks = batch["brain_mask"].to(device)
                has_regression_targets = batch["has_regression_target"].to(device)
                soft_msp_labels = batch["is_msp_label"].to(device)
                cov_labels = batch["cov_label"].to(device)

                pred_heatmaps_logits, cls_logit, cov_logits = model(images)

                val_loss, _ = compute_coverage_aware_combined_loss(
                    pred_heatmaps_logits, cls_logit, cov_logits,
                    target_heatmaps, soft_msp_labels, cov_labels, has_regression_targets,
                    brain_masks,
                    lambda_reg=config.get("LAMBDA_REG_S2", 0.3),
                    lambda_cls=config.get("LAMBDA_CLS_S2", 1.0),
                    lambda_cov=config.get("LAMBDA_COV", 0.3),
                    cls_pos_weight=pos_weight,
                    config=config,
                    current_epoch=stage1_epochs + epoch,
                    total_epochs=total_epochs
                )

                val_losses.append(val_loss.item())

        avg_train_loss = sum(train_losses) / len(train_losses)
        avg_val_loss = sum(val_losses) / len(val_losses)

        if log_file:
            log_message(f"S2 Epoch {epoch+1}/{stage2_epochs}: Train={avg_train_loss:.4f}, Val={avg_val_loss:.4f}",
                       log_file)

        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0

            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'model_type': 'UNetWithDualHeads',
                'config': config
            }, stage2_model_path)

            if log_file:
                log_message(f"  -> Stage 2 best model saved (val_loss: {avg_val_loss:.4f})", log_file)
        else:
            patience_counter += 1

        if patience_counter >= patience:
            if log_file:
                log_message(f"Early stopping at epoch {epoch+1}", log_file)
            break

    return stage2_model_path


def train_heatmap_model_with_coverage_aware_training(train_refs, val_refs, config, paths):
    """
    Complete 2-stage training wrapper with coverage-aware model.

    This is the main training entry point that:
    1. Creates datasets and dataloaders
    2. Initializes UNetWithDualHeads model
    3. Runs Stage 1: Regression training (frozen classification heads)
    4. Runs Stage 2: Joint training (fine-tune all heads)
    5. Returns path to the best Stage 2 model

    Args:
        train_refs: List of training slice references
        val_refs: List of validation slice references
        config: Configuration dictionary
        paths: Dictionary with checkpoint_dir and log_file

    Returns:
        Path: Path to the best Stage 2 model checkpoint
    """
    from pathlib import Path
    import torch
    from models import UNetWithDualHeads
    from data import HeatmapDataset, create_balanced_dataloader
    from utils.logging_utils import log_message

    log_file = paths["log_file"]
    device = torch.device(config["DEVICE"])

    # Epoch configuration
    stage1_epochs = config.get("STAGE1_EPOCHS", 200)
    stage2_epochs = config.get("STAGE2_EPOCHS", 200)

    if log_file:
        log_message(f"=== Complete 2-Stage Training Pipeline ===", log_file)
        log_message(f"  Stage 1 epochs: {stage1_epochs}", log_file)
        log_message(f"  Stage 2 epochs: {stage2_epochs}", log_file)

    # Create datasets
    train_dataset = HeatmapDataset(train_refs, config, is_train=True, split_name="train")
    val_dataset = HeatmapDataset(val_refs, config, is_train=False, split_name="val")

    train_loader = create_balanced_dataloader(train_dataset, config, is_train=True)
    val_loader = create_balanced_dataloader(val_dataset, config, is_train=False)

    # Calculate class weights for imbalance handling
    train_labels = [ref.get("is_msp", False) for ref in train_refs]
    pos_count = sum(train_labels)
    neg_count = len(train_labels) - pos_count

    if pos_count > 0 and neg_count > 0 and log_file:
        pos_weight = neg_count / pos_count
        log_message(f"  Class imbalance ratio: {pos_weight:.2f} (neg/pos)", log_file)

    # Check data
    try:
        first_batch = next(iter(train_loader))
        img_sample = first_batch["image"]
        if log_file:
            log_message(f"  Input image tensor range [{img_sample.min():.2f}, {img_sample.max():.2f}]", log_file)
    except StopIteration:
        if log_file:
            log_message("ERROR: Training loader is empty. Cannot start training.", log_file)
        raise ValueError("Training loader is empty.")

    # Initialize model
    model = UNetWithDualHeads(in_channels=config["IN_CHANNELS"], feat_channels=64).to(device)

    stage1_model_path = paths["checkpoint_dir"] / "best_stage1_regression_model.pth"
    stage2_model_path = paths["checkpoint_dir"] / "best_coverage_aware_heatmap_model.pth"

    # Stage 1: Regression Training
    if stage1_epochs > 0:
        if log_file:
            log_message(f"\n=== Stage 1: Regression Training ({stage1_epochs} epochs) ===", log_file)
        
        stage1_result = train_stage1_heatmap(
            model, train_loader, val_loader, config, paths, fold_id="stage1"
        )
        
        if log_file:
            log_message(f"  Stage 1 complete. Best model saved to: {stage1_model_path}", log_file)
    else:
        if log_file:
            log_message("  Stage 1 skipped (STAGE1_EPOCHS=0)", log_file)

    # Stage 2: Joint Training
    if stage2_epochs > 0:
        if log_file:
            log_message(f"\n=== Stage 2: Joint Training ({stage2_epochs} epochs) ===", log_file)
        
        stage2_result = train_stage2_joint(
            model, train_loader, val_loader, config, paths, 
            fold_id="stage2", stage1_model_path=stage1_model_path
        )
        
        if log_file:
            log_message(f"  Stage 2 complete. Best model saved to: {stage2_model_path}", log_file)
    else:
        if log_file:
            log_message("  Stage 2 skipped (STAGE2_EPOCHS=0)", log_file)
        # If stage 2 is skipped, return stage 1 model
        return stage1_model_path

    if log_file:
        log_message(f"\n=== 2-Stage Training Complete ===", log_file)
        log_message(f"  Final model: {stage2_model_path}", log_file)

    return stage2_model_path
