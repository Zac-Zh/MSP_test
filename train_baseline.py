"""
Baseline Training Script for MSP Detection

This script trains a UNetWithCls model from scratch on your MSP detection dataset.
It provides a simple, complete training workflow that users can run directly.

Usage:
    python train_baseline.py
"""

import torch
import torch.optim as optim
import numpy as np
from pathlib import Path
from tqdm import tqdm

from config import get_default_config
from utils.logging_utils import log_message, setup_logging
from train import prepare_patient_grouped_datasets
from data import HeatmapDataset, create_balanced_dataloader
from models import UNetWithCls
from losses import DiceLoss, FocalLoss


def train_one_epoch(model, train_loader, optimizer, criterion, device, epoch, total_epochs):
    """Train for one epoch."""
    model.train()
    train_losses = []

    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{total_epochs}")
    for batch in pbar:
        images = batch["image"].to(device)
        target_heatmaps = batch["target_heatmap"].to(device)
        is_msp_labels = batch["is_msp_label"].to(device)

        # Forward pass
        heatmap_logits, cls_logits = model(images)

        # Compute losses
        heatmap_loss = criterion(torch.sigmoid(heatmap_logits), target_heatmaps)
        cls_loss = torch.nn.functional.binary_cross_entropy_with_logits(
            cls_logits.squeeze(),
            is_msp_labels.squeeze().float()
        )

        # Combined loss
        total_loss = heatmap_loss + 0.5 * cls_loss

        # Backward pass
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        train_losses.append(total_loss.item())
        pbar.set_postfix(loss=f"{total_loss.item():.4f}")

    return np.mean(train_losses)


def validate_one_epoch(model, val_loader, criterion, device):
    """Validate for one epoch."""
    model.eval()
    val_losses = []

    with torch.no_grad():
        for batch in val_loader:
            images = batch["image"].to(device)
            target_heatmaps = batch["target_heatmap"].to(device)
            is_msp_labels = batch["is_msp_label"].to(device)

            # Forward pass
            heatmap_logits, cls_logits = model(images)

            # Compute losses
            heatmap_loss = criterion(torch.sigmoid(heatmap_logits), target_heatmaps)
            cls_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                cls_logits.squeeze(),
                is_msp_labels.squeeze().float()
            )

            total_loss = heatmap_loss + 0.5 * cls_loss
            val_losses.append(total_loss.item())

    return np.mean(val_losses) if val_losses else float('inf')


def main():
    """Main training function."""

    # ===== CONFIGURATION =====
    config = get_default_config()

    # UPDATE THESE PATHS TO YOUR DATA
    config["IMAGE_DIR"] = "/path/to/your/images"  # TODO: Update this
    config["LABEL_DIR"] = "/path/to/your/labels"  # TODO: Update this

    # Training hyperparameters
    config["NUM_EPOCHS"] = 100
    config["LEARNING_RATE"] = 1e-4
    config["BATCH_SIZE"] = 8
    config["DEVICE"] = "cuda" if torch.cuda.is_available() else "cpu"

    # Create output directories
    checkpoint_dir = Path("./checkpoints")
    log_dir = Path("./logs")
    checkpoint_dir.mkdir(exist_ok=True)
    log_dir.mkdir(exist_ok=True)

    # Setup logging
    log_file = log_dir / "training_baseline.log"
    setup_logging(log_file)

    log_message("=" * 60, log_file)
    log_message("MSP Detection - Baseline Training", log_file)
    log_message("=" * 60, log_file)
    log_message(f"Device: {config['DEVICE']}", log_file)
    log_message(f"Image Directory: {config['IMAGE_DIR']}", log_file)
    log_message(f"Label Directory: {config['LABEL_DIR']}", log_file)
    log_message(f"Epochs: {config['NUM_EPOCHS']}", log_file)
    log_message(f"Learning Rate: {config['LEARNING_RATE']}", log_file)
    log_message(f"Batch Size: {config['BATCH_SIZE']}", log_file)
    log_message("=" * 60, log_file)

    device = torch.device(config["DEVICE"])

    # ===== PREPARE DATA =====
    log_message("Preparing datasets...", log_file)

    train_refs, val_refs, patient_groups = prepare_patient_grouped_datasets(config, log_file)

    if not train_refs:
        raise ValueError("No training data found! Check IMAGE_DIR and LABEL_DIR paths.")

    if not val_refs:
        log_message("WARNING: No validation data found. Using training data for validation.", log_file)
        val_refs = train_refs[:max(1, len(train_refs) // 10)]  # Use 10% for validation

    log_message(f"Training slices: {len(train_refs)}", log_file)
    log_message(f"Validation slices: {len(val_refs)}", log_file)

    # Create datasets
    train_dataset = HeatmapDataset(train_refs, config, is_train=True)
    val_dataset = HeatmapDataset(val_refs, config, is_train=False)

    # Create dataloaders
    train_loader = create_balanced_dataloader(train_dataset, config, is_train=True)
    val_loader = create_balanced_dataloader(val_dataset, config, is_train=False)

    log_message(f"Training batches: {len(train_loader)}", log_file)
    log_message(f"Validation batches: {len(val_loader)}", log_file)

    # ===== CREATE MODEL =====
    log_message("Creating model...", log_file)

    model = UNetWithCls(
        n_channels=config["IN_CHANNELS"],
        n_classes=len(config["STRUCTURE_LABELS"]),
        bilinear_unet=config.get("BILINEAR_UNET", True)
    ).to(device)

    num_params = sum(p.numel() for p in model.parameters())
    log_message(f"Model parameters: {num_params:,}", log_file)

    # ===== TRAINING SETUP =====
    criterion = DiceLoss(smooth=1.0)
    optimizer = optim.Adam(model.parameters(), lr=config["LEARNING_RATE"])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["NUM_EPOCHS"])

    best_val_loss = float('inf')
    patience_counter = 0
    patience = config.get("EARLY_STOPPING_PATIENCE", 20)

    best_model_path = checkpoint_dir / "best_baseline_model.pth"

    # ===== TRAINING LOOP =====
    log_message("Starting training...", log_file)
    log_message("=" * 60, log_file)

    for epoch in range(config["NUM_EPOCHS"]):
        # Train
        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, device, epoch, config["NUM_EPOCHS"]
        )

        # Validate
        val_loss = validate_one_epoch(model, val_loader, criterion, device)

        # Log progress
        log_message(
            f"Epoch {epoch+1}/{config['NUM_EPOCHS']}: "
            f"Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}",
            log_file
        )

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': float(val_loss),
                'train_loss': float(train_loss),
                'config': config,
                'model_type': 'UNetWithCls'
            }, best_model_path)
            log_message(f"  → Saved best model (val_loss={val_loss:.4f})", log_file)
            patience_counter = 0
        else:
            patience_counter += 1

        # Early stopping
        if patience_counter >= patience:
            log_message(f"Early stopping triggered at epoch {epoch+1}", log_file)
            break

        # Learning rate scheduling
        scheduler.step()

    # ===== TRAINING COMPLETE =====
    log_message("=" * 60, log_file)
    log_message("Training completed!", log_file)
    log_message(f"Best validation loss: {best_val_loss:.4f}", log_file)
    log_message(f"Best model saved to: {best_model_path}", log_file)
    log_message("=" * 60, log_file)

    print(f"\n✅ Training complete! Best model saved to: {best_model_path}")
    print(f"✅ Best validation loss: {best_val_loss:.4f}")
    print(f"✅ Log file: {log_file}")


if __name__ == "__main__":
    main()
