"""
Example: Model training workflow.

Demonstrates how to train a UNet model for MSP detection.
"""

import torch
import torch.optim as optim
from config import get_default_config
from models import UNetWithCls
from losses import DiceLoss
from data import HeatmapDataset, create_balanced_dataloader


def train_one_epoch_example(model, train_loader, criterion, optimizer, device):
    """Example training loop for one epoch."""
    model.train()
    total_loss = 0.0
    num_batches = 0

    for batch_idx, batch in enumerate(train_loader):
        # Move data to device
        images = batch['image'].to(device)
        target_heatmaps = batch['target_heatmap'].to(device)

        # Forward pass
        heatmap_logits, cls_logits = model(images)

        # Compute loss (heatmap only for this example)
        loss = criterion(torch.sigmoid(heatmap_logits), target_heatmaps)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Track metrics
        total_loss += loss.item()
        num_batches += 1

        if batch_idx % 10 == 0:
            print(f"    Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}")

    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    return avg_loss


def main():
    print("=" * 70)
    print("MSP Detection - Model Training Example")
    print("=" * 70)

    # Configuration
    config = get_default_config()
    device = torch.device(config["DEVICE"])
    print(f"\n✅ Using device: {device}")

    # Create model
    print("\n1. Creating model...")
    model = UNetWithCls(
        n_channels=config["IN_CHANNELS"],
        n_classes=len(config["STRUCTURE_LABELS"]),
        bilinear_unet=config.get("BILINEAR_UNET", True)
    ).to(device)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   Model: UNetWithCls")
    print(f"   Trainable parameters: {num_params:,}")

    # Create loss function
    print("\n2. Creating loss function...")
    criterion = DiceLoss(smooth=1.0)
    print("   Loss: DiceLoss")

    # Create optimizer
    print("\n3. Creating optimizer...")
    optimizer = optim.Adam(
        model.parameters(),
        lr=config["LEARNING_RATE"],
        weight_decay=config.get("WEIGHT_DECAY", 0.0)
    )
    print(f"   Optimizer: Adam")
    print(f"   Learning rate: {config['LEARNING_RATE']}")

    # Create data (placeholder - replace with actual data)
    print("\n4. Creating data loaders...")
    # In practice, you would:
    # - Use find_nifti_pairs() to get file pairs
    # - Create slice references with prepare_patient_grouped_datasets()
    # - Create HeatmapDataset instances
    # - Create DataLoaders with create_balanced_dataloader()

    # Example (commented out - requires actual data):
    """
    from train import prepare_patient_grouped_datasets

    train_refs, val_refs, patient_groups = prepare_patient_grouped_datasets(
        config,
        log_file=None
    )

    train_dataset = HeatmapDataset(train_refs, config, is_train=True)
    val_dataset = HeatmapDataset(val_refs, config, is_train=False)

    train_loader = create_balanced_dataloader(train_dataset, config, is_train=True)
    val_loader = create_balanced_dataloader(val_dataset, config, is_train=False)
    """

    print("   (Skipping data creation - requires actual NIfTI files)")

    # Training loop (example structure)
    print("\n5. Training loop structure:")
    print("   for epoch in range(num_epochs):")
    print("       # Train one epoch")
    print("       train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)")
    print("       ")
    print("       # Validate")
    print("       val_loss = validate_one_epoch(model, val_loader, criterion, device)")
    print("       ")
    print("       # Save best model")
    print("       if val_loss < best_val_loss:")
    print("           torch.save({")
    print("               'epoch': epoch,")
    print("               'model_state_dict': model.state_dict(),")
    print("               'optimizer_state_dict': optimizer.state_dict(),")
    print("               'val_loss': val_loss,")
    print("               'model_type': 'UNetWithCls'")
    print("           }, 'best_model.pth')")

    print("\n" + "=" * 70)
    print("✅ Model training example complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
