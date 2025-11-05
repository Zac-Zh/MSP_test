# MSP Detection - Usage Examples

This directory contains example scripts demonstrating how to use the MSP detection codebase.

## 📚 Available Examples

### 1. **example_data_pipeline.py**
Complete data loading and preprocessing pipeline.

**Demonstrates:**
- Loading NIfTI files with caching
- Extracting and normalizing slices
- Creating datasets with augmentation
- Creating balanced dataloaders
- Batch structure and contents

**Run:**
```bash
python examples/example_data_pipeline.py
```

### 2. **example_model_training.py**
Model training workflow and structure.

**Demonstrates:**
- Model instantiation (UNetWithCls)
- Loss function setup (DiceLoss)
- Optimizer configuration
- Training loop structure
- Model checkpoint saving

**Run:**
```bash
python examples/example_model_training.py
```

### 3. **example_inference.py**
Inference with test-time augmentation.

**Demonstrates:**
- Loading trained models
- Image preprocessing for inference
- Test-time augmentation (horizontal flip)
- Feature extraction for meta-classifier
- Post-processing predictions

**Run:**
```bash
python examples/example_inference.py
```

## 🚀 Quick Start

### Complete Workflow Example

```python
from config import get_default_config
from data import HeatmapDataset, create_balanced_dataloader
from models import UNetWithCls
from losses import DiceLoss
from train import prepare_patient_grouped_datasets
import torch.optim as optim

# 1. Configuration
config = get_default_config()
device = torch.device(config["DEVICE"])

# 2. Prepare datasets
train_refs, val_refs, patient_groups = prepare_patient_grouped_datasets(
    config,
    log_file=None
)

# 3. Create dataloaders
train_dataset = HeatmapDataset(train_refs, config, is_train=True)
train_loader = create_balanced_dataloader(train_dataset, config, is_train=True)

# 4. Create model
model = UNetWithCls(
    n_channels=1,
    n_classes=4,
    bilinear_unet=True
).to(device)

# 5. Setup training
criterion = DiceLoss(smooth=1.0)
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# 6. Train
for epoch in range(num_epochs):
    model.train()
    for batch in train_loader:
        images = batch['image'].to(device)
        targets = batch['target_heatmap'].to(device)

        # Forward
        heatmaps, cls_logits = model(images)
        loss = criterion(torch.sigmoid(heatmaps), targets)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Save checkpoint
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'model_type': 'UNetWithCls'
    }, f'checkpoint_epoch_{epoch}.pth')
```

## 📖 Module Usage Guide

### Data Module

```python
from data import (
    load_nifti_data_cached,
    extract_slice,
    normalize_slice,
    HeatmapDataset,
    create_balanced_dataloader
)

# Load NIfTI file
img_vol = load_nifti_data_cached("path/to/image.nii.gz")

# Extract slice
img_slice = extract_slice(img_vol, slice_idx=100, axis=2)

# Normalize
img_norm = normalize_slice(img_slice, config)

# Create dataset
dataset = HeatmapDataset(data_refs, config, is_train=True)

# Create dataloader
loader = create_balanced_dataloader(dataset, config, is_train=True)
```

### Models Module

```python
from models import UNetHeatmap, UNetWithCls, UNetWithDualHeads

# Heatmap-only model
model1 = UNetHeatmap(n_channels=1, n_classes=4)

# With classification head
model2 = UNetWithCls(n_channels=1, n_classes=4)

# With dual heads (classification + coverage)
model3 = UNetWithDualHeads(in_channels=1, feat_channels=64)
```

### Features Module

```python
from features import extract_heatmap_features

# Extract features from heatmap predictions
features = extract_heatmap_features(
    heatmap_logits,  # [4, H, W]
    brain_mask=brain_mask,  # [H, W]
    config=config
)
# Returns: 58-dimensional feature vector
```

### Inference Module

```python
from inference import apply_tta_horizontal_flip

# Apply test-time augmentation
with torch.no_grad():
    outputs = apply_tta_horizontal_flip(images, model)
# Automatically averages original and horizontally flipped predictions
```

### Evaluation Module

```python
from eval import (
    scan_slice_threshold_youden,
    find_optimal_case_threshold,
    evaluate_case_level
)

# Optimize slice-level threshold
best_thresh, results_df, best_idx = scan_slice_threshold_youden(
    y_true=labels,
    y_score=predictions
)

# Optimize case-level threshold
optimal = find_optimal_case_threshold(
    case_probs=probabilities,
    true_case_labels=labels,
    sens_min=0.7
)

# Aggregate to case-level
case_result = evaluate_case_level(
    volume_slice_probs=slice_probs,
    case_threshold=0.5
)
```

## 🔧 Configuration

Modify configuration in `config/config.py` or override specific values:

```python
from config import get_default_config

config = get_default_config()
config["BATCH_SIZE"] = 16
config["LEARNING_RATE"] = 5e-5
config["NUM_EPOCHS"] = 100
```

## 📝 Notes

- These examples use placeholder paths - replace with your actual data paths
- Some examples are commented out to run without actual data files
- For complete pipelines, see the original `main.py` (being refactored)
- All examples preserve exact functionality from the original codebase

## 🆘 Need Help?

- See module docstrings for detailed API documentation
- Check `REFACTORING_COMPLETE_SUMMARY.md` for comprehensive overview
- Review function signatures and type hints in source files
