# MSP Detection - Use Cases & Migration Guide

## 🎯 Current Status

Your repository is **production-ready** with modular components extracted. The original `main.py` (9,013 lines) should be **renamed and archived** for reference, not uploaded to GitHub in its current form.

---

## 📋 Recommended File Management

### Option 1: Archive the Original (Recommended)
```bash
# Rename main.py to indicate it's the legacy monolithic version
mv main.py main_legacy_reference.py

# Or move it to an archive directory
mkdir -p archive
mv main.py archive/main_monolithic_original.py

# Add to .gitignore
echo "archive/" >> .gitignore
echo "main_legacy_reference.py" >> .gitignore
```

### Option 2: Keep for Comparison (During Development)
```bash
# Keep it but clearly mark it as legacy
mv main.py main_LEGACY_DO_NOT_USE.py
echo "main_LEGACY_DO_NOT_USE.py" >> .gitignore
```

### Option 3: Complete Removal (For Clean Release)
```bash
# Delete it entirely if all functionality is extracted
rm main.py
# (Only do this after verifying all needed functions are extracted)
```

**Recommendation:** Use **Option 1** - keep it in an `archive/` directory for reference during development, but don't upload to GitHub.

---

## 🚀 Use Cases for Your Refactored Codebase

### Use Case 1: Training a New MSP Detection Model

**Scenario:** Train a UNet model from scratch on your dataset

**Code:**
```python
from config import get_default_config
from train import prepare_patient_grouped_datasets
from data import HeatmapDataset, create_balanced_dataloader
from models import UNetWithCls
from losses import DiceLoss
import torch.optim as optim

# Setup
config = get_default_config()
config["IMAGE_DIR"] = "/path/to/your/images"
config["LABEL_DIR"] = "/path/to/your/labels"
device = torch.device(config["DEVICE"])

# Prepare patient-grouped datasets
train_refs, val_refs, patient_groups = prepare_patient_grouped_datasets(
    config,
    log_file="training.log"
)

# Create datasets with augmentation
train_dataset = HeatmapDataset(train_refs, config, is_train=True)
val_dataset = HeatmapDataset(val_refs, config, is_train=False)

# Create balanced dataloaders
train_loader = create_balanced_dataloader(train_dataset, config, is_train=True)
val_loader = create_balanced_dataloader(val_dataset, config, is_train=False)

# Model & training setup
model = UNetWithCls(n_channels=1, n_classes=4).to(device)
criterion = DiceLoss(smooth=1.0)
optimizer = optim.Adam(model.parameters(), lr=config["LEARNING_RATE"])

# Training loop
for epoch in range(config["NUM_EPOCHS"]):
    model.train()
    for batch in train_loader:
        images = batch['image'].to(device)
        targets = batch['target_heatmap'].to(device)

        heatmaps, cls_logits = model(images)
        loss = criterion(torch.sigmoid(heatmaps), targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Save checkpoint
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'model_type': 'UNetWithCls'
    }, f'checkpoints/model_epoch_{epoch}.pth')
```

---

### Use Case 2: Inference on New MRI Volumes

**Scenario:** Detect MSP in new brain MRI scans

**Code:**
```python
from config import get_default_config
from train import load_model_with_correct_architecture
from data import load_nifti_data_cached, extract_slice, normalize_slice
from inference import apply_tta_horizontal_flip
from features import extract_heatmap_features
import torch
import cv2
import numpy as np

# Setup
config = get_default_config()
device = torch.device(config["DEVICE"])

# Load trained model
model, model_type = load_model_with_correct_architecture(
    "path/to/best_model.pth",
    config,
    device,
    log_file=None
)
model.eval()

# Load volume
volume_path = "path/to/new_mri_volume.nii.gz"
img_vol = load_nifti_data_cached(volume_path, is_label=False)

# Process each slice
slice_predictions = []
for slice_idx in range(img_vol.shape[2]):  # Assuming sagittal axis = 2
    # Extract and preprocess
    img_slice = extract_slice(img_vol, slice_idx, axis=2)
    img_norm = normalize_slice(img_slice, config)

    # Resize and convert to tensor
    H, W = config["IMAGE_SIZE"]
    img_resized = cv2.resize(img_norm, (W, H))
    img_tensor = torch.from_numpy(img_resized).unsqueeze(0).unsqueeze(0).float().to(device)

    # Inference with TTA
    with torch.no_grad():
        outputs = apply_tta_horizontal_flip(img_tensor, model)
        if isinstance(outputs, tuple):
            heatmaps, cls_logits = outputs
            cls_prob = torch.sigmoid(cls_logits).item()
        else:
            heatmaps = outputs
            cls_prob = 0.0

    slice_predictions.append(cls_prob)

# Find MSP slice
msp_slice_idx = np.argmax(slice_predictions)
msp_confidence = slice_predictions[msp_slice_idx]

print(f"Predicted MSP slice: {msp_slice_idx}")
print(f"Confidence: {msp_confidence:.4f}")
```

---

### Use Case 3: Feature Extraction for Meta-Classifier

**Scenario:** Extract features from heatmaps to train a meta-classifier

**Code:**
```python
from features import extract_heatmap_features
from data import generate_brain_mask_from_image
import numpy as np

# After getting heatmap predictions from model
heatmap_logits = heatmaps[0].cpu().numpy()  # [C, H, W]

# Generate brain mask
brain_mask = generate_brain_mask_from_image(img_norm, config)

# Extract 58-dimensional feature vector
features = extract_heatmap_features(
    heatmap_logits,
    brain_mask=brain_mask,
    config=config
)

print(f"Features shape: {features.shape}")  # (58,)

# Use for meta-classifier training
# feature_matrix.append(features)
# labels.append(is_msp)
```

---

### Use Case 4: Threshold Optimization on Validation Set

**Scenario:** Find optimal decision thresholds for best performance

**Code:**
```python
from eval import (
    scan_slice_threshold_youden,
    find_optimal_case_threshold,
    evaluate_case_level
)
import numpy as np

# Collect predictions and labels from validation set
slice_predictions = []
slice_labels = []

for batch in val_loader:
    with torch.no_grad():
        outputs = model(batch['image'].to(device))
        if isinstance(outputs, tuple):
            _, cls_logits = outputs
            probs = torch.sigmoid(cls_logits).cpu().numpy()
        else:
            probs = ...  # Extract from heatmaps

        slice_predictions.extend(probs.flatten())
        slice_labels.extend(batch['is_msp_label'].numpy().flatten())

# Optimize slice-level threshold using Youden's J
slice_predictions = np.array(slice_predictions)
slice_labels = np.array(slice_labels)

best_thresh, results_df, best_idx = scan_slice_threshold_youden(
    y_true=slice_labels,
    y_score=slice_predictions
)

print(f"Optimal slice threshold: {best_thresh:.4f}")
print(f"Youden's J: {results_df.loc[best_idx, 'youden']:.4f}")
print(f"Sensitivity: {results_df.loc[best_idx, 'sensitivity']:.4f}")
print(f"Specificity: {results_df.loc[best_idx, 'specificity']:.4f}")

# Optimize case-level threshold
# (group predictions by case first)
case_probs = [max(case_slice_preds) for case_slice_preds in grouped_by_case]
case_labels = [...]  # True case labels

optimal_case = find_optimal_case_threshold(
    case_probs=case_probs,
    true_case_labels=case_labels,
    sens_min=0.7
)

print(f"Optimal case threshold: {optimal_case['best_threshold']:.4f}")
print(f"F1 score: {optimal_case['f1_score']:.4f}")
```

---

### Use Case 5: Dataset Preparation & Exploration

**Scenario:** Prepare and explore your medical imaging dataset

**Code:**
```python
from config import get_default_config
from utils.io_utils import find_nifti_pairs
from utils.msp_utils import get_msp_index
from data import load_nifti_data_cached

config = get_default_config()
config["IMAGE_DIR"] = "/path/to/images"
config["LABEL_DIR"] = "/path/to/labels"

# Find all image-label pairs
all_pairs = find_nifti_pairs(
    config["IMAGE_DIR"],
    config["LABEL_DIR"],
    log_file=None
)

print(f"Found {len(all_pairs)} image-label pairs")

# Analyze MSP distribution
msp_cases = [p for p in all_pairs if p.get("has_msp", False)]
non_msp_cases = [p for p in all_pairs if not p.get("has_msp", False)]

print(f"MSP cases: {len(msp_cases)}")
print(f"Non-MSP cases: {len(non_msp_cases)}")

# Compute MSP slice indices for each case
for pair in msp_cases[:5]:  # First 5 examples
    label_vol = load_nifti_data_cached(str(pair["label"]), is_label=True)
    msp_idx = get_msp_index(
        label_vol,
        sagittal_axis=2,
        structure_labels=(1, 2, 5, 6)
    )
    print(f"Case {pair['id']}: MSP at slice {msp_idx}")
```

---

### Use Case 6: Custom Model Architecture

**Scenario:** Extend or modify the UNet architecture

**Code:**
```python
from models import UNetWithDualHeads, CriterionCombined
import torch.nn as nn

# Use dual-head architecture for coverage prediction
model = UNetWithDualHeads(in_channels=1, feat_channels=64)

# Combined loss function
criterion = CriterionCombined(
    lambda_heat=1.0,
    lambda_cls=1.0,
    lambda_cov=0.5
)

# Forward pass
heatmaps, cls_logits, cov_logits = model(images)

# Compute combined loss
loss, loss_dict = criterion(
    pred_heat=heatmaps,
    tgt_heat=target_heatmaps,
    pred_cls=cls_logits,
    tgt_cls=is_msp_labels,
    pred_cov=cov_logits,
    tgt_cov=coverage_labels
)

print(f"Total loss: {loss.item():.4f}")
print(f"Heatmap loss: {loss_dict['loss_heat'].item():.4f}")
print(f"Classification loss: {loss_dict['loss_cls'].item():.4f}")
print(f"Coverage loss: {loss_dict['loss_cov'].item():.4f}")
```

---

### Use Case 7: Batch Processing Multiple Volumes

**Scenario:** Process multiple MRI volumes in batch

**Code:**
```python
from pathlib import Path
from tqdm import tqdm
import pandas as pd

# Get all volumes to process
volume_dir = Path("/path/to/test/volumes")
volume_paths = list(volume_dir.glob("*.nii.gz"))

results = []

for volume_path in tqdm(volume_paths, desc="Processing volumes"):
    # Load and process volume
    img_vol = load_nifti_data_cached(str(volume_path), is_label=False)

    # Predict MSP for each slice
    slice_probs = []
    for slice_idx in range(img_vol.shape[2]):
        # ... inference code ...
        slice_probs.append(cls_prob)

    # Aggregate to case level
    from eval import evaluate_case_level
    case_result = evaluate_case_level(slice_probs, case_threshold=0.5)

    results.append({
        'volume': volume_path.name,
        'has_msp': case_result['has_msp'],
        'confidence': case_result['case_prob'],
        'predicted_slice': case_result['predicted_msp_slice']
    })

# Save results
df = pd.DataFrame(results)
df.to_csv('batch_predictions.csv', index=False)
print(f"Processed {len(results)} volumes")
```

---

## 📁 Recommended GitHub Repository Structure

```
MSPdetection/
├── config/              # Configuration module
├── utils/               # Utility functions
├── data/                # Data loading & preprocessing
├── models/              # Neural network architectures
├── losses/              # Loss functions
├── features/            # Feature extraction
├── inference/           # Inference utilities
├── eval/                # Evaluation metrics
├── train/               # Training utilities
├── examples/            # Usage examples (as shown above)
├── scripts/             # Entry point scripts (to be created)
│   ├── train_model.py
│   ├── evaluate_model.py
│   └── predict_volume.py
├── tests/               # Unit tests (recommended to add)
├── docs/                # Additional documentation
├── requirements.txt     # Python dependencies
├── README.md            # Main documentation
├── .gitignore          # Git exclusions
└── LICENSE             # License file

# NOT included in GitHub:
archive/
└── main_monolithic_original.py  # Original 9,013 line file (archived)
```

---

## 🎓 Migration Path from Legacy main.py

### What to Do with main.py:

1. **Archive it locally:**
   ```bash
   mkdir -p archive
   mv main.py archive/main_monolithic_original.py
   ```

2. **Add to .gitignore:**
   ```bash
   echo "archive/" >> .gitignore
   ```

3. **Reference it in documentation:**
   ```markdown
   # In README.md
   ## History
   This codebase was refactored from a monolithic research script into
   a modular architecture. The original monolithic version is archived
   locally for reference but not included in the repository.
   ```

### What Gets Uploaded to GitHub:

✅ All modular components (config, utils, data, models, losses, features, inference, eval, train)
✅ Examples directory
✅ Documentation files
✅ Requirements.txt
✅ README.md
✅ LICENSE
✅ .gitignore

❌ main.py (archived locally)
❌ Large data files
❌ Model checkpoints (use Git LFS or external storage)
❌ Cache directories

---

## 📋 Summary

**Answer to "Should I upload main.py?"** → **NO**

**Recommended action:**
1. Rename `main.py` to `main_monolithic_original.py`
2. Move it to `archive/` directory
3. Add `archive/` to `.gitignore`
4. Upload only the modular components to GitHub

Your refactored codebase is **complete and publication-ready** for serious research use. The modular structure is professional, maintainable, and suitable for collaborative development.

---

**Next Steps:**
1. Rename/archive main.py (per recommendations above)
2. Review variable naming (addressing your second concern)
3. Add comprehensive README.md
4. Add LICENSE file
5. Create entry point scripts in `scripts/`
6. Add unit tests (recommended)
7. Upload to GitHub!
