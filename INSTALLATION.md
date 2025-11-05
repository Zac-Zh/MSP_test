# MSP Detection - Installation & Setup Guide

## Quick Start

```bash
# 1. Clone the repository
git clone https://github.com/your-username/MSPdetection.git
cd MSPdetection

# 2. Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Verify installation
python -c "from config import get_default_config; print('✅ Installation successful!')"
```

---

## Prerequisites

### System Requirements
- **Python:** 3.8 or higher
- **CUDA:** 11.0+ (optional, for GPU acceleration)
- **RAM:** 16 GB recommended
- **Storage:** 10 GB+ for models and cache

### Required Libraries
All dependencies are listed in `requirements.txt`:
- PyTorch >= 1.10.0
- NumPy >= 1.21.0
- OpenCV >= 4.5.0
- nibabel (for NIfTI medical imaging)
- albumentations (for data augmentation)
- scikit-learn, scipy, pandas
- tqdm (for progress bars)

---

## Installation Steps

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/MSPdetection.git
cd MSPdetection
```

### 2. Create Virtual Environment (Recommended)

**Linux/Mac:**
```bash
python3 -m venv venv
source venv/bin/activate
```

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

### 3. Install PyTorch

Choose the appropriate command based on your system:

**CPU-only:**
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

**CUDA 11.8:**
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

**CUDA 12.1:**
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

For other versions, see: https://pytorch.org/get-started/locally/

### 4. Install Remaining Dependencies

```bash
pip install -r requirements.txt
```

### 5. Verify Installation

Run this command to verify all imports work:

```bash
python -c "
from config import get_default_config
from data import HeatmapDataset, create_balanced_dataloader
from models import UNetWithCls
from train import prepare_patient_grouped_datasets
from eval import find_optimal_case_threshold
print('✅ All imports successful!')
print('✅ Installation complete!')
"
```

---

## Configuration

### 1. Review Default Configuration

```bash
python -c "from config import get_default_config; import pprint; pprint.pprint(get_default_config())"
```

### 2. Set Your Data Paths

Create a configuration script (e.g., `my_config.py`):

```python
from config import get_default_config

def get_my_config():
    config = get_default_config()

    # Update paths to your data
    config["IMAGE_DIR"] = "/path/to/your/images"
    config["LABEL_DIR"] = "/path/to/your/labels"
    config["CACHE_DIR"] = "./cache"
    config["LOG_DIR"] = "./logs"

    # Adjust training parameters if needed
    config["BATCH_SIZE"] = 16
    config["NUM_EPOCHS"] = 100
    config["LEARNING_RATE"] = 1e-4

    # Set device
    config["DEVICE"] = "cuda" if torch.cuda.is_available() else "cpu"

    return config
```

### 3. Create Required Directories

```bash
mkdir -p cache logs checkpoints results
```

---

## Quick Test

### Test 1: Data Loading

```python
from config import get_default_config
from utils.io_utils import find_nifti_pairs

config = get_default_config()
config["IMAGE_DIR"] = "/path/to/your/images"
config["LABEL_DIR"] = "/path/to/your/labels"

# Find all image-label pairs
pairs = find_nifti_pairs(config["IMAGE_DIR"], config["LABEL_DIR"])
print(f"Found {len(pairs)} image-label pairs")
```

### Test 2: Model Creation

```python
import torch
from models import UNetWithCls

# Create model
model = UNetWithCls(n_channels=1, n_classes=4)
print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")

# Test forward pass
dummy_input = torch.randn(2, 1, 512, 512)  # Batch of 2 images
heatmaps, cls_logits = model(dummy_input)
print(f"Heatmaps shape: {heatmaps.shape}")  # [2, 4, 512, 512]
print(f"Classification logits shape: {cls_logits.shape}")  # [2, 1]
```

### Test 3: Run Examples

```bash
# Test data pipeline
python examples/example_data_pipeline.py

# Test model training (without actual training)
python examples/example_model_training.py

# Test inference
python examples/example_inference.py
```

---

## Data Preparation

### Expected Data Structure

Your data should be organized as follows:

```
/path/to/your/data/
├── images/
│   ├── patient001_scan1.nii.gz
│   ├── patient001_scan2.nii.gz
│   ├── patient002_scan1.nii.gz
│   └── ...
└── labels/
    ├── patient001_scan1_label.nii.gz
    ├── patient001_scan2_label.nii.gz
    ├── patient002_scan1_label.nii.gz
    └── ...
```

### NIfTI File Requirements

- **Format:** NIfTI (.nii or .nii.gz)
- **Orientation:** Sagittal slices along axis 2 (default)
- **Labels:** Integer labels (0=background, 1-6=structures)
- **Naming:** Image and label files should have matching base names

### Structure Labels

The default configuration expects these anatomical structure labels:
- **0:** Background
- **1:** Corpus Callosum
- **2:** Parent of structure 4
- **3:** Parent of structure 5
- **4:** Small structure (remapped to 2)
- **5:** Small structure (remapped to 3)
- **6:** Additional structure

These are automatically handled by `remap_small_structures_to_parent()`.

---

## Training Your Own Model

### Quick Start: Train a Baseline Model

We provide a complete training script that you can run directly:

```bash
# 1. Edit train_baseline.py to set your data paths
nano train_baseline.py  # Update IMAGE_DIR and LABEL_DIR

# 2. Run training
python train_baseline.py

# Output:
# - Trained model saved to: checkpoints/best_baseline_model.pth
# - Training log saved to: logs/training_baseline.log
```

The training script will:
- Automatically split your data into train/validation sets (patient-level split)
- Create balanced dataloaders
- Train a UNetWithCls model
- Save the best model based on validation loss
- Apply early stopping to prevent overfitting

### Evaluate Your Trained Model

After training, evaluate the model performance:

```bash
python evaluate_model.py --model_path checkpoints/best_baseline_model.pth

# Output:
# - Slice-level metrics (accuracy, sensitivity, specificity, F1)
# - Case-level metrics (aggregated predictions)
# - Optimal thresholds for both levels
```

### Predict MSP for a New Volume

Use your trained model to predict MSP on a new volume:

```bash
python predict_volume.py \
    --volume path/to/new_volume.nii.gz \
    --model checkpoints/best_baseline_model.pth \
    --output results/prediction.json

# Output:
# - Predicted MSP slice index
# - Confidence scores for all slices
# - Results saved to JSON file
```

---

## Advanced Usage Examples

### Example 1: Custom Training Loop

See [USE_CASES.md](USE_CASES.md) for complete examples.

```python
from config import get_default_config
from train import prepare_patient_grouped_datasets
from data import HeatmapDataset, create_balanced_dataloader
from models import UNetWithCls
from losses import DiceLoss
import torch.optim as optim

# Setup
config = get_default_config()
config["IMAGE_DIR"] = "/path/to/images"
config["LABEL_DIR"] = "/path/to/labels"

# Prepare datasets
train_refs, val_refs, _ = prepare_patient_grouped_datasets(config)

# Create dataloaders
train_dataset = HeatmapDataset(train_refs, config, is_train=True)
train_loader = create_balanced_dataloader(train_dataset, config, is_train=True)

# Train model
model = UNetWithCls(n_channels=1, n_classes=4).to(config["DEVICE"])
criterion = DiceLoss()
optimizer = optim.Adam(model.parameters(), lr=config["LEARNING_RATE"])

# Custom training loop
for epoch in range(config["NUM_EPOCHS"]):
    for batch in train_loader:
        images = batch["image"].to(device)
        targets = batch["target_heatmap"].to(device)

        heatmaps, cls_logits = model(images)
        loss = criterion(torch.sigmoid(heatmaps), targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

### Example 2: Programmatic Inference

```python
from train import load_model_with_correct_architecture
from data import load_nifti_data_cached, extract_slice, normalize_slice
from inference import apply_tta_horizontal_flip
import torch

# Load model
model, _ = load_model_with_correct_architecture(
    "checkpoints/best_baseline_model.pth",
    config,
    device
)

# Load volume
volume = load_nifti_data_cached("path/to/new_volume.nii.gz", is_label=False)

# Process slices
for slice_idx in range(volume.shape[2]):
    img_slice = extract_slice(volume, slice_idx, axis=2)
    img_norm = normalize_slice(img_slice, config)
    # ... inference code ...
```

---

## Troubleshooting

### Import Errors

**Problem:** `ModuleNotFoundError: No module named 'torch'`

**Solution:**
```bash
# Ensure virtual environment is activated
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows

# Reinstall PyTorch
pip install torch torchvision
```

### CUDA Errors

**Problem:** `RuntimeError: CUDA out of memory`

**Solution:**
```python
# Reduce batch size in config
config["BATCH_SIZE"] = 8  # or smaller

# Or use CPU
config["DEVICE"] = "cpu"
```

### Data Loading Errors

**Problem:** `ValueError: No NIfTI pairs found!`

**Solution:**
```bash
# Check directory paths
ls /path/to/images/*.nii.gz
ls /path/to/labels/*.nii.gz

# Verify file naming matches
python -c "
from utils.io_utils import find_nifti_pairs
pairs = find_nifti_pairs('/path/to/images', '/path/to/labels')
print(f'Found {len(pairs)} pairs')
"
```

### Memory Issues

**Problem:** System runs out of RAM during data loading

**Solution:**
```python
# Adjust cache size in config
config["CACHE_SIZE"] = 50  # Reduce from default 100

# Or disable caching for very large datasets
# (modify data/loaders.py to use load_nifti_data instead of load_nifti_data_cached)
```

---

## Directory Structure After Setup

```
MSPdetection/
├── cache/              # Cached preprocessed data (created automatically)
├── checkpoints/        # Saved model checkpoints (create manually)
├── config/             # Configuration module
├── data/               # Data loading & preprocessing
├── docs/               # Documentation
├── eval/               # Evaluation metrics
├── examples/           # Usage examples
├── features/           # Feature extraction
├── inference/          # Inference utilities
├── logs/               # Training logs (created automatically)
├── losses/             # Loss functions
├── models/             # Model architectures
├── results/            # Evaluation results (create manually)
├── train/              # Training utilities
├── utils/              # Utility functions
├── venv/               # Virtual environment (created by you)
├── CONTRIBUTING.md     # Contribution guidelines
├── INSTALLATION.md     # This file
├── LICENSE             # License information
├── README.md           # Project overview
├── requirements.txt    # Python dependencies
└── USE_CASES.md        # Detailed usage examples
```

---

## Next Steps

1. **Review configuration:** Check [config/config.py](config/config.py)
2. **Explore examples:** See [examples/](examples/) directory
3. **Read use cases:** Check [USE_CASES.md](USE_CASES.md) for complete workflows
4. **Prepare data:** Organize your NIfTI files
5. **Start training:** Follow Example 1 in [USE_CASES.md](USE_CASES.md)

---

## Getting Help

- **Examples:** See [examples/README.md](examples/README.md)
- **Use Cases:** See [USE_CASES.md](USE_CASES.md)
- **API Documentation:** Check docstrings in module files
- **Issues:** Report bugs on GitHub Issues

---

## License

See [LICENSE](LICENSE) file for details.

---

**Installation Guide Version:** 1.0
**Last Updated:** November 2, 2025
**Python Version:** 3.8+
**PyTorch Version:** 1.10.0+
