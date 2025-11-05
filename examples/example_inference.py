"""
Example: Model inference with test-time augmentation.

Demonstrates how to use trained models for MSP detection inference.
"""

import torch
import numpy as np
from config import get_default_config
from train import load_model_with_correct_architecture
from inference import apply_tta_horizontal_flip
from features import extract_heatmap_features
from data import load_nifti_data_cached, extract_slice, normalize_slice
import cv2


def main():
    print("=" * 70)
    print("MSP Detection - Inference Example")
    print("=" * 70)

    # Configuration
    config = get_default_config()
    device = torch.device(config["DEVICE"])
    print(f"\n✅ Using device: {device}")

    # Load trained model
    print("\n1. Loading trained model...")
    model_path = "path/to/best_model.pth"

    # In practice, you would load the model like this:
    """
    model, model_type = load_model_with_correct_architecture(
        model_path,
        config,
        device,
        log_file=None
    )
    model.eval()
    print(f"   Loaded model type: {model_type}")
    """
    print("   (Skipping model loading - requires trained checkpoint)")

    # Load and preprocess image
    print("\n2. Loading and preprocessing image...")
    # In practice:
    """
    img_vol = load_nifti_data_cached("path/to/test_image.nii.gz", is_label=False)
    slice_idx = 100
    img_slice = extract_slice(img_vol, slice_idx, axis=2)
    img_slice_norm = normalize_slice(img_slice, config)

    # Resize to model input size
    H, W = config["IMAGE_SIZE"]
    img_resized = cv2.resize(img_slice_norm, (W, H), interpolation=cv2.INTER_LINEAR)
    img_tensor = torch.from_numpy(img_resized).unsqueeze(0).unsqueeze(0).float().to(device)
    """
    print("   (Skipping image loading - requires actual NIfTI file)")

    # Inference with TTA
    print("\n3. Running inference with test-time augmentation...")
    print("   Example code:")
    print("   ```python")
    print("   with torch.no_grad():")
    print("       # Apply horizontal flip TTA")
    print("       outputs = apply_tta_horizontal_flip(img_tensor, model)")
    print("       ")
    print("       # Handle different model types")
    print("       if isinstance(outputs, tuple):")
    print("           if len(outputs) == 3:")
    print("               # UNetWithDualHeads")
    print("               heatmaps, cls_logits, cov_logits = outputs")
    print("           elif len(outputs) == 2:")
    print("               # UNetWithCls")
    print("               heatmaps, cls_logits = outputs")
    print("       else:")
    print("           # UNetHeatmap")
    print("           heatmaps = outputs")
    print("   ```")

    # Feature extraction
    print("\n4. Extracting features for meta-classifier...")
    print("   Example code:")
    print("   ```python")
    print("   # Convert heatmap logits to numpy")
    print("   heatmap_logits_np = heatmaps[0].cpu().numpy()  # [C, H, W]")
    print("   ")
    print("   # Extract 58-dimensional feature vector")
    print("   features = extract_heatmap_features(")
    print("       heatmap_logits_np,")
    print("       brain_mask=None,  # Or provide brain mask if available")
    print("       config=config")
    print("   )")
    print("   print(f'Features shape: {features.shape}')  # (58,)")
    print("   ")
    print("   # Use meta-classifier for final prediction")
    print("   # meta_clf_prob = meta_classifier.predict_proba(features.reshape(1, -1))[0, 1]")
    print("   ```")

    # Post-processing
    print("\n5. Post-processing predictions...")
    print("   - Apply sigmoid to heatmap logits to get probabilities")
    print("   - Threshold heatmaps to get binary masks")
    print("   - Find peak locations in heatmaps for keypoint detection")
    print("   - Aggregate slice-level predictions to case-level decision")

    print("\n" + "=" * 70)
    print("✅ Inference example complete!")
    print("=" * 70)
    print("\nKey functions used:")
    print("  - load_model_with_correct_architecture() - Load trained model")
    print("  - apply_tta_horizontal_flip() - Test-time augmentation")
    print("  - extract_heatmap_features() - Feature extraction")
    print("  - Data preprocessing functions")


if __name__ == "__main__":
    main()
