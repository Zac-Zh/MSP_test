"""
Single Volume MSP Prediction Script

This script predicts the MSP slice for a single MRI volume.

Usage:
    python predict_volume.py --volume path/to/volume.nii.gz --model checkpoints/best_baseline_model.pth
"""

import torch
import numpy as np
import argparse
from pathlib import Path

from config import get_default_config
from train import load_model_with_correct_architecture
from data import load_nifti_data_cached, extract_slice, normalize_slice
from inference import apply_tta_horizontal_flip
import cv2


def predict_msp_for_volume(volume_path, model, config, device, use_tta=True):
    """
    Predict MSP slice for a single volume.

    Args:
        volume_path: Path to NIfTI volume
        model: Trained model
        config: Configuration dict
        device: torch device
        use_tta: Whether to use test-time augmentation

    Returns:
        dict with prediction results
    """
    # Load volume
    print(f"Loading volume: {volume_path}")
    volume_data = load_nifti_data_cached(str(volume_path), is_label=False)

    print(f"Volume shape: {volume_data.shape}")

    # Get sagittal axis (default is axis 2)
    sagittal_axis = config.get("SAGITTAL_AXIS", 2)
    num_slices = volume_data.shape[sagittal_axis]

    print(f"Number of slices (axis {sagittal_axis}): {num_slices}")

    # Process each slice
    slice_probs = []
    H, W = config["IMAGE_SIZE"]

    model.eval()

    print("Processing slices...")
    for slice_idx in range(num_slices):
        # Extract and preprocess slice
        img_slice = extract_slice(volume_data, slice_idx, axis=sagittal_axis)
        img_norm = normalize_slice(img_slice, config)

        # Resize to model input size
        img_resized = cv2.resize(img_norm, (W, H))

        # Convert to tensor
        img_tensor = torch.from_numpy(img_resized).unsqueeze(0).unsqueeze(0).float().to(device)

        # Inference
        with torch.no_grad():
            if use_tta:
                outputs = apply_tta_horizontal_flip(img_tensor, model)
            else:
                outputs = model(img_tensor)

            # Extract classification probability
            if isinstance(outputs, tuple):
                _, cls_logits = outputs
                prob = torch.sigmoid(cls_logits).item()
            else:
                # If only heatmap output, use max probability
                heatmaps = outputs
                prob = torch.sigmoid(heatmaps).max().item()

        slice_probs.append(prob)

    slice_probs = np.array(slice_probs)

    # Find MSP slice (slice with highest probability)
    msp_slice_idx = int(np.argmax(slice_probs))
    msp_confidence = float(slice_probs[msp_slice_idx])

    # Compute statistics
    results = {
        'volume_path': str(volume_path),
        'num_slices': num_slices,
        'predicted_msp_slice': msp_slice_idx,
        'msp_confidence': msp_confidence,
        'mean_confidence': float(slice_probs.mean()),
        'std_confidence': float(slice_probs.std()),
        'max_confidence': float(slice_probs.max()),
        'all_slice_probs': slice_probs.tolist()
    }

    return results


def main():
    """Main inference function."""
    parser = argparse.ArgumentParser(description='Predict MSP slice for a single volume')
    parser.add_argument('--volume', type=str, required=True,
                        help='Path to NIfTI volume file')
    parser.add_argument('--model', type=str, default='checkpoints/best_baseline_model.pth',
                        help='Path to trained model checkpoint')
    parser.add_argument('--no-tta', action='store_true',
                        help='Disable test-time augmentation')
    parser.add_argument('--output', type=str, default=None,
                        help='Path to save prediction results (JSON format)')
    args = parser.parse_args()

    # Check if volume exists
    volume_path = Path(args.volume)
    if not volume_path.exists():
        print(f"ERROR: Volume not found: {volume_path}")
        return

    # Check if model exists
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"ERROR: Model not found: {model_path}")
        return

    # ===== CONFIGURATION =====
    config = get_default_config()
    config["DEVICE"] = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(config["DEVICE"])

    print("=" * 60)
    print("MSP Detection - Single Volume Prediction")
    print("=" * 60)
    print(f"Volume: {volume_path}")
    print(f"Model: {model_path}")
    print(f"Device: {config['DEVICE']}")
    print(f"TTA: {'Disabled' if args.no_tta else 'Enabled'}")
    print("=" * 60)

    # ===== LOAD MODEL =====
    print("\nLoading model...")
    model, model_type = load_model_with_correct_architecture(
        str(model_path),
        config,
        device,
        log_file=None
    )
    print(f"✅ Loaded model type: {model_type}")

    # ===== PREDICT =====
    print("\nRunning inference...")
    results = predict_msp_for_volume(
        volume_path,
        model,
        config,
        device,
        use_tta=not args.no_tta
    )

    # ===== DISPLAY RESULTS =====
    print("\n" + "=" * 60)
    print("Prediction Results")
    print("=" * 60)
    print(f"Volume: {volume_path.name}")
    print(f"Total slices: {results['num_slices']}")
    print(f"\n🎯 Predicted MSP slice: {results['predicted_msp_slice']}")
    print(f"   Confidence: {results['msp_confidence']:.4f}")
    print(f"\nStatistics:")
    print(f"   Mean confidence: {results['mean_confidence']:.4f}")
    print(f"   Std confidence:  {results['std_confidence']:.4f}")
    print(f"   Max confidence:  {results['max_confidence']:.4f}")
    print("=" * 60)

    # ===== SAVE RESULTS =====
    if args.output:
        import json
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"\n✅ Results saved to: {output_path}")

    # ===== VISUALIZATION SUGGESTION =====
    print("\n💡 Tip: To visualize the prediction, you can:")
    print(f"   1. Load the volume at slice {results['predicted_msp_slice']}")
    print(f"   2. Check probabilities in all_slice_probs")
    print(f"   3. Plot slice probabilities vs. slice index")


if __name__ == "__main__":
    main()
