"""
Example: Complete data loading and preprocessing pipeline.

Demonstrates how to use the data module for MSP detection.
"""

from config import get_default_config
from data import (
    HeatmapDataset,
    create_balanced_dataloader,
    load_nifti_data_cached,
    extract_slice,
    normalize_slice,
)

def main():
    # Load configuration
    config = get_default_config()

    print("=" * 70)
    print("MSP Detection - Data Pipeline Example")
    print("=" * 70)

    # Example 1: Load a NIfTI file with caching
    print("\n1. Loading NIfTI file with caching...")
    example_image_path = "path/to/your/image.nii.gz"
    # img_vol = load_nifti_data_cached(example_image_path, is_label=False)
    # print(f"   Loaded volume shape: {img_vol.shape}")

    # Example 2: Extract and normalize a slice
    print("\n2. Extracting and normalizing a slice...")
    # slice_idx = 100
    # img_slice = extract_slice(img_vol, slice_idx, axis=2)
    # img_slice_norm = normalize_slice(img_slice, config)
    # print(f"   Slice shape: {img_slice.shape}")
    # print(f"   Normalized range: [{img_slice_norm.min():.3f}, {img_slice_norm.max():.3f}]")

    # Example 3: Create dataset from slice references
    print("\n3. Creating dataset from slice references...")

    # Create sample data references (in practice, use find_nifti_pairs)
    sample_data_refs = [
        {
            "image_path": "path/to/image1.nii.gz",
            "label_path": "path/to/label1.nii.gz",
            "slice_idx": 100,
            "case_id": "patient_001",
            "patient_id": "patient_001",
            "is_msp": True
        },
        # Add more references...
    ]

    # Create training dataset with augmentation
    # train_dataset = HeatmapDataset(
    #     sample_data_refs,
    #     config,
    #     is_train=True,
    #     split_name="train"
    # )
    # print(f"   Dataset size: {len(train_dataset)} slices")

    # Example 4: Create balanced dataloader
    print("\n4. Creating balanced dataloader...")
    # train_loader = create_balanced_dataloader(
    #     train_dataset,
    #     config,
    #     is_train=True
    # )
    # print(f"   Batch size: {config['BATCH_SIZE']}")
    # print(f"   Number of batches: {len(train_loader)}")

    # Example 5: Iterate through batches
    print("\n5. Batch structure:")
    print("   Each batch contains:")
    print("     - 'image': [B, 1, H, W] - Input images")
    print("     - 'target_heatmap': [B, 4, H, W] - Target heatmaps (4 structures)")
    print("     - 'brain_mask': [B, H, W] - Brain masks")
    print("     - 'is_msp_label': [B, 1] - MSP labels (0 or 1)")
    print("     - 'cov_label': [B] - Coverage labels (0/1/2)")
    print("     - 'has_regression_target': [B] - Boolean flags")
    print("     - 'slice_id': List[str] - Slice identifiers")

    # for batch_idx, batch in enumerate(train_loader):
    #     if batch_idx == 0:  # Show first batch
    #         print(f"\n   First batch:")
    #         print(f"     Images shape: {batch['image'].shape}")
    #         print(f"     Targets shape: {batch['target_heatmap'].shape}")
    #         print(f"     Brain masks shape: {batch['brain_mask'].shape}")
    #         break

    print("\n" + "=" * 70)
    print("✅ Data pipeline example complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
