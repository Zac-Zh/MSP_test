"""
PyTorch Dataset classes for MSP detection.

Provides dataset implementations for loading and preprocessing medical imaging
data for training and evaluation.
"""

import torch
import cv2
import numpy as np
from torch.utils.data import Dataset
from .loaders import load_nifti_data_cached
from .preprocessing import (
    preprocess_and_cache,
    extract_slice,
    normalize_slice,
    generate_brain_mask_from_image,
    create_target_heatmap_with_distance_transform,
    get_transforms,
    remap_small_structures_to_parent,
)


class HeatmapDataset(Dataset):
    def __init__(self, data_references, config, is_train=True, split_name="train"):
        self.data_references = data_references
        self.config = config
        self.is_train = is_train
        self.split_name = split_name
        self.sagittal_axis = config["SAGITTAL_AXIS"]
        self.structure_labels = config["STRUCTURE_LABELS"]
        self.cache_dir = config["CACHE_DIR"]
        self.transforms = get_transforms(config, is_train)

        if not self.data_references:
            # log_message(f"Warning: Initialized {split_name} dataset with 0 references.", config.get("log_file"))
            pass
        else:
            # log_message(f"Initialized {split_name} dataset with {len(data_references)} slice references.",
            #             config.get("log_file"))
            pass

        # Stability/Consistency Check: Ensure heatmap channel definitions match
        if "HEATMAP_LABEL_MAP" in self.config:
            assert set(self.config["HEATMAP_LABEL_MAP"]) == set(self.structure_labels), \
                "STRUCTURE_LABELS and HEATMAP_LABEL_MAP must match to keep channel semantics aligned."


    def __len__(self):
        return len(self.data_references)

    def __getitem__(self, idx: int):
        """
        Loads and preprocesses a single slice:
          - Image preprocessing / normalization
          - Target heatmap (generated only for GT with offset==0)
          - Tri-classification coverage label (cov_label in {0,1,2})
          - Other metadata (soft_label, category, offset...)
        """
        slice_ref = self.data_references[idx]
        slice_id_str = f"{slice_ref.get('case_id', 'unknown')}_{slice_ref.get('slice_idx', idx)}"
        log_file = self.config.get("log_file")

        # Define is_msp once and use it consistently
        is_msp = bool(slice_ref.get("is_msp", False))

        try:
            # ------------------------------------------------------------------
            # 1. Load volume data & extract slice
            # ------------------------------------------------------------------
            img_vol, _ = preprocess_and_cache(str(slice_ref["image_path"]),
                                              self.cache_dir,
                                              self.config,
                                              log_file)
            if img_vol is None:
                return self._get_dummy_item(slice_id_str, is_msp=is_msp)

            slice_idx = slice_ref["slice_idx"]
            img_slice = extract_slice(img_vol, slice_idx, self.sagittal_axis)
            if img_slice is None:
                return self._get_dummy_item(slice_id_str, is_msp=is_msp)

            # 2. Normalize
            img_slice_norm = normalize_slice(img_slice, self.config)

            # 3. Generate brain mask
            if self.config.get("GENERATE_BRAIN_MASK_FROM_IMAGE", True):
                brain_mask_orig = generate_brain_mask_from_image(img_slice_norm, self.config)
            else:
                brain_mask_orig = np.ones_like(img_slice_norm, dtype=np.float32)

            # 4. Load annotation info
            num_channels = len(self.structure_labels)
            target_heatmap = np.zeros((num_channels, *img_slice_norm.shape), dtype=np.float32)
            has_reg_target = False
            cov_label_val = 0  # Default: no structure

            # ------------------------------------------------------------------
            # 5. Generate target heatmap & coverage label only for GT MSP slices
            # ------------------------------------------------------------------
            if is_msp:
                try:
                    label_vol = load_nifti_data_cached(str(slice_ref["label_path"]), is_label=True)
                    if label_vol is not None:
                        label_slice = extract_slice(label_vol, slice_idx, self.sagittal_axis)
                        if label_slice is not None:
                            # Generate target heatmap
                            target_heatmap = create_target_heatmap_with_distance_transform(
                                label_slice,
                                self.structure_labels,
                                img_slice_norm.shape,
                                brain_mask=brain_mask_orig
                            )
                            has_reg_target = True

                            # Calculate coverage label
                            remapped_label = remap_small_structures_to_parent(label_slice)
                            union_mask = np.isin(remapped_label, self.structure_labels)
                            coverage = union_mask.mean()
                            if coverage >= 0.7:
                                cov_label_val = 2
                            elif coverage > 0:
                                cov_label_val = 1
                            else:
                                cov_label_val = 0
                except Exception as e_label:
                    has_reg_target = False
                    cov_label_val = 0
                    # log_message(f"Label processing error @ {slice_id_str}: {e_label}", log_file)

            # ------------------------------------------------------------------
            # 6. Data Augmentation / Resizing
            # ------------------------------------------------------------------
            if self.transforms:
                # Albumentations requires HWC format
                target_hwc = np.transpose(target_heatmap, (1, 2, 0)).astype(np.float32)
                aug = self.transforms(
                    image=img_slice_norm.astype(np.float32),
                    target=target_hwc,
                    brain_mask=brain_mask_orig.astype(np.float32)
                )
                img_aug = aug["image"]
                target_aug = aug["target"]
                brain_mask = aug["brain_mask"]

                # Albumentations ToTensorV2 already returns a torch.Tensor
                img_tensor = img_aug.float()
                target_tensor = target_aug.permute(2, 0, 1).float() # HWC -> CHW
                brain_mask_t = brain_mask.float()
                # Ensure brain mask is strictly binary
                brain_mask_t = (brain_mask_t > 0.5).float()

            else:
                H, W = self.config["IMAGE_SIZE"]
                img_resized = cv2.resize(img_slice_norm, (W, H), interpolation=cv2.INTER_LINEAR)
                target_resized = cv2.resize(np.transpose(target_heatmap, (1, 2, 0)), (W, H),
                                            interpolation=cv2.INTER_NEAREST)
                brain_resized = cv2.resize(brain_mask_orig, (W, H), interpolation=cv2.INTER_NEAREST)

                img_tensor = torch.from_numpy(img_resized.astype(np.float32)).unsqueeze(0).float()
                target_tensor = torch.from_numpy(np.transpose(target_resized, (2, 0, 1)).astype(np.float32)).float()
                brain_mask_t = torch.from_numpy(brain_resized.astype(np.float32)).float()
                # Ensure brain mask is strictly binary
                brain_mask_t = (brain_mask_t > 0.5).float()

            # ------------------------------------------------------------------
            # 7. Package and return (with critical fix)
            # ------------------------------------------------------------------
            # Critical Fix: Create independent, resizable memory copies to avoid errors
            # with multiprocessing in DataLoader.
            img_tensor = img_tensor.contiguous().clone()
            target_tensor = target_tensor.contiguous().clone()
            brain_mask_t = brain_mask_t.contiguous().clone()

            return {
                "image": img_tensor,
                "target_heatmap": target_tensor,
                "brain_mask": brain_mask_t,
                "has_regression_target": torch.tensor(has_reg_target, dtype=torch.bool),
                "is_msp_label": torch.tensor(float(is_msp), dtype=torch.float32).unsqueeze(0),
                "cov_label": torch.tensor(cov_label_val, dtype=torch.long),
                "slice_id": slice_id_str,
                "true_is_msp": torch.tensor(float(is_msp), dtype=torch.float32).unsqueeze(0),
            }

        except Exception as e:
            # log_message(f"CRITICAL ERROR in __getitem__ {slice_id_str}: {e}\n{traceback.format_exc()}",
            #             log_file)
            return self._get_dummy_item(f"error_{slice_id_str}", is_msp=is_msp)

    def _get_dummy_item(self, slice_id, is_msp=False):
        """Generates a placeholder item in case of a loading error."""
        H, W = self.config["IMAGE_SIZE"]
        n_heatmap_channels = len(self.structure_labels)
        return {
            "image": torch.zeros(self.config["IN_CHANNELS"], H, W, dtype=torch.float32),
            "target_heatmap": torch.zeros(n_heatmap_channels, H, W, dtype=torch.float32),
            "brain_mask": torch.ones(H, W, dtype=torch.float32),
            "has_regression_target": torch.tensor(False, dtype=torch.bool),
            "is_msp_label": torch.tensor(float(is_msp), dtype=torch.float32).unsqueeze(0),
            "slice_id": slice_id,
            "true_is_msp": torch.tensor(float(is_msp), dtype=torch.float32).unsqueeze(0),
            "cov_label": torch.tensor(0, dtype=torch.long),
        }
