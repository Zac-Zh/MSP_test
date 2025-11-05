# Variable Naming Review & Corrections

## 🔍 Analysis of Confusing Variable Names

After systematic review of all modules, here are the confusing/non-intuitive variable names that need correction:

---

## ⚠️ Issues Found

### 1. **eval/metrics.py**

**Issue:** Single-letter variables in mathematical contexts
- `P`, `N` - Positive and Negative counts
- `J` - Youden's J statistic
- `t` - threshold
- `y`, `y_hat` - predictions

**Assessment:** These are **standard statistical notation** - should be kept as-is with comments

---

### 2. **data/preprocessing.py**

**Issue:** `s` for slice in `extract_slice()`
```python
s = volume_data[slice_idx, :, :]
return s.copy()
```

**Fix Needed:** ✅ Change `s` → `slice_2d` for clarity

---

### 3. **features/extraction.py**

**Issue:** Multiple confusing variables:
- `cx_vals`, `cy_vals` - centroid x/y values (unclear)
- `sx`, `sy` - spacing x/y (too short)
- `C_in`, `H`, `W` - Channels, Height, Width (C_in is confusing)
- `probs_np` - redundant `_np` suffix

**Fixes Needed:**
- ✅ `cx_vals` → `centroid_x_values`
- ✅ `cy_vals` → `centroid_y_values`
- ✅ `C_in` → `num_channels`
- ✅ Keep `H`, `W` (standard for image dimensions)
- ✅ Keep `sx`, `sy` in specific context (standard for spacing)

---

### 4. **models/unet_dual_heads.py**

**Issue:** `p4`, `p5`, `gt4`, `gt5` in compute_keypoint_constrained_loss
- Not present in current extraction (function not extracted yet)

---

### 5. **data/datasets.py**

**Issue:** Abbreviations and unclear names:
- `ref` - data reference (too short)
- `img_vol` - image volume (acceptable but could be clearer)
- `msp_idx` - MSP index (acceptable)
- `remapped` - result of remap_small_structures_to_parent (unclear)

**Fixes Needed:**
- ✅ `ref` → `slice_reference` (in loops)
- ✅ `remapped` → `remapped_label_slice`

---

### 6. **train/helpers.py**

**Issue:**
- `refs_list` - redundant suffix
- `desc_str` - abbreviated description

**Fixes Needed:**
- ✅ `refs_list` → `slice_references`
- ✅ `desc_str` → `description`

---

## 🔧 Specific Corrections Needed

### Priority 1: Critical Clarity Issues

These variables cause confusion and should be fixed:

1. `data/preprocessing.py::extract_slice()`: `s` → `slice_2d`
2. `data/datasets.py::__getitem__()`: `ref` → `slice_ref`
3. `data/datasets.py::__getitem__()`: `remapped` → `remapped_label`
4. `features/extraction.py`: `C_in` → `num_input_channels`
5. `features/extraction.py`: `cx_vals`, `cy_vals` → `centroid_x_list`, `centroid_y_list`
6. `train/helpers.py`: `refs_list` → `slice_references`, `desc_str` → `description`

### Priority 2: Moderate Issues (Keep with Better Comments)

These are acceptable but need clear comments:

1. `eval/metrics.py`: `P`, `N`, `J`, `t` - Add docstring explaining statistical notation
2. `features/extraction.py`: `probs` - Clear in context
3. `data/preprocessing.py`: `img_vol` - Clear abbreviation

### Priority 3: Standard Conventions (Keep As-Is)

These follow standard conventions:

1. `H`, `W` - Height, Width (universal in image processing)
2. `C`, `B` - Channels, Batch (standard in PyTorch)
3. `sx`, `sy` - Spacing x/y (standard in medical imaging)
4. `eps` - Epsilon (mathematical standard)
5. `thresh` - Threshold (acceptable abbreviation)

---

## ✅ Action Plan

I will now systematically fix all Priority 1 issues while ensuring:
- ✅ No variable conflicts
- ✅ No illegal overriding
- ✅ Functionality remains identical
- ✅ All references are updated consistently

---

## 📋 Files to Modify

1. `data/preprocessing.py` - Fix `s` → `slice_2d`
2. `data/datasets.py` - Fix `ref` → `slice_ref`, `remapped` → `remapped_label`
3. `features/extraction.py` - Fix `C_in` → `num_input_channels`, `cx_vals/cy_vals` → `centroid_x_list/centroid_y_list`
4. `train/helpers.py` - Fix `refs_list` → `slice_references`, `desc_str` → `description`

---

## ⚠️ Variables to Keep (Standard Notation)

The following should NOT be changed as they follow established conventions:

- `P`, `N` (Positive/Negative counts in statistics)
- `TP`, `TN`, `FP`, `FN` (True/False Positive/Negative - universal)
- `H`, `W`, `C`, `B` (Height, Width, Channels, Batch - PyTorch standard)
- `x`, `y` in mathematical contexts (coordinates)
- `t` for threshold in statistical loops
- `eps` for epsilon
- Greek letters in loss functions (alpha, gamma, etc.)

---

Now proceeding with systematic corrections...
