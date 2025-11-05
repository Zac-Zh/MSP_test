# Contributing to MSP Detection

Thank you for your interest in contributing to the MSP Detection project! This document provides guidelines for contributing to the ongoing refactoring effort.

## 🎯 Project Status

This project is currently undergoing active refactoring from a monolithic research script (`main.py`, 9,013 lines) into a modular, maintainable codebase. See `REFACTORING_STATUS.md` for detailed progress.

## 🤝 How to Contribute

### Types of Contributions

1. **Code Refactoring** (High Priority)
   - Extract functions from `main.py` into appropriate modules
   - Improve existing refactored code
   - Add missing functionality

2. **Documentation**
   - Improve docstrings
   - Write tutorials
   - Create example notebooks
   - Expand README

3. **Testing**
   - Write unit tests for refactored modules
   - Add integration tests
   - Improve test coverage

4. **Bug Fixes**
   - Fix issues in refactored code
   - Report bugs in original `main.py`

## 📋 Getting Started

### 1. Setup Development Environment

```bash
# Fork the repository on GitHub
git clone https://github.com/your-username/msp-detection.git
cd msp-detection

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Install development dependencies
pip install pytest black flake8 mypy
```

### 2. Verify Setup

```bash
# Test imports
python3 -c "from config import get_default_config; print('✅ Setup OK')"
```

### 3. Choose a Task

Check `REFACTORING_STATUS.md` for tasks marked as:
- 🚧 **In Progress** - Partially completed
- ⏳ **Pending** - Not started yet

## 📝 Code Standards

### Python Style

Follow PEP 8 and these additional guidelines:

```python
# ✅ Good: Clear function with comprehensive docstring
def extract_slice(volume_data: np.ndarray, slice_idx: int, axis: int = 2) -> Optional[np.ndarray]:
    """
    Extracts a 2D slice from a 3D medical imaging volume.

    This operation is fundamental for processing volumetric MRI/CT data with
    2D convolutional networks. Different axes correspond to different anatomical
    planes: axis=0 (sagittal), axis=1 (coronal), axis=2 (axial).

    Args:
        volume_data: Input 3D volume array with shape (H, W, D)
        slice_idx: Zero-indexed position along the specified axis
        axis: Axis along which to extract (0, 1, or 2)

    Returns:
        2D numpy array containing the slice, or None if extraction fails
    """
    # Implementation
    pass

# ❌ Bad: No docstring, no type hints
def extract_slice(volume_data, slice_idx, axis=2):
    # Extract slice
    pass
```

### Documentation Standards

1. **Every Function Must Have**:
   - Purpose description
   - Args section with type and description
   - Returns section
   - Example usage (for complex functions)

2. **Every Module Must Have**:
   - Top-level docstring explaining purpose
   - List of key functions/classes
   - Typical usage example

3. **Avoid**:
   - Version notes ("v2.0 fixed bug")
   - Debug comments ("TODO: fix this later")
   - Personal notes ("Bob: this doesn't work")

### Naming Conventions

```python
# Functions: lowercase with underscores
def load_nifti_data():
    pass

# Classes: PascalCase
class UNetHeatmap:
    pass

# Constants: UPPERCASE
MAX_SLICE_INDEX = 256

# Private: leading underscore
def _internal_helper():
    pass
```

## 🔄 Refactoring Process

### Extracting a Function from main.py

1. **Locate the function** in `main.py`
2. **Identify dependencies** (what other functions does it call?)
3. **Determine correct module** (config, data, models, etc.)
4. **Copy and clean** the function:
   - Remove debug comments
   - Add comprehensive docstring
   - Add type hints
   - Test it works

5. **Update imports** in the target module's `__init__.py`

### Example: Extracting a Function

**Original in main.py**:
```python
def some_function(x):
    # Debug: this was fixed in v3
    result = x * 2  # multiply by 2
    return result
```

**Refactored in appropriate module**:
```python
def some_function(x: float) -> float:
    """
    Doubles the input value.

    This operation is used in the preprocessing pipeline to
    adjust intensity values.

    Args:
        x: Input value to double

    Returns:
        The input value multiplied by two

    Example:
        >>> some_function(5.0)
        10.0
    """
    return x * 2
```

## 🧪 Testing

### Writing Tests

Create tests in `tests/` directory:

```python
# tests/test_preprocessing.py
import numpy as np
from data.preprocessing import extract_slice

def test_extract_slice_axis_2():
    """Test slice extraction along axis 2."""
    volume = np.random.randn(128, 128, 100)
    slice_2d = extract_slice(volume, 50, axis=2)

    assert slice_2d is not None
    assert slice_2d.shape == (128, 128)
    assert np.allclose(slice_2d, volume[:, :, 50])

def test_extract_slice_invalid_index():
    """Test handling of invalid slice index."""
    volume = np.random.randn(128, 128, 100)
    result = extract_slice(volume, 200, axis=2)

    assert result is None
```

### Running Tests

```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_preprocessing.py

# Run with coverage
pytest --cov=. tests/
```

## 📖 Documentation

### Adding Documentation

1. **Function Docstrings**: Follow Google style
2. **Module Docstrings**: Explain module purpose
3. **README Updates**: Keep main README current
4. **Example Notebooks**: Create in `notebooks/`

### Documentation Example

```python
"""
data/preprocessing.py

Image preprocessing utilities for MRI data.

This module provides functions for:
- Slice extraction from 3D volumes
- Intensity normalization
- Brain mask generation
- Heatmap target creation

Typical usage:
    from data.preprocessing import extract_slice, normalize_slice

    volume = load_nifti_data("scan.nii.gz")
    slice_2d = extract_slice(volume, 100, axis=2)
    normalized = normalize_slice(slice_2d, config)
"""
```

## 🔀 Git Workflow

### Branch Naming

- `feature/add-datasets` - New features
- `refactor/extract-training` - Refactoring work
- `fix/normalization-bug` - Bug fixes
- `docs/improve-readme` - Documentation

### Commit Messages

```bash
# ✅ Good commits
git commit -m "refactor: extract dataset classes to data/datasets.py"
git commit -m "feat: add comprehensive docstrings to preprocessing module"
git commit -m "fix: correct slice extraction for axis=0"
git commit -m "docs: update README with installation instructions"

# ❌ Bad commits
git commit -m "fixed stuff"
git commit -m "wip"
git commit -m "asdf"
```

### Pull Request Process

1. **Create Branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make Changes**
   - Write code following standards
   - Add tests
   - Update documentation

3. **Test Locally**
   ```bash
   pytest tests/
   flake8 .
   black --check .
   ```

4. **Commit and Push**
   ```bash
   git add .
   git commit -m "feat: your descriptive message"
   git push origin feature/your-feature-name
   ```

5. **Open Pull Request**
   - Reference related issues
   - Describe changes clearly
   - Include before/after examples if applicable

## 🎯 Specific Contribution Areas

### Easy (Good for First Contribution)

- [ ] Add docstrings to functions missing them
- [ ] Write unit tests for `data/preprocessing.py`
- [ ] Improve README examples
- [ ] Fix typos in documentation
- [ ] Add type hints to functions

### Medium

- [ ] Extract dataset classes from `main.py`
- [ ] Refactor training loop functions
- [ ] Create visualization utilities
- [ ] Write integration tests
- [ ] Create example Jupyter notebook

### Advanced

- [ ] Complete training module refactoring
- [ ] Implement full inference pipeline
- [ ] Optimize data loading performance
- [ ] Add continuous integration (CI/CD)
- [ ] Create Docker container

## 📊 Code Review Checklist

Before submitting, ensure:

- [ ] Code follows PEP 8
- [ ] All functions have comprehensive docstrings
- [ ] Type hints added
- [ ] Tests written and passing
- [ ] No debug comments or version notes
- [ ] `__init__.py` updated with new exports
- [ ] README updated if needed
- [ ] No circular dependencies introduced

## ❓ Questions?

- Check `REFACTORING_STATUS.md` for progress
- See `README.md` for project overview
- Open an issue for questions

## 📄 License

By contributing, you agree that your contributions will be licensed under the MIT License.

---

**Thank you for contributing to MSP Detection!** 🎉
