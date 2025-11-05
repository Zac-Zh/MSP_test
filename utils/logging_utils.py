"""
Logging utilities for experiment tracking and output management.

Provides functions for structured logging, directory creation, and
experiment organization with timestamped output directories.
"""

import os
import datetime
from pathlib import Path
from typing import Optional, Dict


def log_message(message: str, log_file: Optional[str] = None) -> None:
    """
    Logs a message to console and optionally to a file.

    This function provides dual-output logging, printing to stdout for
    real-time monitoring and appending to a log file for persistent records.

    Args:
        message: Text message to log
        log_file: Optional path to log file for persistent storage

    Example:
        >>> log_message("Training started", "logs/experiment.log")
        >>> log_message("Epoch 1 complete")  # Console only
    """
    print(message)
    if log_file:
        try:
            with open(log_file, 'a', encoding='utf-8') as f:
                f.write(f"{message}\n")
        except Exception:
            pass


def create_dir_with_permissions(path_obj: Path, mode: int = 0o755) -> Path:
    """
    Creates a directory with specific permissions.

    Args:
        path_obj: Path object for directory to create
        mode: Unix permissions (default: 0o755 = rwxr-xr-x)

    Returns:
        Path object of created directory

    Raises:
        FileExistsError: If path exists but is not a directory
    """
    if not path_obj.exists():
        path_obj.mkdir(parents=True, exist_ok=False)
        try:
            os.chmod(path_obj, mode)
        except Exception as e:
            print(f"Warning: Could not chmod {path_obj} to {oct(mode)}: {e}")
    elif not path_obj.is_dir():
        raise FileExistsError(f"Path {path_obj} exists but is not a directory.")
    return path_obj


def setup_logging(output_dir_base: str) -> Dict[str, Path]:
    """
    Sets up logging directory structure with timestamped run folders.

    Creates a hierarchical output structure for each experimental run:
    - output_dir/run_YYYYMMDD-HHMMSS/
      - checkpoints/
      - visualizations/
      - run_log.txt

    This organization enables easy comparison between runs and prevents
    accidental overwriting of previous results.

    Args:
        output_dir_base: Base directory for all experimental outputs

    Returns:
        Dictionary containing paths:
            - 'run_dir': Root directory for this run
            - 'log_file': Path to log file (str)
            - 'checkpoint_dir': Directory for model checkpoints
            - 'viz_dir': Directory for visualizations

    Example:
        >>> paths = setup_logging("/experiments/msp_detection")
        >>> print(paths['run_dir'])
        /experiments/msp_detection/run_20241031-143022
    """
    current_timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    current_run_dir_name = f"run_{current_timestamp}"
    run_dir_path = Path(output_dir_base) / current_run_dir_name

    # Create base output directory
    Path(output_dir_base).mkdir(parents=True, exist_ok=True)

    # Create run-specific directories
    run_dir = create_dir_with_permissions(run_dir_path)
    checkpoints_dir = create_dir_with_permissions(run_dir / "checkpoints")
    visualizations_dir = create_dir_with_permissions(run_dir / "visualizations")

    log_file_path = run_dir / "run_log.txt"

    # Initial log messages
    log_message(f"✅ Logging initialized. Run Directory: {run_dir}")
    log_message(f"  Log file: {log_file_path}")
    log_message(f"  Checkpoints: {checkpoints_dir}")
    log_message(f"  Visualizations: {visualizations_dir}")

    return {
        "run_dir": run_dir,
        "log_file": str(log_file_path),
        "checkpoint_dir": checkpoints_dir,
        "viz_dir": visualizations_dir
    }
