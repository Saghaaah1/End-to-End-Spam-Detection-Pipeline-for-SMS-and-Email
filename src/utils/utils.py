"""
Utility functions for Spam NLP Pipeline.

This module contains helper functions used across the pipeline:
- data file validation
- simple step headers
- time formatting

You can later extend it with a more advanced logger if you want
(similar to the Air Quality project).
"""

import time
from pathlib import Path
from typing import Dict, Any

from utils.config import DATA_PATH, SMS_FILE, EMAIL_FILE


def validate_data_files() -> bool:
    """Check if required data files exist in the data/ directory."""
    sms_path = DATA_PATH / SMS_FILE
    email_path = DATA_PATH / EMAIL_FILE

    missing = []
    if not sms_path.exists():
        missing.append(str(sms_path))
    if not email_path.exists():
        missing.append(str(email_path))

    if missing:
        msg = "Missing data file(s):\n" + "\n".join(missing)
        raise FileNotFoundError(msg)

    print(f" Data files found in {DATA_PATH}")
    return True


def format_time_elapsed(start_time: float, end_time: float) -> str:
    """Format elapsed time in a readable way."""
    elapsed = end_time - start_time
    if elapsed < 60:
        return f"{elapsed:.1f} seconds"
    else:
        minutes = int(elapsed // 60)
        seconds = elapsed % 60
        return f"{minutes}m {seconds:.1f}s"


def print_step_header(step_number: int, step_name: str):
    """Print a formatted step header (simple version)."""
    print(f"\n STEP {step_number}: {step_name}")


def print_results_summary(results: Dict[str, Any]):
    """Print a formatted summary of evaluation results."""
    print("\n RESULTS SUMMARY")
    for key, value in results.items():
        if isinstance(value, float):
            print(f" - {key}: {value:.3f}")
        else:
            print(f" - {key}: {value}")
