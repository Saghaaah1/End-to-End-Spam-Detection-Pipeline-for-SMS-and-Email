"""
Configuration for Spam NLP Pipeline.

This module contains all configuration constants used throughout
the pipeline.
"""

from pathlib import Path

# =============================================================================
# PROJECT PATHS
# =============================================================================
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_PATH = PROJECT_ROOT / "data"

# =============================================================================
# DATA CONFIGURATION
# =============================================================================

# File names
SMS_FILE = "sms_spam.csv"
EMAIL_FILE = "email_spam.csv"

# Column names 
MESSAGE_COL = "message"
LABEL_COL = "label"

# =============================================================================
# PREPROCESSING CONFIGURATION
# =============================================================================

# Train/test split
TRAIN_TEST_SPLIT_SIZE: float = 0.2
RANDOM_STATE: int = 3

# Default vocabulary size
NB_FEATURES_DEFAULT: int = 5000

# Default token pattern (can override it in experiments)
DEFAULT_TOKEN_REGEX: str = r"(\S+)"

# =============================================================================
# MODEL CONFIGURATION
# =============================================================================

# Default max iterations for Logistic Regression
NB_ITERATIONS_DEFAULT: int = 1000



