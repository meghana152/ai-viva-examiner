"""
Helper Utilities for AI Viva Examiner.

This module will contain general utility functions such as:
- Data validation and file path resolution
- Streamlit session state helpers
- Result formatting and report generation helpers
"""

import os
from typing import Any, Dict


def check_data_files_exist(required_paths: list) -> Dict[str, bool]:
    """
    Check if required dataset and model artifact files exist on disk.
    """
    return {path: os.path.exists(path) for path in required_paths}


def format_score_display(score: float) -> str:
    """
    Format a floating-point score as a clean percentage or scale display string.
    """
    return f"{score * 100:.1f}%"
