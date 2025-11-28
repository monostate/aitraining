"""
Pytest configuration for trainer tests.
Handles setup tasks like downloading required NLTK data.
"""

import nltk
import pytest


def pytest_configure(config):
    """Download required NLTK data before running tests."""
    try:
        # Download punkt_tab tokenizer data
        nltk.download("punkt_tab", quiet=True)
    except Exception:
        # If download fails, try the older punkt tokenizer
        try:
            nltk.download("punkt", quiet=True)
        except Exception:
            pass  # Continue even if downloads fail
