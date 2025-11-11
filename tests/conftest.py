"""Shared pytest fixtures for all tests."""

import pytest
import numpy as np
import torch
import tempfile
import shutil
from pathlib import Path

# Auto-use fixture to set seeds before each test
@pytest.fixture(autouse=True)
def reset_random_state():
    """Reset random state before each test."""
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)


@pytest.fixture(scope="session")
def temp_data_dir():
    """Session-scoped temporary directory for test data."""
    tmpdir = tempfile.mkdtemp()
    yield Path(tmpdir)
    shutil.rmtree(tmpdir)


@pytest.fixture
def small_model_config():
    """Configuration for small test models."""
    return {
        "d_model": 32,
        "nhead": 2,
        "layers": 1,
        "ff_dim": 64,
        "drop": 0.1
    }


@pytest.fixture
def device():
    """Get appropriate device for testing."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")