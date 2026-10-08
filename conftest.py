# Its presence puts the project folder on sys.path for pytest.
import os
from pathlib import Path

import pytest

# Set before tests import app: a known key, and a limit high enough not to trip the other tests.
os.environ["API_KEY"] = "test-key"
os.environ["RATE_LIMIT"] = "1000/minute"


def pytest_configure(config):
    if not (Path(__file__).parent / "model.joblib").exists():
        raise pytest.UsageError("model.joblib not found. Run train.py first: the API tests need the model.")
