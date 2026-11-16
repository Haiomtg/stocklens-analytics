import numpy as np
import pandas as pd
import pytest

from stocklens import db


@pytest.fixture
def conn():
    c = db.connect(":memory:")
    yield c
    c.close()


@pytest.fixture
def walk():
    """A reproducible random-walk price series (1500 business days)."""
    rng = np.random.default_rng(1)
    idx = pd.bdate_range("2019-01-01", periods=1500)
    return pd.Series(100 * np.exp(np.cumsum(rng.normal(0.0003, 0.012, 1500))), index=idx, name="X")
