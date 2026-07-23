import numpy as np
import pandas as pd
import pytest

from src.metrics import realized_volatility, rmspe


def test_rmspe_is_zero_for_exact_prediction():
    actual = np.array([0.1, 0.2, 0.3])
    assert rmspe(actual, actual) == 0.0


def test_rmspe_matches_manual_percentage_error():
    actual = np.array([1.0, 2.0])
    predicted = np.array([1.1, 1.8])
    assert rmspe(actual, predicted) == pytest.approx(0.1)


def test_realized_volatility():
    returns = pd.Series([0.03, 0.04, np.nan])
    assert realized_volatility(returns) == pytest.approx(0.05)
