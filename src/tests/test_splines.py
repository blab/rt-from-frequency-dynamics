from rt_from_frequency_dynamics import Spline, SplineDeriv
import numpy as np


def test_spline():
    k = 10
    t = np.arange(100)
    T = len(t)
    s = np.linspace(0, T, k)
    X = Spline.matrix(t, s, order=3)

    # Check shape
    assert X.shape == (T, k + 1)

    # Check that multiplying all ones gives 1
    assert np.isclose(X @ np.ones(k + 1), 1.0).all()


def test_spline_deriv():
    k = 10
    t = np.arange(100)
    T = len(t)
    s = np.linspace(0, T, k)
    X = SplineDeriv.matrix(t, s, order=3)

    assert X.shape == (T, k + 1)

    # Check that multiplying all ones gives 0
    assert np.isclose(X @ np.ones(k + 1), 0.0).all()
