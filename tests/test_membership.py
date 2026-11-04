import numpy as np
import pandas as pd
import pytest

from src.membership import (
    triangular_membership,
    trapezoidal_membership,
    fit_membership_functions,
    fuzzify_sample,
    fuzzify_dataframe,
)


def test_triangular_membership_peak_and_sides():
    assert triangular_membership(5, 0, 5, 10) == pytest.approx(1.0)
    assert triangular_membership(2.5, 0, 5, 10) == pytest.approx(0.5)
    assert triangular_membership(7.5, 0, 5, 10) == pytest.approx(0.5)
    assert triangular_membership(-1, 0, 5, 10) == 0.0
    assert triangular_membership(10, 0, 5, 10) == 0.0


def test_triangular_left_and_right_shoulders():
    assert triangular_membership(-5, 0, 0, 10) == 1.0
    assert triangular_membership(5, 0, 0, 10) == pytest.approx(0.5)
    assert triangular_membership(15, 0, 10, 10) == 1.0
    assert triangular_membership(5, 0, 10, 10) == pytest.approx(0.5)


def test_triangular_rejects_invalid_parameters():
    with pytest.raises(ValueError):
        triangular_membership(2, 5, 3, 10)
    with pytest.raises(ValueError):
        triangular_membership(float("nan"), 0, 5, 10)


def test_trapezoidal_membership():
    assert trapezoidal_membership(0, 0, 2, 4, 6) == 0.0
    assert trapezoidal_membership(1, 0, 2, 4, 6) == pytest.approx(0.5)
    assert trapezoidal_membership(3, 0, 2, 4, 6) == 1.0
    assert trapezoidal_membership(5, 0, 2, 4, 6) == pytest.approx(0.5)
    assert trapezoidal_membership(7, 0, 2, 4, 6) == 0.0


def test_fit_membership_functions_uses_training_ranges():
    X = pd.DataFrame({"f1": [0.0, 5.0, 10.0], "f2": [2.0, 4.0, 6.0]})
    mfs = fit_membership_functions(X)

    assert mfs["f1"]["Low"] == (0.0, 0.0, 5.0)
    assert mfs["f1"]["Medium"] == (0.0, 5.0, 10.0)
    assert mfs["f1"]["High"] == (5.0, 10.0, 10.0)


def test_fuzzify_sample_matches_expected_structure():
    mfs = {"f1": {
        "Low": (0.0, 0.0, 5.0),
        "Medium": (0.0, 5.0, 10.0),
        "High": (5.0, 10.0, 10.0),
    }}
    degrees = fuzzify_sample({"f1": 5.0}, mfs)

    assert degrees["f1"]["Low"] == 0.0
    assert degrees["f1"]["Medium"] == 1.0
    assert degrees["f1"]["High"] == 0.0


def test_fuzzify_dataframe():
    X = pd.DataFrame({"f1": [0.0, 5.0, 10.0]})
    mfs = fit_membership_functions(X)
    fuzzified = fuzzify_dataframe(X, mfs)

    assert len(fuzzified) == 3
    assert fuzzified[1]["f1"]["Medium"] == 1.0


def test_fit_rejects_non_dataframe_and_empty_data():
    with pytest.raises(TypeError):
        fit_membership_functions([[1, 2]])
    with pytest.raises(ValueError):
        fit_membership_functions(pd.DataFrame())
