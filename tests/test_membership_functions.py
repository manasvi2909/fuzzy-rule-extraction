
import numpy as np
import pandas as pd
import pytest

from src.membership_functions import MembershipFunctionBank


@pytest.fixture
def training_data():
    return pd.DataFrame({
        "feature_1": [-2.0, -1.0, 0.0, 1.0, 2.0],
        "feature_2": [10.0, 20.0, 30.0, 40.0, 50.0]
    })


def test_fit_learns_training_ranges(training_data):
    bank = MembershipFunctionBank().fit(training_data)

    assert bank.feature_ranges["feature_1"] == {
        "Low": -2.0,
        "Medium": 0.0,
        "High": 2.0
    }


def test_low_medium_high_at_training_points(training_data):
    bank = MembershipFunctionBank().fit(training_data)

    result = bank.fuzzify_sample(training_data.iloc[2])

    assert result["feature_1"] == {
        "Low": 0.0,
        "Medium": 1.0,
        "High": 0.0
    }


def test_membership_degrees_are_between_zero_and_one(training_data):
    bank = MembershipFunctionBank().fit(training_data)

    result = bank.fuzzify_sample({
        "feature_1": -0.5,
        "feature_2": 35.0
    })

    for terms in result.values():
        for degree in terms.values():
            assert 0.0 <= degree <= 1.0


def test_membership_degrees_overlap(training_data):
    bank = MembershipFunctionBank().fit(training_data)

    result = bank.fuzzify_sample({
        "feature_1": -1.0,
        "feature_2": 20.0
    })

    assert result["feature_1"]["Low"] == 0.5
    assert result["feature_1"]["Medium"] == 0.5


def test_outside_training_range_uses_shoulders(training_data):
    bank = MembershipFunctionBank().fit(training_data)

    result = bank.fuzzify_sample({
        "feature_1": -5.0,
        "feature_2": 100.0
    })

    assert result["feature_1"]["Low"] == 1.0
    assert result["feature_2"]["High"] == 1.0


def test_constant_feature(training_data):
    X = pd.DataFrame({"constant": [5.0, 5.0, 5.0]})
    bank = MembershipFunctionBank().fit(X)

    result = bank.fuzzify_sample({"constant": 5.0})

    assert result["constant"] == {
        "Low": 0.0,
        "Medium": 1.0,
        "High": 0.0
    }


def test_fuzzify_dataset_returns_one_result_per_row(training_data):
    bank = MembershipFunctionBank().fit(training_data)

    results = bank.fuzzify_dataset(training_data)

    assert len(results) == len(training_data)
    assert len(results[0]) == 2


def test_fuzzify_before_fit_raises_error(training_data):
    bank = MembershipFunctionBank()

    with pytest.raises(RuntimeError):
        bank.fuzzify_sample(training_data.iloc[0])


def test_empty_training_data_raises_error():
    bank = MembershipFunctionBank()

    with pytest.raises(ValueError):
        bank.fit(pd.DataFrame())


def test_non_numeric_training_data_raises_error():
    bank = MembershipFunctionBank()

    X = pd.DataFrame({"feature": ["a", "b", "c"]})

    with pytest.raises(TypeError):
        bank.fit(X)


def test_non_finite_training_data_raises_error():
    bank = MembershipFunctionBank()

    X = pd.DataFrame({"feature": [1.0, np.nan, 3.0]})

    with pytest.raises(ValueError):
        bank.fit(X)


def test_wrong_feature_names_raise_error(training_data):
    bank = MembershipFunctionBank().fit(training_data)

    with pytest.raises(ValueError):
        bank.fuzzify_sample({"wrong_feature": 1.0})


def test_wrong_feature_order_raises_error(training_data):
    bank = MembershipFunctionBank().fit(training_data)

    X = training_data[["feature_2", "feature_1"]]

    with pytest.raises(ValueError):
        bank.fuzzify_dataset(X)
