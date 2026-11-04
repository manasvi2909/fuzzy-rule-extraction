import numpy as np
import pandas as pd
import pytest

from src.fuzzy_inference import (
    infer_sample,
    predict_dataset,
    make_inference_function,
)
from src.rule_types import Rule


@pytest.fixture
def membership_functions():
    return {
        "f1": {
            "Low": (0.0, 0.0, 5.0),
            "Medium": (0.0, 5.0, 10.0),
            "High": (5.0, 10.0, 10.0),
        },
        "f2": {
            "Low": (0.0, 0.0, 5.0),
            "Medium": (0.0, 5.0, 10.0),
            "High": (5.0, 10.0, 10.0),
        },
    }


def test_single_rule_predicts_its_consequent(membership_functions):
    rules = [Rule({"f1": "Medium"}, 1, 0.8, 1)]
    result = infer_sample(
        {"f1": 5.0, "f2": 5.0}, rules, membership_functions
    )

    assert result["prediction"] == 1
    assert result["covered"] is True
    assert result["fallback"] is False
    assert result["class_scores"][1] == pytest.approx(0.8)


def test_and_uses_minimum_t_norm(membership_functions):
    rules = [Rule({"f1": "Low", "f2": "Medium"}, 1, 1.0, 1)]
    result = infer_sample(
        {"f1": 2.5, "f2": 5.0}, rules, membership_functions
    )

    # Low(f1=2.5) = 0.5; Medium(f2=5) = 1.0; min = 0.5.
    assert result["class_scores"][1] == pytest.approx(0.5)


def test_rule_weight_is_normalized_by_support(membership_functions):
    rules = [Rule({"f1": "Medium"}, 1, 1.6, 2)]
    result = infer_sample(
        {"f1": 5.0, "f2": 5.0}, rules, membership_functions
    )

    assert result["class_scores"][1] == pytest.approx(0.8)


def test_max_aggregation_keeps_strongest_rule_per_class(membership_functions):
    rules = [
        Rule({"f1": "Medium"}, 0, 0.4, 1),
        Rule({"f2": "Medium"}, 0, 0.9, 1),
        Rule({"f1": "Medium"}, 1, 0.7, 1),
    ]
    result = infer_sample(
        {"f1": 5.0, "f2": 5.0}, rules, membership_functions
    )

    assert result["class_scores"][0] == pytest.approx(0.9)
    assert result["class_scores"][1] == pytest.approx(0.7)
    assert result["prediction"] == 0


def test_no_matching_rule_uses_fallback(membership_functions):
    rules = [Rule({"f1": "Low"}, 1, 1.0, 1)]
    result = infer_sample(
        {"f1": 10.0, "f2": 5.0}, rules, membership_functions,
        fallback_class=0
    )

    assert result["prediction"] == 0
    assert result["covered"] is False
    assert result["fallback"] is True


def test_tie_uses_class_order(membership_functions):
    rules = [
        Rule({"f1": "Medium"}, 1, 0.8, 1),
        Rule({"f2": "Medium"}, 0, 0.8, 1),
    ]
    result = infer_sample(
        {"f1": 5.0, "f2": 5.0}, rules, membership_functions,
        class_order=(0, 1)
    )

    assert result["prediction"] == 0


def test_empty_rule_base_uses_fallback(membership_functions):
    result = infer_sample(
        {"f1": 5.0, "f2": 5.0}, [], membership_functions
    )
    assert result["prediction"] == 0
    assert result["fallback"] is True


def test_predict_dataset_returns_experiment_runner_tuple(membership_functions):
    X = pd.DataFrame({"f1": [5.0, 0.0], "f2": [5.0, 5.0]})
    rules = [Rule({"f1": "Medium"}, 1, 0.8, 1)]

    predictions, covered, fallback = predict_dataset(
        X, rules, membership_functions
    )

    assert predictions.tolist() == [1, 0]
    assert covered.tolist() == [True, False]
    assert fallback.tolist() == [False, True]


def test_make_inference_function_has_expected_signature(membership_functions):
    X = pd.DataFrame({"f1": [5.0], "f2": [5.0]})
    rules = [Rule({"f1": "Medium"}, 1, 0.8, 1)]
    inference_fn = make_inference_function(membership_functions)

    predictions, covered, fallback = inference_fn(X, rules)
    assert predictions.tolist() == [1]
    assert covered.tolist() == [True]
    assert fallback.tolist() == [False]


def test_invalid_rule_type_is_rejected(membership_functions):
    with pytest.raises(TypeError):
        infer_sample(
            {"f1": 5.0, "f2": 5.0}, [{"antecedents": {}}],
            membership_functions
        )
