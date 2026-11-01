import pytest

from src.evaluate import evaluate_metrics
from src.rule_types import Rule


@pytest.fixture
def sample_rules():
    return [
        Rule({"f1": "Low", "f2": "High"}, 0, 0.8, 5),
        Rule({"f1": "High"}, 1, 0.9, 8)
    ]


def test_evaluation_metrics(sample_rules):
    y_true = [0, 1, 0, 1]
    fuzzy = [0, 1, 1, 1]
    mlp = [0, 1, 0, 1]

    covered = [True, True, True, False]

    metrics = evaluate_metrics(
        y_true,
        fuzzy,
        mlp,
        sample_rules,
        covered=covered
    )

    assert metrics["accuracy"] == pytest.approx(0.75)
    assert metrics["macro_f1"] == pytest.approx(0.7333333)
    assert metrics["fidelity"] == pytest.approx(0.75)
    assert metrics["rule_count"] == 2
    assert metrics["mean_rule_length"] == pytest.approx(1.5)
    assert metrics["coverage"] == pytest.approx(0.75)
    assert metrics["fallback_rate"] == pytest.approx(0.25)


def test_fallback_only(sample_rules):
    fallback = [False, False, True, True]

    metrics = evaluate_metrics(
        [0, 1, 0, 1],
        [0, 1, 0, 1],
        [0, 1, 0, 1],
        sample_rules,
        fallback=fallback
    )

    assert metrics["coverage"] == pytest.approx(0.5)
    assert metrics["fallback_rate"] == pytest.approx(0.5)


def test_empty_rule_base():
    metrics = evaluate_metrics(
        [0, 1],
        [0, 1],
        [0, 1],
        []
    )

    assert metrics["rule_count"] == 0
    assert metrics["mean_rule_length"] == 0.0


def test_mismatched_prediction_lengths(sample_rules):
    with pytest.raises(ValueError):
        evaluate_metrics(
            [0, 1],
            [0],
            [0, 1],
            sample_rules
        )


def test_empty_evaluation_data(sample_rules):
    with pytest.raises(ValueError):
        evaluate_metrics([], [], [], sample_rules)


def test_invalid_coverage_flags(sample_rules):
    with pytest.raises(ValueError):
        evaluate_metrics(
            [0, 1],
            [0, 1],
            [0, 1],
            sample_rules,
            covered=[True, False],
            fallback=[True, False]
        )