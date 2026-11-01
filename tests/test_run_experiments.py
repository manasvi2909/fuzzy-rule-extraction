import pytest

from src.rule_types import Rule
from src.run_experiments import run_rule_experiments


@pytest.fixture
def sample_rules():
    return [
        Rule({"f1": "Low"}, 0, 0.9, 10),
        Rule({"f1": "High"}, 1, 0.8, 8),
        Rule({"f2": "Medium"}, 0, 0.7, 6)
    ]


def dummy_inference(X, rules):
    """
    Temporary inference function for testing orchestration.
    It is NOT the project's actual Mamdani inference engine.
    """
    predictions = [0, 1, 0, 1]
    covered = [True, True, True, False]
    fallback = [False, False, False, True]

    return predictions, covered, fallback


def test_experiment_configurations(sample_rules):
    X = [[1], [2], [3], [4]]
    y_true = [0, 1, 0, 1]
    mlp_predictions = [0, 1, 1, 1]

    results = run_rule_experiments(
        X,
        y_true,
        mlp_predictions,
        sample_rules,
        dummy_inference,
        top_n_values=(1, 2)
    )

    assert len(results) == 3
    assert results[0]["pruning_setting"] == "all"
    assert results[1]["pruning_setting"] == "top_1"
    assert results[2]["pruning_setting"] == "top_2"


def test_experiment_metrics(sample_rules):
    X = [[1], [2], [3], [4]]

    results = run_rule_experiments(
        X,
        [0, 1, 0, 1],
        [0, 1, 1, 1],
        sample_rules,
        dummy_inference,
        top_n_values=(2,)
    )

    metrics = results[0]

    assert metrics["accuracy"] == pytest.approx(1.0)
    assert metrics["fidelity"] == pytest.approx(0.75)
    assert metrics["coverage"] == pytest.approx(0.75)
    assert metrics["fallback_rate"] == pytest.approx(0.25)


def test_duplicate_configurations_are_removed(sample_rules):
    X = [[1], [2], [3], [4]]

    results = run_rule_experiments(
        X,
        [0, 1, 0, 1],
        [0, 1, 0, 1],
        sample_rules,
        dummy_inference,
        top_n_values=(10, 25, 50)
    )

    # All top-N configurations retain the same three rules.
    assert len(results) == 1


def test_invalid_inference_output(sample_rules):
    def invalid_inference(X, rules):
        return [0, 1]

    with pytest.raises(ValueError):
        run_rule_experiments(
            [[1], [2]],
            [0, 1],
            [0, 1],
            sample_rules,
            invalid_inference
        )


def test_mismatched_input_lengths(sample_rules):
    with pytest.raises(ValueError):
        run_rule_experiments(
            [[1], [2]],
            [0],
            [0, 1],
            sample_rules,
            dummy_inference
        )