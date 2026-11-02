
import numpy as np
import pandas as pd
import pytest

from src.analyze_results import (
    validate_results,
    calculate_tradeoffs,
    aggregate_by_seed,
    identify_pareto_front,
    select_best_configuration
)


@pytest.fixture
def results():
    return pd.DataFrame([
        {
            "model": "Fuzzy",
            "pruning_setting": "all",
            "split": "validation",
            "seed": 42,
            "accuracy": 0.95,
            "macro_f1": 0.94,
            "fidelity": 0.98,
            "rule_count": 100,
            "mean_rule_length": 4.0,
            "coverage": 1.0,
            "fallback_rate": 0.0
        },
        {
            "model": "Fuzzy",
            "pruning_setting": "top_50",
            "split": "validation",
            "seed": 42,
            "accuracy": 0.93,
            "macro_f1": 0.92,
            "fidelity": 0.95,
            "rule_count": 50,
            "mean_rule_length": 3.0,
            "coverage": 0.96,
            "fallback_rate": 0.04
        },
        {
            "model": "Fuzzy",
            "pruning_setting": "top_25",
            "split": "validation",
            "seed": 42,
            "accuracy": 0.90,
            "macro_f1": 0.89,
            "fidelity": 0.91,
            "rule_count": 25,
            "mean_rule_length": 2.5,
            "coverage": 0.90,
            "fallback_rate": 0.10
        },
        {
            "model": "Fuzzy",
            "pruning_setting": "all",
            "split": "validation",
            "seed": 7,
            "accuracy": 0.94,
            "macro_f1": 0.93,
            "fidelity": 0.97,
            "rule_count": 100,
            "mean_rule_length": 4.0,
            "coverage": 1.0,
            "fallback_rate": 0.0
        }
    ])


def test_validate_results(results):
    validated = validate_results(results)

    assert len(validated) == 4
    assert "accuracy" in validated.columns


def test_validate_missing_column(results):
    results = results.drop(columns=["fidelity"])

    with pytest.raises(ValueError):
        validate_results(results)


def test_validate_duplicate_configuration(results):
    duplicate = pd.concat(
        [results, results.iloc[[0]]],
        ignore_index=True
    )

    with pytest.raises(ValueError):
        validate_results(duplicate)


def test_calculate_tradeoffs(results):
    output = calculate_tradeoffs(
        results,
        mlp_baseline=0.97
    )

    pruned = output[
        (output["pruning_setting"] == "top_50") &
        (output["seed"] == 42)
    ].iloc[0]

    assert pruned["rule_reduction_pct"] == pytest.approx(50.0)
    assert pruned["mean_rule_length_reduction_pct"] == pytest.approx(25.0)
    assert pruned["accuracy_loss_vs_unpruned"] == pytest.approx(0.02)
    assert pruned["coverage_deterioration"] == pytest.approx(0.04)
    assert pruned["fidelity_loss"] == pytest.approx(0.05)
    assert pruned["accuracy_loss_vs_mlp"] == pytest.approx(0.04)


def test_calculate_tradeoffs_without_mlp(results):
    output = calculate_tradeoffs(results)

    assert output["mlp_accuracy"].isna().all()
    assert output["accuracy_loss_vs_mlp"].isna().all()


def test_calculate_tradeoffs_with_seed_specific_mlp(results):
    baseline = pd.DataFrame([
        {"split": "validation", "seed": 42, "accuracy": 0.97},
        {"split": "validation", "seed": 7, "accuracy": 0.96}
    ])

    output = calculate_tradeoffs(results, baseline)

    row = output[
        (output["pruning_setting"] == "all") &
        (output["seed"] == 7)
    ].iloc[0]

    assert row["accuracy_loss_vs_mlp"] == pytest.approx(0.02)


def test_aggregate_by_seed(results):
    summary = aggregate_by_seed(results)

    row = summary[
        summary["pruning_setting"] == "all"
    ].iloc[0]

    assert row["accuracy_mean"] == pytest.approx(0.945)
    assert row["accuracy_count"] == 2
    assert row["accuracy_std"] == pytest.approx(
        np.std([0.95, 0.94], ddof=1)
    )


def test_aggregate_single_seed_std_is_nan(results):
    summary = aggregate_by_seed(results)

    row = summary[
        summary["pruning_setting"] == "top_50"
    ].iloc[0]

    assert row["accuracy_count"] == 1
    assert np.isnan(row["accuracy_std"])


def test_pareto_front(results):
    output = identify_pareto_front(results)

    assert "pareto_efficient" in output.columns

    # The top-50 configuration is dominated by the unpruned
    # configuration on all six objectives.
    row = output[
        output["pruning_setting"] == "top_50"
    ].iloc[0]

    assert row["pareto_efficient"]


def test_pareto_front_invalid_objective(results):
    with pytest.raises(ValueError):
        identify_pareto_front(
            results,
            maximize=["unknown_metric"]
        )


def test_select_best_configuration(results):
    selected = select_best_configuration(
        results,
        split="validation",
        min_accuracy=0.90,
        min_fidelity=0.90,
        min_coverage=0.90
    )

    assert selected["pruning_setting"] == "top_25"


def test_select_best_configuration_respects_constraints(results):
    selected = select_best_configuration(
        results,
        split="validation",
        min_accuracy=0.92,
        min_fidelity=0.94,
        min_coverage=0.95
    )

    assert selected["pruning_setting"] == "top_50"


def test_select_best_rejects_test_split(results):
    results["split"] = "test"

    with pytest.raises(ValueError):
        select_best_configuration(results, split="test")


def test_select_best_no_eligible_configuration(results):
    with pytest.raises(ValueError):
        select_best_configuration(
            results,
            min_accuracy=0.99
        )


def test_invalid_threshold(results):
    with pytest.raises(ValueError):
        select_best_configuration(
            results,
            min_accuracy=1.5
        )
