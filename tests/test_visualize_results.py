import pandas as pd
import pytest

from src.visualize_results import (
    load_results,
    aggregate_results,
    generate_report_figures
)


@pytest.fixture
def sample_results(tmp_path):
    path = tmp_path / "metrics.csv"

    df = pd.DataFrame([
        {
            "model": "Fuzzy",
            "pruning_setting": "top_10",
            "split": "test",
            "seed": 42,
            "accuracy": 0.90,
            "macro_f1": 0.89,
            "fidelity": 0.92,
            "rule_count": 10,
            "mean_rule_length": 3.0,
            "coverage": 0.95,
            "fallback_rate": 0.05
        },
        {
            "model": "Fuzzy",
            "pruning_setting": "top_10",
            "split": "test",
            "seed": 43,
            "accuracy": 0.92,
            "macro_f1": 0.91,
            "fidelity": 0.94,
            "rule_count": 10,
            "mean_rule_length": 3.0,
            "coverage": 0.96,
            "fallback_rate": 0.04
        },
        {
            "model": "Fuzzy",
            "pruning_setting": "all",
            "split": "test",
            "seed": 42,
            "accuracy": 0.94,
            "macro_f1": 0.93,
            "fidelity": 0.97,
            "rule_count": 30,
            "mean_rule_length": 4.0,
            "coverage": 0.99,
            "fallback_rate": 0.01
        }
    ])

    df.to_csv(path, index=False)
    return path


def test_load_results(sample_results):
    df = load_results(sample_results)

    assert len(df) == 3
    assert "accuracy" in df.columns


def test_aggregate_results(sample_results):
    df = load_results(sample_results)
    summary = aggregate_results(df)

    row = summary[
        (summary["pruning_setting"] == "top_10")
    ].iloc[0]

    assert row["accuracy_mean"] == pytest.approx(0.91)
    assert row["accuracy_count"] == 2
    assert row["rule_count"] == 10


def test_generate_report_figures(sample_results, tmp_path):
    output_dir = tmp_path / "figures"

    generate_report_figures(
        metrics_path=sample_results,
        output_dir=output_dir,
        split="test"
    )

    assert (output_dir / "01_performance_vs_rule_count.png").exists()
    assert (output_dir / "01b_macro_f1_vs_rule_count.png").exists()
    assert (output_dir / "02_fidelity_vs_rule_count.png").exists()
    assert (output_dir / "03_coverage_fallback_vs_rule_count.png").exists()
    assert (output_dir / "04_accuracy_fidelity_tradeoff.png").exists()
    assert (tmp_path / "metrics_summary.csv").exists()


def test_invalid_metric_range(sample_results):
    df = pd.read_csv(sample_results)
    df.loc[0, "accuracy"] = 1.5
    df.to_csv(sample_results, index=False)

    with pytest.raises(ValueError):
        load_results(sample_results)


def test_duplicate_experiment_rows(sample_results):
    df = pd.read_csv(sample_results)
    df = pd.concat([df, df.iloc[[0]]], ignore_index=True)
    df.to_csv(sample_results, index=False)

    with pytest.raises(ValueError):
        load_results(sample_results)