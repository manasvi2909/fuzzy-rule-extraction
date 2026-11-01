
import numpy as np
import pandas as pd
import pytest

from src.rule_types import Rule
from src.report_tables import (
    build_comparison_table,
    build_rule_table,
    export_comparison_table,
    export_rule_table
)


@pytest.fixture
def results():
    return pd.DataFrame([
        {
            "model": "Fuzzy",
            "pruning_setting": "all",
            "split": "test",
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
            "split": "test",
            "seed": 42,
            "accuracy": 0.93,
            "macro_f1": 0.92,
            "fidelity": 0.95,
            "rule_count": 50,
            "mean_rule_length": 3.0,
            "coverage": 0.96,
            "fallback_rate": 0.04
        }
    ])


@pytest.fixture
def rules():
    return [
        Rule(
            antecedents={
                "radius": "High",
                "texture": "Low"
            },
            consequent=1,
            weight=1.25,
            support=5
        ),
        Rule(
            antecedents={
                "radius": "Low"
            },
            consequent=0,
            weight=0.75,
            support=3
        )
    ]


def test_build_comparison_table(results):
    table = build_comparison_table(results, split="test")

    assert len(table) == 2
    assert "accuracy_mean" in table.columns
    assert "fidelity_mean" in table.columns
    assert "rule_count_mean" in table.columns


def test_comparison_table_with_mlp_baseline(results):
    table = build_comparison_table(
        results,
        split="test",
        mlp_baseline=0.97
    )

    row = table[
        table["pruning_setting"] == "top_50"
    ].iloc[0]

    assert row["accuracy_loss_vs_mlp"] == pytest.approx(0.04)


def test_comparison_table_invalid_split(results):
    with pytest.raises(ValueError):
        build_comparison_table(results, split="validation")


def test_build_rule_table(rules):
    table = build_rule_table(rules)

    assert len(table) == 2
    assert table.iloc[0]["rule_number"] == 1
    assert table.iloc[0]["rule_length"] == 2
    assert table.iloc[0]["consequent"] == "1"
    assert table.iloc[0]["support"] == 5
    assert table.iloc[0]["weight"] == pytest.approx(1.25)


def test_rule_antecedents_are_sorted(rules):
    table = build_rule_table(rules)

    assert table.iloc[0]["antecedents"] == (
        "radius IS High AND texture IS Low"
    )


def test_rule_table_limit(rules):
    table = build_rule_table(rules, max_rules=1)

    assert len(table) == 1


def test_rule_table_empty():
    table = build_rule_table([])

    assert table.empty
    assert list(table.columns) == [
        "rule_number",
        "antecedents",
        "consequent",
        "weight",
        "support",
        "rule_length"
    ]


def test_rule_table_invalid_limit(rules):
    with pytest.raises(ValueError):
        build_rule_table(rules, max_rules=-1)


def test_rule_table_invalid_type():
    with pytest.raises(TypeError):
        build_rule_table(["not a rule"])


def test_rule_table_invalid_antecedents():
    rule = Rule(
        antecedents="radius=High",
        consequent=1,
        weight=0.5,
        support=2
    )

    with pytest.raises(TypeError):
        build_rule_table([rule])


def test_rule_table_invalid_weight():
    rule = Rule(
        antecedents={"radius": "High"},
        consequent=1,
        weight=float("nan"),
        support=2
    )

    with pytest.raises(ValueError):
        build_rule_table([rule])


def test_export_comparison_table(results, tmp_path):
    table, paths = export_comparison_table(
        results,
        output_dir=tmp_path,
        split="test"
    )

    assert len(table) == 2

    for path in paths.values():
        assert path.exists()
        assert path.stat().st_size > 0


def test_export_rule_table(rules, tmp_path):
    table, paths = export_rule_table(
        rules,
        output_dir=tmp_path
    )

    assert len(table) == 2

    for path in paths.values():
        assert path.exists()
        assert path.stat().st_size > 0
