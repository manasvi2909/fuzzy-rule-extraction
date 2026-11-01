import pytest

from src.rule_types import Rule
from src.prune_rules import rank_rules, prune_rules, prune_multiple


@pytest.fixture
def sample_rules():
    return [
        Rule({"f1": "Low"}, 0, 0.4, 5),
        Rule({"f1": "High"}, 1, 0.9, 3),
        Rule({"f2": "Medium"}, 0, 0.7, 8),
        Rule({"f2": "Low"}, 1, 0.9, 6)
    ]


def test_rule_ranking(sample_rules):
    ranked = rank_rules(sample_rules)

    assert ranked[0].weight == 0.9
    assert ranked[0].support == 6


def test_top_n_pruning(sample_rules):
    pruned = prune_rules(sample_rules, 2)

    assert len(pruned) == 2
    assert pruned[0].weight == 0.9
    assert pruned[0].support == 6
    assert pruned[1].weight == 0.9
    assert pruned[1].support == 3


def test_pruning_all_rules(sample_rules):
    pruned = prune_rules(sample_rules, 10)

    assert len(pruned) == 4


def test_pruning_zero_rules(sample_rules):
    assert prune_rules(sample_rules, 0) == []


def test_invalid_top_n(sample_rules):
    with pytest.raises(ValueError):
        prune_rules(sample_rules, -1)

    with pytest.raises(TypeError):
        prune_rules(sample_rules, 2.5)


def test_multiple_pruning_settings(sample_rules):
    results = prune_multiple(sample_rules, [1, 2, 3])

    assert len(results[1]) == 1
    assert len(results[2]) == 2
    assert len(results[3]) == 3