import pytest

from src.extract_rules import (
    generate_candidate_rule,
    resolve_conflicts,
    extract_rules
)


def test_candidate_rule_generation():
    memberships = {
        "radius": {"Low": 0.1, "Medium": 0.8, "High": 0.1},
        "texture": {"Low": 0.2, "Medium": 0.3, "High": 0.5}
    }

    rule = generate_candidate_rule(memberships, 1)

    assert rule.antecedents == {
        "radius": "Medium",
        "texture": "High"
    }
    assert rule.consequent == 1
    assert rule.weight == pytest.approx(0.4)
    assert rule.support == 1


def test_conflict_resolution():
    memberships = {
        "radius": {"Low": 0.1, "Medium": 0.8, "High": 0.1}
    }

    r1 = generate_candidate_rule(memberships, 1)
    r2 = generate_candidate_rule(memberships, 1)
    r3 = generate_candidate_rule(memberships, 0)

    rules = resolve_conflicts([r1, r2, r3])

    assert len(rules) == 1
    assert rules[0].consequent == 1
    assert rules[0].support == 2
    assert rules[0].weight == pytest.approx(1.6)


def test_extract_rules():
    samples = [
        {"f1": {"Low": 0.9, "High": 0.1}},
        {"f1": {"Low": 0.8, "High": 0.2}},
        {"f1": {"Low": 0.1, "High": 0.9}}
    ]

    predictions = [0, 0, 1]

    rules = extract_rules(samples, predictions)

    assert len(rules) == 2
    assert rules[0].consequent == 0
    assert rules[0].support == 2


def test_invalid_membership_degree():
    memberships = {
        "f1": {"Low": 1.2, "High": 0.1}
    }

    with pytest.raises(ValueError):
        generate_candidate_rule(memberships, 1)


def test_mismatched_input_lengths():
    with pytest.raises(ValueError):
        extract_rules([{}], [])