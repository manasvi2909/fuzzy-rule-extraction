from src.rule_types import Rule


def test_rule_creation():
    rule = Rule(
        antecedents={"feature_1": "High"},
        consequent=1,
        weight=0.8,
        support=5
    )

    assert rule.antecedents == {"feature_1": "High"}
    assert rule.consequent == 1
    assert rule.weight == 0.8
    assert rule.support == 5
