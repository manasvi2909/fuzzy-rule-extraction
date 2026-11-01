from src.rule_types import Rule


def rank_rules(rules):
    """Return rules sorted by importance."""
    return sorted(
        rules,
        key=lambda r: (
            -r.weight,
            -r.support,
            repr(sorted(r.antecedents.items()))
        )
    )


def prune_rules(rules, top_n):
    """
    Retain the top N rules based on weight and support.

    Parameters:
        rules: List of Rule objects.
        top_n: Maximum number of rules to retain.

    Returns:
        Pruned list of Rule objects.
    """
    if not isinstance(top_n, int) or isinstance(top_n, bool):
        raise TypeError("top_n must be an integer.")

    if top_n < 0:
        raise ValueError("top_n cannot be negative.")

    if any(not isinstance(rule, Rule) for rule in rules):
        raise TypeError("All elements must be Rule objects.")

    ranked = rank_rules(rules)

    return ranked[:top_n]


def prune_multiple(rules, top_n_values):
    """
    Generate multiple pruned rule bases for comparison.

    Example:
        {10: [...], 25: [...], 50: [...]}
    """
    return {
        n: prune_rules(rules, n)
        for n in top_n_values
    }