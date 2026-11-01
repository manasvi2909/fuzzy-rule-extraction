from collections import defaultdict
from math import prod, isfinite

from src.rule_types import Rule


def generate_candidate_rule(membership_degrees, consequent):
    """
    Generate one fuzzy rule from a sample's membership degrees.

    membership_degrees format:
    {
        "feature_1": {"Low": 0.1, "Medium": 0.8, "High": 0.1},
        "feature_2": {"Low": 0.2, "Medium": 0.3, "High": 0.5}
    }
    """
    if not membership_degrees:
        raise ValueError("Membership degrees cannot be empty.")

    antecedents = {}
    strengths = []

    for feature, terms in membership_degrees.items():
        if not terms:
            raise ValueError(f"No membership terms for {feature}.")

        for term, degree in terms.items():
            if not isinstance(degree, (int, float)):
                raise ValueError("Membership degrees must be numeric.")
            if not isfinite(degree) or not 0 <= degree <= 1:
                raise ValueError("Membership degrees must be between 0 and 1.")

        # Deterministic tie-breaking: alphabetically smallest term wins.
        best_term = min(
            terms,
            key=lambda term: (-terms[term], str(term))
        )

        antecedents[feature] = best_term
        strengths.append(terms[best_term])

    return Rule(
        antecedents=antecedents,
        consequent=consequent,
        weight=prod(strengths),
        support=1
    )


def resolve_conflicts(candidate_rules):
    """
    Merge identical rules and resolve conflicting consequents.

    For identical antecedents, retain the consequent with the
    highest sum of candidate rule weights.
    """
    groups = defaultdict(lambda: defaultdict(
        lambda: {"weight": 0.0, "support": 0}
    ))

    for rule in candidate_rules:
        key = tuple(sorted(rule.antecedents.items()))
        group = groups[key][rule.consequent]

        group["weight"] += rule.weight
        group["support"] += rule.support

    resolved = []

    for antecedents_key, consequents in groups.items():
        # Highest total weight wins. Ties are resolved deterministically.
        winner = min(
            consequents,
            key=lambda c: (
                -consequents[c]["weight"],
                str(type(c)),
                repr(c)
            )
        )

        stats = consequents[winner]

        resolved.append(
            Rule(
                antecedents=dict(antecedents_key),
                consequent=winner,
                weight=stats["weight"],
                support=stats["support"]
            )
        )

    return sorted(
        resolved,
        key=lambda r: (
            -r.weight,
            -r.support,
            repr(sorted(r.antecedents.items()))
        )
    )


def extract_rules(samples, predictions):
    """
    Extract and resolve fuzzy rules from multiple samples.

    samples: list of membership-degree dictionaries
    predictions: corresponding MLP-predicted labels
    """
    if len(samples) != len(predictions):
        raise ValueError("Samples and predictions must have equal length.")

    candidates = [
        generate_candidate_rule(sample, prediction)
        for sample, prediction in zip(samples, predictions)
    ]

    return resolve_conflicts(candidates)