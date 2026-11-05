"""
Mamdani-inspired fuzzy rule classification for the project's binary classifier.

Each rule's antecedent is evaluated using the minimum t-norm
(fuzzy AND). The resulting antecedent strength is used directly
as the rule's firing strength.

The strongest rule for each class is retained using max aggregation.
The class with the largest aggregated activation wins.

Because the consequents are categorical class labels (0/1), this
project uses class-wise activation scores and winner-take-all
classification rather than centroid defuzzification.
"""

from math import isfinite

import numpy as np
import pandas as pd

from src.membership import fuzzify_sample
from src.rule_types import Rule


DEFAULT_CLASS_ORDER = (0, 1)
DEFAULT_FALLBACK_CLASS = 0  # Training-set majority class: Benign.


def _validate_rules(rules):
    if rules is None:
        raise TypeError("rules cannot be None.")
    rules = list(rules)
    if any(not isinstance(rule, Rule) for rule in rules):
        raise TypeError("Every item in rules must be a Rule.")
    for rule in rules:
        if not rule.antecedents:
            raise ValueError("A rule must have at least one antecedent.")
        if not isinstance(rule.support, int) or isinstance(rule.support, bool):
            raise TypeError("Rule support must be a positive integer.")
        if rule.support <= 0:
            raise ValueError("Rule support must be a positive integer.")
        if not isinstance(rule.weight, (int, float)) or not isfinite(rule.weight):
            raise ValueError("Rule weight must be a finite number.")
        if rule.weight < 0:
            raise ValueError("Rule weight cannot be negative.")
    return rules


def infer_sample(
    sample,
    rules,
    membership_functions,
    fallback_class=DEFAULT_FALLBACK_CLASS,
    class_order=DEFAULT_CLASS_ORDER,
):
    """
    Infer the class for one sample and return a prediction plus an audit trace.

    The firing strength is determined exclusively by the minimum
    membership degree across the rule's antecedents.

    Rule weights and support are retained as extraction metadata
    for ranking, pruning and analysis, but do not scale firing strength.
    """
    rules = _validate_rules(rules)
    if class_order is None or not class_order:
        raise ValueError("class_order must contain at least one class label.")

    memberships = fuzzify_sample(sample, membership_functions)
    class_scores = {}
    fired_rules = []

    for index, rule in enumerate(rules):
        missing = set(rule.antecedents) - set(memberships)
        if missing:
            raise ValueError(
                f"Rule {index} refers to unknown features: {sorted(missing)}"
            )

        degrees = []
        for feature, term in rule.antecedents.items():
            if term not in memberships[feature]:
                raise ValueError(
                    f"Unknown term {term!r} for feature {feature!r}."
                )
            degrees.append(memberships[feature][term])

        # Mamdani AND: minimum t-norm.
        antecedent_strength = min(degrees)

        # Use the minimum t-norm directly
        firing_strength = antecedent_strength

        if firing_strength > 0.0:
            label = rule.consequent
            class_scores[label] = max(
                class_scores.get(label, 0.0),
                firing_strength
            )
            fired_rules.append({
                "rule_index": index,
                "antecedent_strength": float(antecedent_strength),
                "firing_strength": float(firing_strength),
                "rule_weight": float(rule.weight),
                "support": rule.support,
                "consequent": label,
                "antecedents": dict(rule.antecedents),
            })

    if not class_scores:
        return {
            "prediction": fallback_class,
            "covered": False,
            "fallback": True,
            "class_scores": {},
            "fired_rules": [],
        }

    # Deterministic tie handling: first class in class_order wins.
    # Include any labels not explicitly listed, in stable repr order.
    labels = list(class_scores)
    ordered_labels = [label for label in class_order if label in class_scores]
    ordered_labels.extend(
        sorted(
            (label for label in labels if label not in ordered_labels),
            key=lambda label: (str(type(label)), repr(label))
        )
    )
    prediction = max(
        ordered_labels,
        key=lambda label: class_scores[label]
    )

    return {
        "prediction": prediction,
        "covered": True,
        "fallback": False,
        "class_scores": class_scores,
        "fired_rules": fired_rules,
    }


def predict_dataset(
    X,
    rules,
    membership_functions,
    fallback_class=DEFAULT_FALLBACK_CLASS,
    class_order=DEFAULT_CLASS_ORDER,
):
    """
    Predict a complete DataFrame.

    Returns the exact three-item tuple required by run_rule_experiments:
        predictions, covered, fallback
    """
    if not isinstance(X, pd.DataFrame):
        raise TypeError("X must be a pandas DataFrame.")

    rules = _validate_rules(rules)
    predictions = []
    covered = []
    fallback = []

    for _, row in X.iterrows():
        result = infer_sample(
            row,
            rules,
            membership_functions,
            fallback_class=fallback_class,
            class_order=class_order,
        )
        predictions.append(result["prediction"])
        covered.append(result["covered"])
        fallback.append(result["fallback"])

    return (
        np.asarray(predictions),
        np.asarray(covered, dtype=bool),
        np.asarray(fallback, dtype=bool),
    )


def make_inference_function(
    membership_functions,
    fallback_class=DEFAULT_FALLBACK_CLASS,
    class_order=DEFAULT_CLASS_ORDER,
):
    """
    Create an inference_fn(X, rules) callback for run_rule_experiments().
    """
    def inference_fn(X, rules):
        return predict_dataset(
            X,
            rules,
            membership_functions,
            fallback_class=fallback_class,
            class_order=class_order,
        )

    return inference_fn
