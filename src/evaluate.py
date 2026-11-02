import numpy as np
from sklearn.metrics import accuracy_score, f1_score


def evaluate_metrics(
    y_true,
    fuzzy_predictions,
    mlp_predictions,
    rules,
    covered=None,
    fallback=None
):
    """
    Evaluate a fuzzy rule-based model.

    Parameters:
        y_true: Actual class labels.
        fuzzy_predictions: Predictions from fuzzy inference.
        mlp_predictions: Predictions from the teacher MLP.
        rules: Extracted/pruned Rule objects.
        covered: Boolean array indicating whether each sample
                 was handled by the fuzzy rule base.
        fallback: Boolean array indicating whether fallback
                  was used for each sample.

    Returns:
        Dictionary of evaluation metrics.
    """

    y_true = np.asarray(y_true)
    fuzzy_predictions = np.asarray(fuzzy_predictions)
    mlp_predictions = np.asarray(mlp_predictions)

    n = len(y_true)

    if n == 0:
        raise ValueError("Evaluation data cannot be empty.")

    if len(fuzzy_predictions) != n or len(mlp_predictions) != n:
        raise ValueError("All prediction arrays must have equal length.")

    if covered is not None:
        covered = np.asarray(covered, dtype=bool)
        if len(covered) != n:
            raise ValueError("Coverage array length must match predictions.")

    if fallback is not None:
        fallback = np.asarray(fallback, dtype=bool)
        if len(fallback) != n:
            raise ValueError("Fallback array length must match predictions.")

    if covered is None and fallback is not None:
        covered = ~fallback

    if fallback is None and covered is not None:
        fallback = ~covered

    if covered is not None and fallback is not None:
        if np.any(covered == fallback):
            raise ValueError(
                "Coverage and fallback flags must be logical opposites."
            )

    accuracy = accuracy_score(y_true, fuzzy_predictions)

    macro_f1 = f1_score(
        y_true,
        fuzzy_predictions,
        average="macro",
        zero_division=0
    )

    fidelity = accuracy_score(mlp_predictions, fuzzy_predictions)

    rule_count = len(rules)

    mean_rule_length = (
        float(np.mean([len(rule.antecedents) for rule in rules]))
        if rules else 0.0
    )

    coverage = float(np.mean(covered)) if covered is not None else np.nan

    fallback_rate = (
        float(np.mean(fallback)) if fallback is not None else np.nan
    )

    return {
        "accuracy": float(accuracy),
        "macro_f1": float(macro_f1),
        "fidelity": float(fidelity),
        "rule_count": rule_count,
        "mean_rule_length": mean_rule_length,
        "coverage": coverage,
        "fallback_rate": fallback_rate
    }