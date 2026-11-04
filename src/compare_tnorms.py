
import math
import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    recall_score,
    confusion_matrix
)

from src.membership import fuzzify_sample


# Load existing data and artifacts.
val = pd.read_csv("data/validation_processed.csv")

X = val.drop(columns=["row_id", "true_label"])
y = val["true_label"].to_numpy()

rules = joblib.load("artifacts/selected_rules_top100.joblib")
mf = joblib.load("artifacts/membership_functions.joblib")


def predict_with_tnorm(X, selected_rules, method):

    predictions = []
    covered = []

    for _, sample in X.iterrows():

        memberships = fuzzify_sample(sample, mf)
        scores = {}

        for rule in selected_rules:

            degrees = [
                memberships[feature][term]
                for feature, term in rule.antecedents.items()
            ]

            if method == "min":
                strength = min(degrees)
            elif method == "product":
                strength = math.prod(degrees)
            else:
                raise ValueError("Unknown t-norm")

            if strength > 0:
                label = rule.consequent
                scores[label] = max(
                    scores.get(label, 0),
                    strength
                )

        if not scores:
            predictions.append(0)
            covered.append(False)
        else:
            # Deterministic tie-breaking: class 0 wins.
            predictions.append(
                max((0, 1), key=lambda c: scores.get(c, 0))
            )
            covered.append(True)

    return np.array(predictions), np.array(covered)


def evaluate(name, selected_rules, method):

    pred, coverage = predict_with_tnorm(
        X, selected_rules, method
    )

    print(f"\n{name}")
    print("-" * 45)
    print("Rules:", len(selected_rules))
    print("T-norm:", method)
    print("Accuracy:", round(accuracy_score(y, pred), 4))
    print("Macro-F1:", round(
        f1_score(y, pred, average="macro"), 4
    ))
    print("Malignant recall:", round(
        recall_score(y, pred, pos_label=1), 4
    ))
    print("Coverage:", round(coverage.mean(), 4))
    print("Confusion matrix:")
    print(confusion_matrix(y, pred))


# Original 100-rule system.
evaluate("ORIGINAL — MINIMUM", rules, "min")
evaluate("ORIGINAL — PRODUCT", rules, "product")

# Experimental 99-rule system without original rule #0.
rules_without_0 = [
    rule for i, rule in enumerate(rules) if i != 0
]

evaluate("WITHOUT RULE #0 — MINIMUM", rules_without_0, "min")
evaluate("WITHOUT RULE #0 — PRODUCT", rules_without_0, "product")
