
import joblib
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    recall_score,
    confusion_matrix
)

from src.fuzzy_inference import predict_dataset


# Load validation data and existing artifacts.
val = pd.read_csv("data/validation_processed.csv")

rules = joblib.load("artifacts/selected_rules_top100.joblib")
membership_functions = joblib.load(
    "artifacts/membership_functions.joblib"
)

X = val.drop(columns=["row_id", "true_label"])
y = val["true_label"].to_numpy()


def evaluate(name, selected_rules):

    predictions, covered, fallback = predict_dataset(
        X,
        selected_rules,
        membership_functions
    )

    print(f"\n{name}")
    print("-" * 40)
    print("Rules:", len(selected_rules))
    print("Accuracy:", round(accuracy_score(y, predictions), 4))
    print("Macro-F1:", round(
        f1_score(y, predictions, average="macro"), 4
    ))
    print("Malignant recall:", round(
        recall_score(y, predictions, pos_label=1), 4
    ))
    print("Coverage:", round(covered.mean(), 4))
    print("Confusion matrix:")
    print(confusion_matrix(y, predictions))


# Original system.
evaluate("ORIGINAL 100-RULE SYSTEM", rules)

# Remove only rule #0.
ablated_rules = [
    rule for i, rule in enumerate(rules)
    if i != 0
]

evaluate("WITHOUT RULE #0", ablated_rules)
