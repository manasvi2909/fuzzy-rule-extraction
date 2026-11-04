
import joblib
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    classification_report
)

from src.fuzzy_inference import predict_dataset


# Load frozen artifacts.
test = pd.read_csv("data/test_processed.csv")
mlp = pd.read_csv("results/mlp_test_predictions.csv")

rules = joblib.load("artifacts/selected_rules_top100.joblib")
mf = joblib.load("artifacts/membership_functions.joblib")

# Exclude original rule #0.
candidate_rules = [
    rule for i, rule in enumerate(rules)
    if i != 0
]

X = test.drop(columns=["row_id", "true_label"])
y = test["true_label"].to_numpy()

# Generate candidate predictions.
pred, covered, fallback = predict_dataset(
    X,
    candidate_rules,
    mf
)

# Align MLP predictions by row_id.
comparison = pd.DataFrame({
    "row_id": test["row_id"],
    "fuzzy_prediction": pred
}).merge(
    mlp[["row_id", "predicted_label"]],
    on="row_id",
    how="inner",
    validate="one_to_one"
)

assert len(comparison) == len(test), "Test rows did not align."

mlp_predictions = comparison["predicted_label"].to_numpy()

print("\nFINAL CANDIDATE — TEST RESULTS")
print("=" * 45)

print("Rules:", len(candidate_rules))
print("Accuracy:", round(accuracy_score(y, pred), 4))
print("Macro-F1:", round(
    f1_score(y, pred, average="macro"), 4
))
print("Malignant precision:", round(
    precision_score(y, pred, pos_label=1), 4
))
print("Malignant recall:", round(
    recall_score(y, pred, pos_label=1), 4
))
print("Malignant F1:", round(
    f1_score(y, pred, pos_label=1), 4
))
print("Fidelity:", round(
    accuracy_score(mlp_predictions, pred), 4
))
print("Coverage:", round(covered.mean(), 4))
print("Fallback rate:", round(fallback.mean(), 4))

print("\nConfusion Matrix:")
print(confusion_matrix(y, pred))

print("\nClassification Report:")
print(classification_report(
    y, pred,
    target_names=["Benign", "Malignant"]
))

# Save the final candidate's test predictions.
output = test[["row_id", "true_label"]].copy()
output["predicted_label"] = pred

output.to_csv(
    "results/final_candidate_test_predictions.csv",
    index=False
)

print("\nSaved: results/final_candidate_test_predictions.csv")
