import joblib
import pandas as pd
from pathlib import Path
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score

Path("artifacts").mkdir(exist_ok=True)
Path("results").mkdir(exist_ok=True)

# Load processed datasets and MLP predictions
data = {}
mlp = {}

for split in ["train", "validation", "test"]:
    data[split] = pd.read_csv(f"data/{split}_processed.csv")
    mlp[split] = pd.read_csv(f"results/mlp_{split}_predictions.csv")

features = [c for c in data["train"].columns if c not in ["row_id", "true_label"]]

X_train = data["train"][features]
X_val = data["validation"][features]
X_test = data["test"][features]

y_train = mlp["train"]["predicted_label"]
y_val_mlp = mlp["validation"]["predicted_label"]
y_test_mlp = mlp["test"]["predicted_label"]

# Select tree complexity using validation fidelity
rows = []

for depth in [2, 3, 4, 5, 6, 8, 10]:
    for leaf in [2, 5, 10, 15]:
        model = DecisionTreeClassifier(
            max_depth=depth,
            min_samples_leaf=leaf,
            random_state=42
        )
        model.fit(X_train, y_train)

        pred = model.predict(X_val)
        rows.append({
            "max_depth": depth,
            "min_samples_leaf": leaf,
            "validation_fidelity": accuracy_score(y_val_mlp, pred),
            "validation_accuracy": accuracy_score(
                data["validation"]["true_label"], pred
            ),
            "validation_macro_f1": f1_score(
                data["validation"]["true_label"], pred, average="macro"
            )
        })

validation_results = pd.DataFrame(rows)
validation_results = validation_results.sort_values(
    ["validation_fidelity", "max_depth", "min_samples_leaf"],
    ascending=[False, True, False]
)
validation_results.to_csv("results/decision_tree_validation.csv", index=False)

best = validation_results.iloc[0]
best_depth = int(best["max_depth"])
best_leaf = int(best["min_samples_leaf"])

# Refit selected tree on training data only
tree = DecisionTreeClassifier(
    max_depth=best_depth,
    min_samples_leaf=best_leaf,
    random_state=42
)
tree.fit(X_train, y_train)

joblib.dump(tree, "artifacts/decision_tree_surrogate.joblib")

# Evaluate once on test data
tree_pred = tree.predict(X_test)
y_test = data["test"]["true_label"]

tree_accuracy = accuracy_score(y_test, tree_pred)
tree_f1 = f1_score(y_test, tree_pred, average="macro")
tree_fidelity = accuracy_score(y_test_mlp, tree_pred)

pd.DataFrame({
    "row_id": data["test"]["row_id"],
    "true_label": y_test,
    "mlp_prediction": y_test_mlp,
    "tree_prediction": tree_pred
}).to_csv("results/decision_tree_test_predictions.csv", index=False)

# Load final fuzzy model predictions
fuzzy = pd.read_csv("results/final_candidate_test_predictions.csv")

assert fuzzy["row_id"].equals(data["test"]["row_id"])
assert fuzzy["true_label"].equals(y_test)

fuzzy_pred = fuzzy["predicted_label"]

# Consolidated test comparison
comparison = pd.DataFrame([
    {
        "model": "MLP (teacher)",
        "accuracy": accuracy_score(y_test, y_test_mlp),
        "macro_f1": f1_score(y_test, y_test_mlp, average="macro"),
        "fidelity": 1.0
    },
    {
        "model": "Fuzzy surrogate",
        "accuracy": accuracy_score(y_test, fuzzy_pred),
        "macro_f1": f1_score(y_test, fuzzy_pred, average="macro"),
        "fidelity": accuracy_score(y_test_mlp, fuzzy_pred)
    },
    {
        "model": "Decision tree surrogate",
        "accuracy": tree_accuracy,
        "macro_f1": tree_f1,
        "fidelity": tree_fidelity
    }
])

comparison.to_csv("results/model_comparison_test.csv", index=False)

print("\nSelected decision tree")
print(f"Max depth: {best_depth}")
print(f"Minimum samples per leaf: {best_leaf}")
print(f"Validation fidelity: {best['validation_fidelity']:.4f}")

print("\nTest comparison")
print(comparison.to_string(index=False))
print("\nSaved:")
print("- artifacts/decision_tree_surrogate.joblib")
print("- results/decision_tree_validation.csv")
print("- results/decision_tree_test_predictions.csv")
print("- results/model_comparison_test.csv")
