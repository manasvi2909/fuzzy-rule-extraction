
import joblib
import pandas as pd

from src.fuzzy_inference import infer_sample


val = pd.read_csv("data/validation_processed.csv")

rules = joblib.load("artifacts/selected_rules_top100.joblib")
membership_functions = joblib.load(
    "artifacts/membership_functions.joblib"
)

ablated_rules = [r for i, r in enumerate(rules) if i != 0]

records = []

for _, row in val.iterrows():
    row_id = int(row["row_id"])
    actual = int(row["true_label"])
    sample = row.drop(["row_id", "true_label"])

    original = infer_sample(
        sample, rules, membership_functions
    )

    ablated = infer_sample(
        sample, ablated_rules, membership_functions
    )

    records.append({
        "row_id": row_id,
        "actual": actual,
        "original_prediction": original["prediction"],
        "ablated_prediction": ablated["prediction"],
        "original_benign_score": original["class_scores"].get(0, 0),
        "original_malignant_score": original["class_scores"].get(1, 0),
        "ablated_benign_score": ablated["class_scores"].get(0, 0),
        "ablated_malignant_score": ablated["class_scores"].get(1, 0),
        "rule_0_fired": any(
            r["rule_index"] == 0
            for r in original["fired_rules"]
        )
    })

df = pd.DataFrame(records)

changed = df[
    df["original_prediction"] != df["ablated_prediction"]
]

corrected = changed[
    changed["original_prediction"] != changed["actual"]
]

new_errors = changed[
    changed["original_prediction"] == changed["actual"]
]

print("\nALL CHANGED PREDICTIONS")
print(changed.to_string(index=False))

print("\nPREVIOUSLY INCORRECT, NOW CORRECT")
print(corrected.to_string(index=False))

print("\nPREVIOUSLY CORRECT, NOW INCORRECT")
print(new_errors.to_string(index=False))

print("\nRULE #0 ACTIVATION")
print(
    df.groupby("rule_0_fired")[
        ["original_benign_score", "original_malignant_score"]
    ].mean()
)

df.to_csv("results/rule_0_ablation_changes.csv", index=False)

print("\nSaved: results/rule_0_ablation_changes.csv")
