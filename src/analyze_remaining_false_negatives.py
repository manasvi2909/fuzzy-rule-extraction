
import joblib
import pandas as pd

from src.fuzzy_inference import infer_sample


val = pd.read_csv("data/validation_processed.csv")
diagnostic = pd.read_csv("results/activation_diagnostic.csv")

rules = joblib.load("artifacts/selected_rules_top100.joblib")
membership_functions = joblib.load(
    "artifacts/membership_functions.joblib"
)

# Temporarily exclude rule #0.
rules_without_0 = [
    r for i, r in enumerate(rules) if i != 0
]

# Identify the original false negatives.
false_negatives = diagnostic[
    (diagnostic["actual"] == 1) &
    (diagnostic["prediction"] == 0)
]

val = val.set_index("row_id")

remaining = []

for _, item in false_negatives.iterrows():

    row_id = int(item["row_id"])
    sample = val.loc[row_id].drop("true_label")

    result = infer_sample(
        sample,
        rules_without_0,
        membership_functions
    )

    if result["prediction"] != 0:
        continue

    benign_rules = sorted(
        [
            r for r in result["fired_rules"]
            if r["consequent"] == 0
        ],
        key=lambda r: r["firing_strength"],
        reverse=True
    )

    malignant_rules = sorted(
        [
            r for r in result["fired_rules"]
            if r["consequent"] == 1
        ],
        key=lambda r: r["firing_strength"],
        reverse=True
    )

    remaining.append({
        "row_id": row_id,
        "benign_score": result["class_scores"].get(0, 0),
        "malignant_score": result["class_scores"].get(1, 0),
        "benign_rule_count": len(benign_rules),
        "malignant_rule_count": len(malignant_rules),
        "top_benign_rules": [
            (r["rule_index"], round(r["firing_strength"], 4))
            for r in benign_rules[:5]
        ],
        "top_malignant_rules": [
            (r["rule_index"], round(r["firing_strength"], 4))
            for r in malignant_rules[:5]
        ]
    })

result = pd.DataFrame(remaining)

print("\nREMAINING FALSE NEGATIVES AFTER REMOVING RULE #0")
print(result.to_string(index=False))

result.to_csv(
    "results/remaining_false_negatives.csv",
    index=False
)

print("\nSaved: results/remaining_false_negatives.csv")
