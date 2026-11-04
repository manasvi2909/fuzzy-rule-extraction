
import json
import joblib
import pandas as pd

from src.fuzzy_inference import infer_sample


# Load existing artifacts and validation data.
val = pd.read_csv("data/validation_processed.csv")
diagnostic = pd.read_csv("results/activation_diagnostic.csv")

rules = joblib.load("artifacts/selected_rules_top100.joblib")
membership_functions = joblib.load(
    "artifacts/membership_functions.joblib"
)

# Identify false negatives and correctly classified malignant samples.
false_negatives = diagnostic[
    (diagnostic["actual"] == 1) &
    (diagnostic["prediction"] == 0)
]

true_positives = diagnostic[
    (diagnostic["actual"] == 1) &
    (diagnostic["prediction"] == 1)
]

val = val.set_index("row_id")

all_details = []
all_summaries = []


def analyze_samples(samples, sample_type):
    for _, item in samples.iterrows():

        row_id = int(item["row_id"])

        # Retrieve the original validation sample.
        sample = val.loc[row_id].drop("true_label")

        result = infer_sample(
            sample,
            rules,
            membership_functions
        )

        fired = result["fired_rules"]

        # Separate fired rules by their consequent.
        for class_label in (0, 1):
            class_rules = [
                r for r in fired
                if r["consequent"] == class_label
            ]

            class_rules.sort(
                key=lambda r: r["firing_strength"],
                reverse=True
            )

            # Retain the five strongest rules per class.
            for rank, rule in enumerate(class_rules[:5], start=1):
                all_details.append({
                    "sample_type": sample_type,
                    "row_id": row_id,
                    "actual": int(item["actual"]),
                    "prediction": int(result["prediction"]),
                    "class": class_label,
                    "class_name": (
                        "Benign" if class_label == 0
                        else "Malignant"
                    ),
                    "rank": rank,
                    "rule_index": rule["rule_index"],
                    "firing_strength": rule["firing_strength"],
                    "rule_weight": rule["rule_weight"],
                    "support": rule["support"],
                    "antecedents": json.dumps(
                        rule["antecedents"],
                        sort_keys=True
                    )
                })

        benign_rules = [
            r for r in fired if r["consequent"] == 0
        ]
        malignant_rules = [
            r for r in fired if r["consequent"] == 1
        ]

        benign_rules.sort(
            key=lambda r: r["firing_strength"],
            reverse=True
        )
        malignant_rules.sort(
            key=lambda r: r["firing_strength"],
            reverse=True
        )

        all_summaries.append({
            "sample_type": sample_type,
            "row_id": row_id,
            "actual": int(item["actual"]),
            "prediction": int(result["prediction"]),
            "benign_score": result["class_scores"].get(0, 0),
            "malignant_score": result["class_scores"].get(1, 0),
            "benign_firing_rules": len(benign_rules),
            "malignant_firing_rules": len(malignant_rules),
            "strongest_benign_rule": (
                benign_rules[0]["rule_index"]
                if benign_rules else None
            ),
            "strongest_malignant_rule": (
                malignant_rules[0]["rule_index"]
                if malignant_rules else None
            ),
            "covered": result["covered"]
        })


analyze_samples(false_negatives, "false_negative")
analyze_samples(true_positives, "true_positive")


details = pd.DataFrame(all_details)
summary = pd.DataFrame(all_summaries)

details.to_csv(
    "results/false_negative_rule_details.csv",
    index=False
)

summary.to_csv(
    "results/false_negative_rule_summary.csv",
    index=False
)

print("\nSAMPLE-LEVEL RULE ACTIVATION SUMMARY")
print(summary.to_string(index=False))

print("\nTOP FIRED RULES")
print(details.to_string(index=False))

print("\nSaved:")
print("results/false_negative_rule_details.csv")
print("results/false_negative_rule_summary.csv")
