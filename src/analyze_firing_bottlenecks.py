
import joblib
import pandas as pd

from src.fuzzy_inference import infer_sample
from src.membership import fuzzify_sample


val = pd.read_csv("data/validation_processed.csv").set_index("row_id")

rules = joblib.load("artifacts/selected_rules_top100.joblib")
mf = joblib.load("artifacts/membership_functions.joblib")

# Remove original rule #0 for this diagnostic.
rules = [r for i, r in enumerate(rules) if i != 0]

diagnostic = pd.read_csv("results/remaining_false_negatives.csv")

for _, item in diagnostic.iterrows():

    row_id = int(item["row_id"])

    if not item["benign_rule_count"]:
        print(f"\nRow {row_id}: UNCOVERED")
        continue

    sample = val.loc[row_id].drop("true_label")
    memberships = fuzzify_sample(sample, mf)

    result = infer_sample(sample, rules, mf)

    print(f"\n{'=' * 55}")
    print("ROW:", row_id)

    for label, class_name in [(0, "BENIGN"), (1, "MALIGNANT")]:

        fired = sorted(
            [
                r for r in result["fired_rules"]
                if r["consequent"] == label
            ],
            key=lambda r: r["firing_strength"],
            reverse=True
        )

        if not fired:
            print(f"\n{class_name}: No rules fired")
            continue

        rule = fired[0]

        degrees = [
            (feature, term, memberships[feature][term])
            for feature, term in rule["antecedents"].items()
        ]

        minimum = min(x[2] for x in degrees)

        bottlenecks = [
            (feature, term, round(degree, 6))
            for feature, term, degree in degrees
            if abs(degree - minimum) < 1e-9
        ]

        print(f"\n{class_name}")
        print("Rule:", rule["rule_index"] + 1)
        print("Firing strength:", round(rule["firing_strength"], 6))
        print("Bottleneck antecedents:", bottlenecks)
