
import joblib
import pandas as pd

from src.fuzzy_inference import infer_sample
from src.membership import fuzzify_sample


val = pd.read_csv("data/validation_processed.csv").set_index("row_id")

rules = joblib.load("artifacts/selected_rules_top100.joblib")
mf = joblib.load("artifacts/membership_functions.joblib")

# Remove original rule #0.
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

    benign = sorted(
        [r for r in result["fired_rules"] if r["consequent"] == 0],
        key=lambda r: r["firing_strength"],
        reverse=True
    )

    malignant = sorted(
        [r for r in result["fired_rules"] if r["consequent"] == 1],
        key=lambda r: r["firing_strength"],
        reverse=True
    )

    if not malignant:
        continue

    b = benign[0]
    m = malignant[0]

    print(f"\n{'=' * 55}")
    print(f"ROW: {row_id}")
    print("Benign rule:", b["rule_index"] + 1)
    print("Benign strength:", round(b["firing_strength"], 4))
    print("Malignant rule:", m["rule_index"] + 1)
    print("Malignant strength:", round(m["firing_strength"], 4))

    differences = []

    for feature in b["antecedents"]:
        bt = b["antecedents"][feature]
        mt = m["antecedents"][feature]

        if bt != mt:
            differences.append({
                "feature": feature,
                "benign_term": bt,
                "malignant_term": mt,
                "sample_memberships": memberships[feature]
            })

    print("Differing antecedents:")
    for d in differences:
        print(d)

    print("Number of differing features:", len(differences))
