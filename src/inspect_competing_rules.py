
import joblib
import pandas as pd
from collections import Counter

rules = joblib.load("artifacts/selected_rules_top100.joblib")

df = pd.read_csv("results/remaining_false_negatives.csv")

benign_counts = Counter()
malignant_counts = Counter()

for _, row in df.iterrows():
    for idx, strength in eval(row["top_benign_rules"]):
        benign_counts[idx] += 1

    for idx, strength in eval(row["top_malignant_rules"]):
        malignant_counts[idx] += 1


def show_rules(title, counts):
    print(f"\n{title}")

    for idx, count in counts.most_common():
        # Convert post-ablation index to original index.
        original_idx = idx + 1

        rule = rules[original_idx]

        print(f"\nOriginal rule #{original_idx}")
        print("Consequent:", rule.consequent)
        print("Appears in top 5 for:", count, "samples")
        print("Weight:", rule.weight)
        print("Support:", rule.support)
        print("Antecedents:", rule.antecedents)


show_rules("RECURRING BENIGN RULES", benign_counts)
show_rules("RECURRING MALIGNANT RULES", malignant_counts)
