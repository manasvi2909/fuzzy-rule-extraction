from pathlib import Path

import pandas as pd

from src.evaluate import evaluate_metrics
from src.prune_rules import prune_rules


def run_rule_experiments(
    X,
    y_true,
    mlp_predictions,
    rules,
    inference_fn,
    top_n_values=(10, 25, 50, 100),
    split="validation",
    seed=42
):
    """
    Evaluate full and pruned fuzzy rule bases.

    inference_fn must accept (X, rules) and return:
        predictions, covered, fallback
    """

    if len(X) != len(y_true) or len(y_true) != len(mlp_predictions):
        raise ValueError("Input and prediction lengths must match.")

    if not rules:
        raise ValueError("The extracted rule base cannot be empty.")

    if len(set(top_n_values)) != len(top_n_values):
        raise ValueError("Pruning configurations must be unique.")

    if any(not isinstance(n, int) or isinstance(n, bool) or n <= 0
           for n in top_n_values):
        raise ValueError("Every top-N value must be a positive integer.")

    # Full rule base plus unique pruned configurations.
    configurations = [("all", rules)]

    seen_sizes = {len(rules)}

    for n in sorted(top_n_values):
        pruned = prune_rules(rules, n)

        if len(pruned) not in seen_sizes:
            configurations.append((f"top_{n}", pruned))
            seen_sizes.add(len(pruned))

    results = []

    for setting, selected_rules in configurations:

        output = inference_fn(X, selected_rules)

        if not isinstance(output, tuple) or len(output) != 3:
            raise ValueError(
                "inference_fn must return "
                "(predictions, covered, fallback)."
            )

        predictions, covered, fallback = output

        metrics = evaluate_metrics(
            y_true=y_true,
            fuzzy_predictions=predictions,
            mlp_predictions=mlp_predictions,
            rules=selected_rules,
            covered=covered,
            fallback=fallback
        )

        results.append({
            "model": "Fuzzy",
            "pruning_setting": setting,
            "split": split,
            "seed": seed,
            **metrics
        })

    return results


def save_metrics(results, output_path="results/metrics.csv"):
    """Save experiment results to a CSV file."""

    if not results:
        raise ValueError("No experiment results to save.")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(results)
    df.to_csv(output_path, index=False)

    return df