
from pathlib import Path

import pandas as pd

from src.analyze_results import (
    validate_results,
    aggregate_by_seed,
    calculate_tradeoffs
)
import math


COMPARISON_COLUMNS = [
    "model",
    "pruning_setting",
    "accuracy",
    "macro_f1",
    "fidelity",
    "rule_count",
    "mean_rule_length",
    "coverage",
    "fallback_rate"
]

RULE_COLUMNS = [
    "rule_number",
    "antecedents",
    "consequent",
    "weight",
    "support",
    "rule_length"
]


def build_comparison_table(df, split="test", mlp_baseline=None):
    """
    Build a report-ready comparison table.

    Repeated seeds are aggregated using mean and sample
    standard deviation. Single-seed standard deviations
    remain NaN.
    """

    df = validate_results(df)

    data = df[df["split"] == split].copy()

    if data.empty:
        raise ValueError(f"No results available for split: {split}")

    summary = aggregate_by_seed(data)

    columns = [
        "model",
        "pruning_setting"
    ]

    for metric in [
        "accuracy",
        "macro_f1",
        "fidelity",
        "rule_count",
        "mean_rule_length",
        "coverage",
        "fallback_rate"
    ]:
        columns.extend([
            f"{metric}_mean",
            f"{metric}_std",
            f"{metric}_count"
        ])

    summary = summary[columns].copy()

    # Attach teacher accuracy loss and other derived quantities.
    if mlp_baseline is not None:

        detailed = calculate_tradeoffs(
            data,
            mlp_baseline=mlp_baseline
        )

        derived = (
            detailed.groupby(
                ["model", "pruning_setting"]
            )[
                [
                    "accuracy_loss_vs_mlp",
                    "accuracy_loss_vs_unpruned",
                    "rule_reduction_pct",
                    "mean_rule_length_reduction_pct",
                    "coverage_deterioration",
                    "fidelity_loss"
                ]
            ]
            .mean()
            .reset_index()
        )

        summary = summary.merge(
            derived,
            on=["model", "pruning_setting"],
            how="left",
            validate="one_to_one"
        )

    return summary.sort_values(
        ["model", "rule_count_mean"],
        kind="stable"
    ).reset_index(drop=True)


def build_rule_table(rules, max_rules=None):
    """
    Convert extracted Rule objects into a tabular representation.

    Expected Rule attributes:
        antecedents: dict[str, str]
        consequent: class label
        weight: numeric rule weight
        support: integer support count
    """

    if not isinstance(rules, (list, tuple)):
        raise TypeError("Rules must be provided as a list or tuple.")

    if max_rules is not None:
        if (
            not isinstance(max_rules, int)
            or isinstance(max_rules, bool)
            or max_rules < 0
        ):
            raise ValueError("max_rules must be a non-negative integer.")

        rules = rules[:max_rules]

    rows = []

    for i, rule in enumerate(rules, start=1):

        if not hasattr(rule, "antecedents"):
            raise TypeError("Every rule must have antecedents.")

        antecedents = rule.antecedents

        if not isinstance(antecedents, dict):
            raise TypeError("Rule antecedents must be a dictionary.")

        formatted = " AND ".join(
            f"{feature} IS {term}"
            for feature, term in sorted(antecedents.items())
        )

        weight = float(rule.weight)

        if not math.isfinite(weight) or weight < 0:
            raise ValueError("Rule weight must be finite and non-negative.")

        support = rule.support

        if (
            not isinstance(support, int)
            or isinstance(support, bool)
            or support < 0
        ):
            raise ValueError("Rule support must be a non-negative integer.")

        rows.append({
            "rule_number": i,
            "antecedents": formatted,
            "consequent": str(rule.consequent),
            "weight": weight,
            "support": support,
            "rule_length": len(antecedents)
        })

    return pd.DataFrame(rows, columns=RULE_COLUMNS)


def _export_table(df, output_dir, filename):
    """Export a DataFrame in CSV, Markdown and LaTeX formats."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / f"{filename}.csv"
    md_path = output_dir / f"{filename}.md"
    tex_path = output_dir / f"{filename}.tex"

    df.to_csv(csv_path, index=False)

    md_path.write_text(
        df.to_markdown(index=False),
        encoding="utf-8"
    )

    tex_path.write_text(
        df.to_latex(
            index=False,
            escape=True,
            float_format=lambda x: f"{x:.4f}"
        ),
        encoding="utf-8"
    )

    return {
        "csv": csv_path,
        "markdown": md_path,
        "latex": tex_path
    }


def export_comparison_table(
    df,
    output_dir="report/tables",
    split="test",
    mlp_baseline=None
):
    """Build and export the experiment comparison table."""

    table = build_comparison_table(
        df,
        split=split,
        mlp_baseline=mlp_baseline
    )

    paths = _export_table(
        table,
        output_dir,
        f"comparison_{split}"
    )

    return table, paths


def export_rule_table(
    rules,
    output_dir="report/tables",
    max_rules=None
):
    """Build and export the extracted-rule table."""

    table = build_rule_table(
        rules,
        max_rules=max_rules
    )

    paths = _export_table(
        table,
        output_dir,
        "extracted_rules"
    )

    return table, paths
