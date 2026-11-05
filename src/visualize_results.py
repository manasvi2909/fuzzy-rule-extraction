from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


REQUIRED_COLUMNS = {
    "model",
    "pruning_setting",
    "split",
    "seed",
    "accuracy",
    "macro_f1",
    "fidelity",
    "rule_count",
    "mean_rule_length",
    "coverage",
    "fallback_rate"
}

METRICS = [
    "accuracy",
    "macro_f1",
    "fidelity",
    "rule_count",
    "mean_rule_length",
    "coverage",
    "fallback_rate"
]


def load_results(path):
    """Load and validate raw experiment results."""

    df = pd.read_csv(path)

    missing = REQUIRED_COLUMNS - set(df.columns)

    if missing:
        raise ValueError(f"Missing result columns: {sorted(missing)}")

    if df.empty:
        raise ValueError("Results file is empty.")

    numeric_columns = ["seed"] + METRICS

    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors="raise")

    required_numeric = [
        "accuracy",
        "macro_f1",
        "fidelity",
        "rule_count"
    ]

    if df[required_numeric].isna().any().any():
        raise ValueError("Required metric columns contain missing values.")

    if not df["accuracy"].between(0, 1).all():
        raise ValueError("Accuracy must be between 0 and 1.")

    if not df["macro_f1"].between(0, 1).all():
        raise ValueError("Macro-F1 must be between 0 and 1.")

    if not df["fidelity"].between(0, 1).all():
        raise ValueError("Fidelity must be between 0 and 1.")

    if not df["coverage"].between(0, 1).all():
        raise ValueError("Coverage must be between 0 and 1.")

    if not df["fallback_rate"].between(0, 1).all():
        raise ValueError("Fallback rate must be between 0 and 1.")

    if (df["rule_count"] < 0).any():
        raise ValueError("Rule count cannot be negative.")

    if (df["mean_rule_length"] < 0).any():
        raise ValueError("Mean rule length cannot be negative.")

    # A row should represent one unique experimental run/configuration.
    keys = ["model", "pruning_setting", "split", "seed"]

    if df.duplicated(keys).any():
        raise ValueError("Duplicate experiment configurations detected.")

    return df


def aggregate_results(df):
    """
    Aggregate repeated seeds.

    Standard deviation uses sample standard deviation.
    It is NaN when only one seed is available.
    """

    group_columns = [
        "model",
        "pruning_setting",
        "split",
        "rule_count"
    ]

    summary = (
        df.groupby(group_columns)[METRICS]
        .agg(["mean", "std", "count"])
        .reset_index()
    )

    # Flatten multi-level column names.
    summary.columns = [
        "_".join(str(x) for x in col if x != "")
        if isinstance(col, tuple) else col
        for col in summary.columns
    ]

    return summary


def _save_figure(fig, output_path):
    """Save a high-resolution figure and close it."""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
        facecolor="white"
    )

    plt.close(fig)


def _plot_metric(
    df,
    metric,
    ylabel,
    title,
    output_path,
    split="test"
):
    """Plot a metric against retained rule count."""

    data = df[df["split"] == split].copy()

    if data.empty:
        return False

    grouped = (
        data.groupby(
            ["model", "pruning_setting", "rule_count"]
        )[metric]
        .agg(["mean", "std", "count"])
        .reset_index()
        .sort_values("rule_count")
    )

    fig, ax = plt.subplots(figsize=(8, 5.5))

    for model in grouped["model"].unique():
        part = grouped[grouped["model"] == model]

        errors = part["std"].fillna(0).to_numpy()

        ax.errorbar(
            part["rule_count"],
            part["mean"],
            yerr=errors if (part["count"] > 1).any() else None,
            marker="o",
            markersize=6,
            linewidth=2,
            capsize=4,
            label=model
        )

    ax.set_xlabel("Number of retained rules", fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(f"{title} ({split.capitalize()} split)", fontsize=13)

    # Full metric scale avoids exaggerating small differences.
    if metric in {"accuracy", "macro_f1", "fidelity"}:
        ax.set_ylim(0, 1)

    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(frameon=False)
    fig.tight_layout()

    _save_figure(fig, output_path)
    return True


def plot_performance(df, output_dir, split="test"):
    """Plot accuracy and macro-F1 against rule count."""

    output_dir = Path(output_dir)

    _plot_metric(
        df,
        "accuracy",
        "Accuracy",
        "Predictive Accuracy vs Rule-Base Size",
        output_dir / "01_performance_vs_rule_count.png",
        split
    )

    _plot_metric(
        df,
        "macro_f1",
        "Macro-F1",
        "Macro-F1 vs Rule-Base Size",
        output_dir / "01b_macro_f1_vs_rule_count.png",
        split
    )


def plot_fidelity(df, output_dir, split="test"):
    """Plot fuzzy-model fidelity to the MLP."""

    _plot_metric(
        df,
        "fidelity",
        "Fidelity (agreement with MLP)",
        "Fidelity vs Rule-Base Size",
        Path(output_dir) / "02_fidelity_vs_rule_count.png",
        split
    )


def plot_coverage_fallback(df, output_dir, split="test"):
    """Plot coverage and fallback rate against rule count."""

    data = df[df["split"] == split].copy()

    if data.empty:
        return

    fig, ax = plt.subplots(figsize=(8, 5.5))

    for metric, label in [
        ("coverage", "Coverage"),
        ("fallback_rate", "Fallback rate")
    ]:
        grouped = (
            data.groupby(["model", "rule_count"])[metric]
            .mean()
            .reset_index()
            .sort_values("rule_count")
        )

        for model in grouped["model"].unique():
            part = grouped[grouped["model"] == model]

            ax.plot(
                part["rule_count"],
                part[metric] * 100,
                marker="o",
                linewidth=2,
                label=f"{model} — {label}"
            )

    ax.set_xlabel("Number of retained rules", fontsize=11)
    ax.set_ylabel("Percentage of evaluation samples (%)", fontsize=11)
    ax.set_title(
        f"Rule Coverage and Fallback Rate ({split.capitalize()} split)",
        fontsize=13
    )

    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(frameon=False)
    fig.tight_layout()

    _save_figure(
        fig,
        Path(output_dir) / "03_coverage_fallback_vs_rule_count.png"
    )


def plot_accuracy_fidelity_tradeoff(df, output_dir, split="test"):
    """Show the accuracy-fidelity trade-off without crowded annotations."""

    data = df[df["split"] == split].copy()

    if data.empty:
        return

    grouped = (
        data.groupby(
            ["model", "pruning_setting", "rule_count"]
        )[["accuracy", "fidelity"]]
        .mean()
        .reset_index()
        .sort_values("rule_count")
    )

    fig, ax = plt.subplots(figsize=(8, 6))

    for model in grouped["model"].unique():
        part = grouped[grouped["model"] == model]

        ax.scatter(
            part["accuracy"],
            part["fidelity"],
            s=90,
            label=model,
            zorder=3
        )

    ax.set_xlabel("Predictive accuracy", fontsize=11)
    ax.set_ylabel("Fidelity to MLP", fontsize=11)
    ax.set_title(
        f"Accuracy-Fidelity Trade-off ({split.capitalize()} split)",
        fontsize=13
    )

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(frameon=False)
    fig.tight_layout()

    _save_figure(
        fig,
        Path(output_dir) / "04_accuracy_fidelity_tradeoff.png"
    )


def generate_report_figures(
    metrics_path="results/metrics.csv",
    output_dir="results/figures",
    split="test"
):
    """Generate all figures and the aggregated summary."""

    df = load_results(metrics_path)
    summary = aggregate_results(df)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary.to_csv(
        output_dir.parent / "metrics_summary.csv",
        index=False
    )

    plot_performance(df, output_dir, split)
    plot_fidelity(df, output_dir, split)
    plot_coverage_fallback(df, output_dir, split)
    plot_accuracy_fidelity_tradeoff(df, output_dir, split)

    return summary