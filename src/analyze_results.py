
from __future__ import annotations

import numpy as np
import pandas as pd


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

DEFAULT_MAXIMIZE = [
    "accuracy",
    "fidelity",
    "coverage"
]

DEFAULT_MINIMIZE = [
    "rule_count",
    "mean_rule_length",
    "fallback_rate"
]


def validate_results(df):
    """Validate the experiment-results DataFrame."""

    missing = REQUIRED_COLUMNS - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing result columns: {sorted(missing)}"
        )

    if df.empty:
        raise ValueError("Results DataFrame is empty.")

    df = df.copy()

    numeric = ["seed"] + METRICS

    for col in numeric:
        df[col] = pd.to_numeric(df[col], errors="raise")

    required = [
        "accuracy",
        "macro_f1",
        "fidelity",
        "rule_count",
        "mean_rule_length",
        "coverage",
        "fallback_rate"
    ]

    if df[required].isna().any().any():
        raise ValueError("Required metrics cannot contain missing values.")

    bounded = [
        "accuracy",
        "macro_f1",
        "fidelity",
        "coverage",
        "fallback_rate"
    ]

    for col in bounded:
        if not df[col].between(0, 1).all():
            raise ValueError(f"{col} must be between 0 and 1.")

    if (df["rule_count"] < 0).any():
        raise ValueError("Rule count cannot be negative.")

    if (df["mean_rule_length"] < 0).any():
        raise ValueError("Mean rule length cannot be negative.")

    keys = ["model", "pruning_setting", "split", "seed"]

    if df.duplicated(keys).any():
        raise ValueError("Duplicate experiment configurations detected.")

    return df


def calculate_tradeoffs(df, mlp_baseline=None, baseline_setting="all"):
    """
    Calculate interpretability/performance trade-offs.

    mlp_baseline may be:
      - None
      - A scalar MLP accuracy
      - A DataFrame containing split, seed, accuracy

    The unpruned fuzzy configuration is used as the reference
    for rule reduction and fuzzy-model performance deterioration.
    """

    df = validate_results(df)

    keys = ["model", "split", "seed"]

    baseline = df[
        df["pruning_setting"] == baseline_setting
    ][
        keys + [
            "accuracy",
            "rule_count",
            "mean_rule_length",
            "coverage"
        ]
    ].copy()

    if baseline.duplicated(keys).any():
        raise ValueError("Multiple unpruned baselines found for a run.")

    baseline = baseline.rename(columns={
        "accuracy": "unpruned_accuracy",
        "rule_count": "unpruned_rule_count",
        "mean_rule_length": "unpruned_mean_rule_length",
        "coverage": "unpruned_coverage"
    })

    result = df.merge(
        baseline,
        on=keys,
        how="left",
        validate="many_to_one"
    )

    result["accuracy_loss_vs_unpruned"] = (
        result["unpruned_accuracy"] - result["accuracy"]
    )

    result["rule_reduction_pct"] = np.where(
        result["unpruned_rule_count"] > 0,
        (
            1 - result["rule_count"] /
            result["unpruned_rule_count"]
        ) * 100,
        0.0
    )

    result["mean_rule_length_reduction_pct"] = np.where(
        result["unpruned_mean_rule_length"] > 0,
        (
            1 - result["mean_rule_length"] /
            result["unpruned_mean_rule_length"]
        ) * 100,
        0.0
    )

    result["coverage_deterioration"] = (
        result["unpruned_coverage"] - result["coverage"]
    )

    result["fidelity_loss"] = 1 - result["fidelity"]

    # MLP baseline is separate from fuzzy experiment rows.
    result["mlp_accuracy"] = np.nan

    if mlp_baseline is not None:

        if np.isscalar(mlp_baseline):
            result["mlp_accuracy"] = float(mlp_baseline)

        elif isinstance(mlp_baseline, pd.DataFrame):

            required = {"split", "seed", "accuracy"}

            if not required.issubset(mlp_baseline.columns):
                raise ValueError(
                    "MLP baseline DataFrame must contain "
                    "split, seed and accuracy."
                )

            teacher = mlp_baseline[
                ["split", "seed", "accuracy"]
            ].rename(columns={"accuracy": "mlp_accuracy"})

            if teacher.duplicated(["split", "seed"]).any():
                raise ValueError("Duplicate MLP baseline entries.")

            result = result.drop(columns=["mlp_accuracy"])

            result = result.merge(
                teacher,
                on=["split", "seed"],
                how="left",
                validate="many_to_one"
            )

        else:
            raise TypeError(
                "mlp_baseline must be None, a scalar, "
                "or a DataFrame."
            )

        if result["mlp_accuracy"].notna().any():
            if not result["mlp_accuracy"].dropna().between(0, 1).all():
                raise ValueError("MLP accuracy must be between 0 and 1.")

    # Positive value means the fuzzy model is less accurate.
    result["accuracy_loss_vs_mlp"] = (
        result["mlp_accuracy"] - result["accuracy"]
    )

    return result


def aggregate_by_seed(df):
    """
    Aggregate repeated seeds for each model/configuration/split.

    Standard deviation is sample standard deviation.
    It remains NaN when only one seed is available.
    """

    df = validate_results(df)

    group_columns = [
        "model",
        "pruning_setting",
        "split"
    ]

    summary = (
        df.groupby(group_columns)[METRICS]
        .agg(["mean", "std", "count"])
        .reset_index()
    )

    summary.columns = [
        "_".join(str(x) for x in col if x != "")
        if isinstance(col, tuple) else col
        for col in summary.columns
    ]

    return summary


def identify_pareto_front(
    df,
    maximize=DEFAULT_MAXIMIZE,
    minimize=DEFAULT_MINIMIZE
):
    """
    Mark Pareto-efficient rows.

    Maximized and minimized objectives are treated equally.
    A row is dominated if another row is at least as good
    on every objective and strictly better on at least one.
    """

    if df.empty:
        raise ValueError("DataFrame is empty.")

    objectives = list(maximize) + list(minimize)

    missing = set(objectives) - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing Pareto objectives: {sorted(missing)}"
        )

    values = df[objectives].to_numpy(dtype=float)

    if not np.isfinite(values).all():
        raise ValueError("Pareto objectives must be finite.")

    # Convert every objective into a maximization objective.
    directions = np.array(
        [1] * len(maximize) + [-1] * len(minimize)
    )

    scores = values * directions

    efficient = np.ones(len(scores), dtype=bool)

    for i, candidate in enumerate(scores):

        dominates = (
            np.all(scores >= candidate, axis=1) &
            np.any(scores > candidate, axis=1)
        )

        if dominates.any():
            efficient[i] = False

    result = df.copy()
    result["pareto_efficient"] = efficient

    return result


def select_best_configuration(
    df,
    split="validation",
    min_accuracy=None,
    min_fidelity=None,
    min_coverage=None
):
    """
    Select the simplest configuration satisfying constraints.

    Selection is restricted to the requested split.
    Use validation for model selection; do not select on test data.

    Priority:
      1. Lowest rule count
      2. Lowest mean rule length
      3. Lowest fallback rate
      4. Highest accuracy
      5. Highest fidelity
    """

    df = validate_results(df)

    if split.lower() == "test":
        raise ValueError(
            "Do not select configurations using test-set results."
        )

    data = df[df["split"] == split].copy()

    if data.empty:
        raise ValueError(f"No results found for split: {split}")

    constraints = [
        ("accuracy", min_accuracy),
        ("fidelity", min_fidelity),
        ("coverage", min_coverage)
    ]

    for metric, threshold in constraints:

        if threshold is not None:

            if not 0 <= threshold <= 1:
                raise ValueError(
                    f"{metric} threshold must be between 0 and 1."
                )

            data = data[data[metric] >= threshold]

    if data.empty:
        raise ValueError(
            "No configuration satisfies the specified constraints."
        )

    data = data.sort_values(
        [
            "rule_count",
            "mean_rule_length",
            "fallback_rate",
            "accuracy",
            "fidelity"
        ],
        ascending=[True, True, True, False, False],
        kind="stable"
    )

    return data.iloc[0].copy()
