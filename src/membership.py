"""
Fuzzy membership functions used by the rule extraction and inference stages.

The project uses three linguistic terms per feature: Low, Medium and High.
Membership parameters are fitted on the processed TRAINING split only and
then reused unchanged for validation/test data.
"""

from math import isfinite
from typing import Mapping

import numpy as np
import pandas as pd


def triangular_membership(x, a, b, c):
    """
    Return the membership degree of x in a triangular fuzzy set.

    a, b, c define the triangle (a <= b <= c). The cases a == b and
    b == c are supported so this function can also represent left and
    right shoulder sets.

    Returns a float in [0, 1].
    """
    values = (x, a, b, c)
    if not all(isinstance(v, (int, float, np.number)) for v in values):
        raise TypeError("x, a, b and c must be numeric.")

    x, a, b, c = map(float, values)

    if not all(isfinite(v) for v in (x, a, b, c)):
        raise ValueError("x and membership parameters must be finite.")

    if not a <= b <= c:
        raise ValueError("Triangle parameters must satisfy a <= b <= c.")

    # A fully degenerate triangle represents a crisp singleton.
    if a == b == c:
        return 1.0 if x == a else 0.0

    # Left shoulder: membership is 1 at and below a, then decreases.
    if a == b:
        if x <= b:
            return 1.0
        if x >= c:
            return 0.0
        return (c - x) / (c - b)

    # Right shoulder: membership increases to 1 at b and stays 1.
    if b == c:
        if x >= b:
            return 1.0
        if x <= a:
            return 0.0
        return (x - a) / (b - a)

    if x <= a or x >= c:
        return 0.0
    if x == b:
        return 1.0
    if x < b:
        return (x - a) / (b - a)
    return (c - x) / (c - b)


def trapezoidal_membership(x, a, b, c, d):
    """
    Return the membership degree in a trapezoidal fuzzy set.

    Parameters must satisfy a <= b <= c <= d. Equal endpoints allow
    left/right shoulders and triangular shapes.
    """
    values = (x, a, b, c, d)
    if not all(isinstance(v, (int, float, np.number)) for v in values):
        raise TypeError("x, a, b, c and d must be numeric.")

    x, a, b, c, d = map(float, values)
    if not all(isfinite(v) for v in (x, a, b, c, d)):
        raise ValueError("x and membership parameters must be finite.")
    if not a <= b <= c <= d:
        raise ValueError("Parameters must satisfy a <= b <= c <= d.")

    if x < a or x > d:
        return 0.0
    if b <= x <= c:
        return 1.0
    if x < b:
        if a == b:
            return 1.0
        return (x - a) / (b - a)
    if c == d:
        return 1.0
    return (d - x) / (d - c)


def fit_membership_functions(X_train):
    """
    Fit Low/Medium/High triangular sets for every feature.

    X_train must contain the already-preprocessed training features.
    For each feature, its training minimum, median and maximum define:
      Low:    (minimum, minimum, median)   -- left shoulder
      Medium: (minimum, median, maximum)   -- triangle
      High:   (median, maximum, maximum)   -- right shoulder

    Returns:
        {
            feature_name: {
                "Low": (a, b, c),
                "Medium": (a, b, c),
                "High": (a, b, c)
            }
        }

    Fit this on the training split only. Do not refit on validation or test.
    """
    if not isinstance(X_train, pd.DataFrame):
        raise TypeError("X_train must be a pandas DataFrame.")
    if X_train.empty:
        raise ValueError("X_train cannot be empty.")
    if not X_train.columns.is_unique:
        raise ValueError("Feature column names must be unique.")

    membership_functions = {}

    for feature in X_train.columns:
        column = pd.to_numeric(X_train[feature], errors="raise").dropna()
        if column.empty:
            raise ValueError(f"Feature {feature!r} contains no valid values.")

        values = column.to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError(f"Feature {feature!r} contains non-finite values.")

        minimum = float(np.min(values))
        median = float(np.median(values))
        maximum = float(np.max(values))

        membership_functions[feature] = {
            "Low": (minimum, minimum, median),
            "Medium": (minimum, median, maximum),
            "High": (median, maximum, maximum),
        }

    return membership_functions


def fuzzify_sample(sample, membership_functions):
    """
    Convert one crisp sample into membership degrees.

    sample may be a mapping/Series: {feature_name: numeric_value}.
    The returned structure matches src.extract_rules.generate_candidate_rule.
    """
    if not isinstance(sample, Mapping) and not isinstance(sample, pd.Series):
        raise TypeError("sample must be a mapping or pandas Series.")

    if not isinstance(membership_functions, Mapping) or not membership_functions:
        raise ValueError("membership_functions must be a non-empty mapping.")

    sample_keys = set(sample.index) if isinstance(sample, pd.Series) else set(sample.keys())
    missing = set(membership_functions) - sample_keys
    if missing:
        raise ValueError(f"Sample is missing features: {sorted(missing)}")

    degrees = {}
    for feature, terms in membership_functions.items():
        value = sample[feature]
        if not isinstance(value, (int, float, np.number)):
            raise TypeError(f"Value for {feature!r} must be numeric.")
        if not isfinite(float(value)):
            raise ValueError(f"Value for {feature!r} must be finite.")

        degrees[feature] = {
            term: triangular_membership(value, *parameters)
            for term, parameters in terms.items()
        }

    return degrees


def fuzzify_dataframe(X, membership_functions):
    """Fuzzify each row of a DataFrame; returns a list of sample dictionaries."""
    if not isinstance(X, pd.DataFrame):
        raise TypeError("X must be a pandas DataFrame.")

    missing = set(membership_functions) - set(X.columns)
    if missing:
        raise ValueError(f"DataFrame is missing features: {sorted(missing)}")

    return [
        fuzzify_sample(row, membership_functions)
        for _, row in X.iterrows()
    ]
