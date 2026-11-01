
import numpy as np
import pandas as pd


class MembershipFunctionBank:
    """
    Training-fitted triangular membership functions.

    Each feature receives three linguistic terms:
        Low, Medium, High

    The minimum and maximum are learned from training data.
    The midpoint is calculated as (minimum + maximum) / 2.

    Low and High are shoulder functions; Medium is triangular.
    """

    TERMS = ("Low", "Medium", "High")

    def __init__(self):
        self.feature_ranges = {}
        self.feature_names = []
        self.fitted = False

    def fit(self, X):
        """
        Fit membership-function parameters using training data only.

        Parameters:
            X: pandas DataFrame of numeric features.
        """
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a pandas DataFrame.")

        if X.empty:
            raise ValueError("Training data cannot be empty.")

        if X.columns.has_duplicates:
            raise ValueError("Feature names must be unique.")

        if not all(pd.api.types.is_numeric_dtype(t) for t in X.dtypes):
            raise TypeError("All features must be numeric.")

        values = X.to_numpy(dtype=float)

        if not np.isfinite(values).all():
            raise ValueError("Training data must contain only finite values.")

        self.feature_names = list(X.columns)
        self.feature_ranges = {}

        for feature in self.feature_names:
            low = float(X[feature].min())
            high = float(X[feature].max())
            mid = (low + high) / 2.0

            self.feature_ranges[feature] = {
                "Low": low,
                "Medium": mid,
                "High": high
            }

        self.fitted = True
        return self

    @staticmethod
    def _calculate_membership(x, low, mid, high):
        """Calculate Low, Medium and High membership degrees."""

        # Handle constant-valued training features.
        if low == high:
            if x < low:
                return {"Low": 1.0, "Medium": 0.0, "High": 0.0}
            if x > high:
                return {"Low": 0.0, "Medium": 0.0, "High": 1.0}
            return {"Low": 0.0, "Medium": 1.0, "High": 0.0}

        # Low: left shoulder.
        if x <= low:
            low_degree = 1.0
        elif x >= mid:
            low_degree = 0.0
        else:
            low_degree = (mid - x) / (mid - low)

        # Medium: triangular membership.
        if x <= low or x >= high:
            medium_degree = 0.0
        elif x <= mid:
            medium_degree = (x - low) / (mid - low)
        else:
            medium_degree = (high - x) / (high - mid)

        # High: right shoulder.
        if x <= mid:
            high_degree = 0.0
        elif x >= high:
            high_degree = 1.0
        else:
            high_degree = (x - mid) / (high - mid)

        return {
            "Low": float(np.clip(low_degree, 0.0, 1.0)),
            "Medium": float(np.clip(medium_degree, 0.0, 1.0)),
            "High": float(np.clip(high_degree, 0.0, 1.0))
        }

    def fuzzify_sample(self, sample):
        """
        Convert one sample into membership degrees.

        Accepts a pandas Series or a dictionary.
        """
        if not self.fitted:
            raise RuntimeError("Fit membership functions before fuzzification.")

        if isinstance(sample, pd.Series):
            sample = sample.to_dict()

        if not isinstance(sample, dict):
            raise TypeError("Sample must be a pandas Series or dictionary.")

        if set(sample.keys()) != set(self.feature_names):
            raise ValueError("Sample features do not match fitted features.")

        result = {}

        for feature in self.feature_names:
            try:
                value = float(sample[feature])
            except (TypeError, ValueError):
                raise ValueError(f"Feature {feature} must be numeric.")

            if not np.isfinite(value):
                raise ValueError(f"Feature {feature} must be finite.")

            params = self.feature_ranges[feature]

            result[feature] = self._calculate_membership(
                value,
                params["Low"],
                params["Medium"],
                params["High"]
            )

        return result

    def fuzzify_dataset(self, X):
        """Convert a DataFrame into a list of membership dictionaries."""
        if not self.fitted:
            raise RuntimeError("Fit membership functions before fuzzification.")

        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a pandas DataFrame.")

        if list(X.columns) != self.feature_names:
            raise ValueError("Dataset features or feature order do not match.")

        return [
            self.fuzzify_sample(row)
            for _, row in X.iterrows()
        ]
