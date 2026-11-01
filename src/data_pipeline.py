
from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd

from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


# Project folders
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
ARTIFACTS_DIR = ROOT / "artifacts"

DATA_DIR.mkdir(parents=True, exist_ok=True)
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42


def load_dataset():
    """Download and prepare the UCI Breast Cancer Wisconsin dataset."""
    dataset = fetch_ucirepo(id=17)

    X = dataset.data.features.copy()
    y = dataset.data.targets.iloc[:, 0].copy()

    # Remove the patient/sample ID if present.
    # IDs are identifiers, not predictive features.
    X = X.drop(columns=["ID", "id"], errors="ignore")

    # Encode diagnosis: Benign = 0, Malignant = 1.
    labels = y.astype(str).str.strip().str.upper()
    y = labels.map({"B": 0, "M": 1})

    if y.isna().any():
        raise ValueError(
            f"Unexpected target labels: {labels[y.isna()].unique().tolist()}"
        )

    # Ensure all input features are numeric.
    X = X.apply(pd.to_numeric, errors="raise")
    y = y.astype(int).reset_index(drop=True)
    X = X.reset_index(drop=True)

    if not X.columns.is_unique:
        raise ValueError("Feature column names must be unique.")

    return X, y


def add_row_ids(X, y, row_ids):
    """Keep stable source-row IDs and labels alongside the features."""
    result = X.iloc[row_ids].reset_index(drop=True).copy()
    result.insert(0, "true_label", y.iloc[row_ids].to_numpy())
    result.insert(0, "row_id", row_ids)
    return result


def main():
    X, y = load_dataset()
    row_ids = np.arange(len(X))

    # First split: 70% train and 30% temporary.
    train_ids, temp_ids = train_test_split(
        row_ids,
        test_size=0.30,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    # Second split: divide temporary data equally into validation and test.
    val_ids, test_ids = train_test_split(
        temp_ids,
        test_size=0.50,
        random_state=RANDOM_STATE,
        stratify=y.iloc[temp_ids],
    )

    splits = {
        "train": train_ids,
        "validation": val_ids,
        "test": test_ids,
    }

    # Save the unscaled splits for interpretation and inspection.
    for split_name, ids in splits.items():
        raw = add_row_ids(X, y, ids)
        raw.to_csv(DATA_DIR / f"{split_name}_raw.csv", index=False)

    # Fit the preprocessing pipeline ONLY on training features.
    preprocessor = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    preprocessor.fit(X.iloc[train_ids])

    # Save the fitted pipeline so the same transformation can be reused.
    joblib.dump(
        preprocessor,
        ARTIFACTS_DIR / "preprocessor.joblib",
    )

    # Apply the fitted pipeline to each split.
    for split_name, ids in splits.items():
        transformed = preprocessor.transform(X.iloc[ids])

        processed = pd.DataFrame(
            transformed,
            columns=X.columns,
        )
        processed.insert(0, "true_label", y.iloc[ids].to_numpy())
        processed.insert(0, "row_id", ids)

        processed.to_csv(
            DATA_DIR / f"{split_name}_processed.csv",
            index=False,
        )

    # Record dataset details, split IDs, and preprocessing configuration.
    metadata = {
        "dataset": "UCI Breast Cancer Wisconsin (Diagnostic)",
        "uci_dataset_id": 17,
        "source": (
            "https://archive.ics.uci.edu/dataset/17/"
            "breast+cancer+wisconsin+diagnostic"
        ),
        "random_state": RANDOM_STATE,
        "target": "Diagnosis",
        "class_labels": {"B": 0, "M": 1},
        "feature_count": len(X.columns),
        "feature_names_order": X.columns.tolist(),
        "feature_types": {
            column: str(X[column].dtype)
            for column in X.columns
        },
        "preprocessing": {
            "imputer": "SimpleImputer(strategy='median')",
            "scaler": "StandardScaler",
            "fit_on": "training split only",
            "encoding": "Benign=0, Malignant=1",
            "identifier_columns_removed": ["ID", "id"],
        },
        "split_sizes": {
            name: int(len(ids))
            for name, ids in splits.items()
        },
        "class_counts": {
            name: {
                str(int(label)): int(count)
                for label, count in
                y.iloc[ids].value_counts().sort_index().items()
            }
            for name, ids in splits.items()
        },
        "split_row_ids": {
            name: ids.tolist()
            for name, ids in splits.items()
        },
        "training_feature_ranges": {
            column: {
                "min": float(X.iloc[train_ids][column].min()),
                "max": float(X.iloc[train_ids][column].max()),
            }
            for column in X.columns
        },
    }

    with open(
        ARTIFACTS_DIR / "feature_metadata.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(metadata, file, indent=2)

    print("Dataset preprocessing completed successfully.")
    print(f"Total rows: {len(X)}")
    print(f"Input features: {len(X.columns)}")

    for name, ids in splits.items():
        print(f"{name}: {len(ids)} rows")

    print(f"Outputs saved under: {ROOT}")


if __name__ == "__main__":
    main()
