
from pathlib import Path
import json

import joblib
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS_DIR = ROOT / "artifacts"
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"

MODEL_PATH = ARTIFACTS_DIR / "mlp_model.joblib"
METADATA_PATH = ARTIFACTS_DIR / "feature_metadata.json"

SPLITS = ("train", "validation", "test")


def load_predictor():
    """Load the trained MLP and its expected feature order."""
    model = joblib.load(MODEL_PATH)

    with open(METADATA_PATH, "r", encoding="utf-8") as file:
        metadata = json.load(file)

    feature_names = metadata["feature_names_order"]

    if model.n_features_in_ != len(feature_names):
        raise ValueError(
            "Model feature count does not match metadata."
        )

    return model, feature_names


def predict_processed(X_processed):
    """
    Predict from an already-preprocessed feature matrix.

    Parameters
    ----------
    X_processed : pandas.DataFrame
        Standardized feature values. Must contain the expected
        30 feature columns. Column order may differ.

    Returns
    -------
    pandas.DataFrame
        Predicted class and probabilities for benign and malignant.
    """
    model, feature_names = load_predictor()

    if not isinstance(X_processed, pd.DataFrame):
        raise TypeError("X_processed must be a pandas DataFrame.")

    missing = set(feature_names) - set(X_processed.columns)
    extra = set(X_processed.columns) - set(feature_names)

    if missing or extra:
        raise ValueError(
            f"Feature mismatch. Missing: {sorted(missing)}; "
            f"unexpected: {sorted(extra)}"
        )

    # Enforce the exact feature order used by the model.
    X = X_processed.loc[:, feature_names]

    if X.isna().any().any():
        raise ValueError(
            "Processed features contain missing values."
        )

    probabilities = model.predict_proba(X)
    predicted_labels = model.predict(X)

    return pd.DataFrame({
        "predicted_label": predicted_labels.astype(int),
        "benign_probability": probabilities[:, 0],
        "malignant_probability": probabilities[:, 1],
    })


def export_split_predictions(split_name):
    """Export predictions while preserving original row identifiers."""
    if split_name not in SPLITS:
        raise ValueError(
            f"split_name must be one of {SPLITS}"
        )

    input_path = DATA_DIR / f"{split_name}_processed.csv"
    df = pd.read_csv(input_path)

    if "row_id" not in df or "true_label" not in df:
        raise ValueError(
            f"{input_path.name} must contain row_id and true_label."
        )

    row_ids = df["row_id"].copy()
    true_labels = df["true_label"].astype(int).copy()

    X_processed = df.drop(columns=["row_id", "true_label"])
    predictions = predict_processed(X_processed)

    # Keep identifiers first to support safe alignment across modules.
    output = pd.DataFrame({
        "row_id": row_ids,
        "true_label": true_labels,
    })

    output = pd.concat(
        [output.reset_index(drop=True),
         predictions.reset_index(drop=True)],
        axis=1,
    )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = RESULTS_DIR / f"mlp_{split_name}_predictions.csv"
    output.to_csv(output_path, index=False)

    print(f"{split_name}: {len(output)} predictions saved to {output_path}")


def main():
    for split_name in SPLITS:
        export_split_predictions(split_name)

    print("\nPrediction export completed.")


if __name__ == "__main__":
    main()
