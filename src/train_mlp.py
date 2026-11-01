
from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd

from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)

# Project folders
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
ARTIFACTS_DIR = ROOT / "artifacts"
RESULTS_DIR = ROOT / "results"

ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42


def load_split(filename):
    """Load processed data and separate features from labels."""
    df = pd.read_csv(DATA_DIR / filename)

    row_ids = df["row_id"].to_numpy()
    y = df["true_label"].astype(int).to_numpy()

    # Metadata columns are not model input features.
    X = df.drop(columns=["row_id", "true_label"])

    return X, y, row_ids


def main():
    # Load training and validation data only.
    X_train, y_train, train_ids = load_split("train_processed.csv")
    X_val, y_val, val_ids = load_split("validation_processed.csv")

    if list(X_train.columns) != list(X_val.columns):
        raise ValueError("Training and validation feature columns do not match.")

    # Small MLP baseline; no extensive hyperparameter tuning.
    model = MLPClassifier(
        hidden_layer_sizes=(16, 8),
        activation="relu",
        solver="adam",
        alpha=0.001,
        learning_rate_init=0.001,
        max_iter=500,
        early_stopping=False,
        random_state=RANDOM_STATE,
    )

    # Train on training data only.
    model.fit(X_train, y_train)

    # Evaluate using the validation set.
    val_pred = model.predict(X_val)
    val_prob = model.predict_proba(X_val)[:, 1]

    metrics = {
        "model": "sklearn.neural_network.MLPClassifier",
        "architecture": [30, 16, 8, 2],
        "hidden_layer_sizes": [16, 8],
        "activation": "relu",
        "solver": "adam",
        "alpha": 0.001,
        "learning_rate_init": 0.001,
        "max_iter": 500,
        "random_state": RANDOM_STATE,
        "early_stopping": False,
        "actual_training_iterations": int(model.n_iter_),
        "converged_within_max_iter": bool(model.n_iter_ < 500),
        "training_rows": int(len(X_train)),
        "validation_rows": int(len(X_val)),
        "validation_accuracy": float(accuracy_score(y_val, val_pred)),
        "validation_precision": float(
            precision_score(y_val, val_pred, zero_division=0)
        ),
        "validation_recall": float(
            recall_score(y_val, val_pred, zero_division=0)
        ),
        "validation_f1": float(
            f1_score(y_val, val_pred, zero_division=0)
        ),
        "validation_roc_auc": float(roc_auc_score(y_val, val_prob)),
        "validation_confusion_matrix": confusion_matrix(
            y_val, val_pred, labels=[0, 1]
        ).tolist(),
        "class_labels": {"benign": 0, "malignant": 1},
        "test_set_used_for_selection": False,
    }

    # Save the trained model.
    joblib.dump(model, ARTIFACTS_DIR / "mlp_model.joblib")

    # Save validation predictions for analysis.
    predictions = pd.DataFrame({
        "row_id": val_ids,
        "true_label": y_val,
        "predicted_label": val_pred,
        "malignant_probability": val_prob,
    })
    predictions.to_csv(
        RESULTS_DIR / "mlp_validation_predictions.csv",
        index=False,
    )

    # Save settings and validation metrics.
    with open(
        RESULTS_DIR / "mlp_metrics.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(metrics, file, indent=2)

    print("\nMLP baseline training completed.")
    print(f"Training rows: {len(X_train)}")
    print(f"Validation rows: {len(X_val)}")
    print(f"Training iterations: {model.n_iter_}")
    print("\nValidation metrics:")

    for name in [
        "validation_accuracy",
        "validation_precision",
        "validation_recall",
        "validation_f1",
        "validation_roc_auc",
    ]:
        print(f"{name}: {metrics[name]:.4f}")

    print("\nSaved model: artifacts/mlp_model.joblib")
    print("Saved metrics: results/mlp_metrics.json")
    print("Saved predictions: results/mlp_validation_predictions.csv")


if __name__ == "__main__":
    main()
