
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
METADATA_PATH = ROOT / "artifacts" / "feature_metadata.json"
DATA_DIR = ROOT / "data"

with open(METADATA_PATH, "r", encoding="utf-8") as f:
    metadata = json.load(f)

feature_names = metadata["feature_names_order"]
processed_ranges = {}

for split in ("train", "validation", "test"):
    path = DATA_DIR / f"{split}_processed.csv"
    df = pd.read_csv(path)

    # Keep the exact feature order recorded in metadata.
    missing = set(feature_names) - set(df.columns)
    if missing:
        raise ValueError(f"{split}: missing features: {sorted(missing)}")

    X = df.loc[:, feature_names]

    processed_ranges[split] = {
        name: {
            "min": float(X[name].min()),
            "max": float(X[name].max()),
            "dtype": str(X[name].dtype),
        }
        for name in feature_names
    }

metadata["processed_feature_ranges"] = processed_ranges
metadata["processed_ranges_note"] = (
    "Ranges computed from processed CSV files. "
    "The scaler was fitted on training data only. "
    "Training ranges are the preferred reference for defining "
    "membership functions in standardized input space."
)

with open(METADATA_PATH, "w", encoding="utf-8") as f:
    json.dump(metadata, f, indent=2)

print("Metadata updated successfully.")
print("Feature count:", len(feature_names))
for split in ("train", "validation", "test"):
    print(
        f"{split}: {len(processed_ranges[split])} feature ranges"
    )
