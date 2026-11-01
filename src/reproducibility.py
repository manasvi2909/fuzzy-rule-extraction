
from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import yaml


DEFAULT_PACKAGES = [
    "numpy",
    "pandas",
    "scikit-learn",
    "matplotlib",
    "PyYAML",
    "pytest",
    "tabulate"
]


def _load_yaml(path):
    """Load a YAML configuration file."""

    path = Path(path)

    if not path.is_file():
        raise FileNotFoundError(f"Configuration not found: {path}")

    with path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file)

    if not isinstance(data, dict):
        raise ValueError("Configuration must contain a YAML mapping.")

    return data


def _load_json(path):
    """Load a JSON metadata file."""

    path = Path(path)

    if not path.is_file():
        raise FileNotFoundError(f"Metadata file not found: {path}")

    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    if not isinstance(data, dict):
        raise ValueError("Metadata must contain a JSON object.")

    return data


def _sha256(path):
    """Calculate the SHA-256 hash of a file."""

    digest = hashlib.sha256()

    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def _package_versions(packages):
    """Collect installed package versions."""

    versions = {}

    for package in packages:
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            versions[package] = None

    return versions


def _git_metadata(cwd):
    """Collect Git metadata when the project is in a Git repository."""

    def run_git(*args):
        try:
            result = subprocess.run(
                ["git", *args],
                cwd=cwd,
                capture_output=True,
                text=True,
                check=True
            )
            return result.stdout.strip()
        except (subprocess.CalledProcessError, FileNotFoundError):
            return None

    return {
        "commit": run_git("rev-parse", "HEAD"),
        "branch": run_git("branch", "--show-current"),
        "working_tree_clean": (
            run_git("status", "--porcelain") == ""
            if run_git("status", "--porcelain") is not None
            else None
        )
    }


def _configuration_checks(config, metadata):
    """Compare important configuration values with dataset metadata."""

    issues = []
    unresolved = []

    project = config.get("project", {})
    data = config.get("data", {})

    configured_dataset = data.get("dataset_name")
    metadata_dataset = metadata.get("dataset")

    configured_target = data.get("target_column")
    metadata_target = metadata.get("target")

    if configured_dataset is None:
        unresolved.append("data.dataset_name")
    elif metadata_dataset and configured_dataset != metadata_dataset:
        issues.append({
            "field": "data.dataset_name",
            "configured": configured_dataset,
            "metadata": metadata_dataset
        })

    if configured_target is None:
        unresolved.append("data.target_column")
    elif metadata_target and configured_target != metadata_target:
        issues.append({
            "field": "data.target_column",
            "configured": configured_target,
            "metadata": metadata_target
        })

    configured_seed = project.get("random_seed")
    metadata_seed = metadata.get("random_state")

    if (
        configured_seed is not None
        and metadata_seed is not None
        and configured_seed != metadata_seed
    ):
        issues.append({
            "field": "project.random_seed",
            "configured": configured_seed,
            "metadata": metadata_seed
        })

    return {
        "unresolved_configuration_fields": unresolved,
        "configuration_metadata_mismatches": issues
    }


def generate_manifest(
    config_path="config.yaml",
    metadata_path="artifacts/feature_metadata.json",
    output_path="results/reproducibility_manifest.json",
    input_paths=None,
    experiment_parameters=None,
    packages=None
):
    """
    Generate a machine-readable experiment reproducibility manifest.

    Paths are interpreted relative to the current working directory
    unless absolute paths are supplied.
    """

    config_path = Path(config_path).resolve()
    metadata_path = Path(metadata_path).resolve()
    output_path = Path(output_path).resolve()

    config = _load_yaml(config_path)
    metadata = _load_json(metadata_path)

    if input_paths is None:
        input_paths = [
            "data/train_processed.csv",
            "data/validation_processed.csv",
            "data/test_processed.csv",
            "results/mlp_train_predictions.csv",
            "results/mlp_validation_predictions.csv",
            "results/mlp_test_predictions.csv"
        ]

    tracked_paths = [config_path, metadata_path]

    for path in input_paths:
        path = Path(path)

        if not path.is_absolute():
            path = Path.cwd() / path

        if path.is_file():
            tracked_paths.append(path.resolve())

    # Remove duplicate paths while preserving order.
    tracked_paths = list(dict.fromkeys(tracked_paths))

    file_hashes = {}

    for path in tracked_paths:
        file_hashes[str(path)] = {
            "sha256": _sha256(path),
            "size_bytes": path.stat().st_size
        }

    project_root = config_path.parent

    checks = _configuration_checks(config, metadata)

    manifest = {
        "manifest_version": "1.0",
        "generated_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),

        "project": {
            "name": config.get("project", {}).get("name"),
            "random_seed": config.get("project", {}).get("random_seed")
        },

        "environment": {
            "python_version": sys.version,
            "python_executable": sys.executable,
            "platform": platform.platform(),
            "system": platform.system(),
            "machine": platform.machine(),
            "packages": _package_versions(
                packages if packages is not None else DEFAULT_PACKAGES
            )
        },

        "git": _git_metadata(project_root),

        "configuration": config,

        "dataset": {
            "name": metadata.get("dataset"),
            "uci_dataset_id": metadata.get("uci_dataset_id"),
            "source": metadata.get("source"),
            "target": metadata.get("target"),
            "class_labels": metadata.get("class_labels"),
            "feature_count": metadata.get("feature_count"),
            "feature_names_order": metadata.get("feature_names_order"),
            "random_state": metadata.get("random_state"),
            "split_sizes": metadata.get("split_sizes"),
            "class_counts": metadata.get("class_counts"),
            "preprocessing": metadata.get("preprocessing")
        },

        "experiment_parameters": experiment_parameters or {},

        "input_files": file_hashes,

        "configuration_checks": checks
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(
            manifest,
            file,
            indent=4,
            ensure_ascii=False,
            default=str
        )

    return manifest
