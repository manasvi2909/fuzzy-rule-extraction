
import json

import pytest
import yaml

from src.reproducibility import (
    generate_manifest,
    _sha256,
    _load_yaml,
    _load_json
)


@pytest.fixture
def project_files(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    (tmp_path / "artifacts").mkdir()
    (tmp_path / "data").mkdir()
    (tmp_path / "results").mkdir()

    config = {
        "project": {
            "name": "test-project",
            "random_seed": 42
        },
        "data": {
            "dataset_name": None,
            "target_column": None,
            "test_size": 0.2,
            "validation_size": 0.2
        },
        "model": {
            "type": "mlp",
            "random_seed": 42
        }
    }

    metadata = {
        "dataset": "UCI Breast Cancer Wisconsin (Diagnostic)",
        "uci_dataset_id": 17,
        "source": "https://example.com/dataset",
        "target": "Diagnosis",
        "class_labels": {
            "B": 0,
            "M": 1
        },
        "feature_count": 30,
        "feature_names_order": ["radius1", "texture1"],
        "random_state": 42,
        "split_sizes": {
            "train": 398,
            "validation": 85,
            "test": 86
        },
        "class_counts": {
            "train": {"0": 250, "1": 148}
        },
        "preprocessing": {
            "imputer": "median",
            "scaler": "StandardScaler",
            "fit_on": "training split only"
        }
    }

    with open("config.yaml", "w") as file:
        yaml.safe_dump(config, file)

    with open("artifacts/feature_metadata.json", "w") as file:
        json.dump(metadata, file)

    (tmp_path / "data/train_processed.csv").write_text(
        "row_id,true_label,feature1\n1,0,0.5\n"
    )

    return tmp_path


def test_generate_manifest(project_files):
    manifest = generate_manifest(packages=[])

    assert manifest["manifest_version"] == "1.0"
    assert manifest["project"]["name"] == "test-project"
    assert manifest["dataset"]["feature_count"] == 30


def test_manifest_output_exists(project_files):
    generate_manifest(packages=[])

    output = project_files / "results/reproducibility_manifest.json"

    assert output.exists()

    with output.open() as file:
        data = json.load(file)

    assert data["manifest_version"] == "1.0"


def test_dataset_metadata_is_recorded(project_files):
    manifest = generate_manifest(packages=[])

    assert manifest["dataset"]["name"] == (
        "UCI Breast Cancer Wisconsin (Diagnostic)"
    )
    assert manifest["dataset"]["target"] == "Diagnosis"
    assert manifest["dataset"]["split_sizes"]["train"] == 398


def test_unresolved_configuration_is_reported(project_files):
    manifest = generate_manifest(packages=[])

    unresolved = manifest[
        "configuration_checks"
    ]["unresolved_configuration_fields"]

    assert "data.dataset_name" in unresolved
    assert "data.target_column" in unresolved


def test_configuration_mismatch_is_reported(project_files):
    config = yaml.safe_load(
        (project_files / "config.yaml").read_text()
    )

    config["data"]["dataset_name"] = "Different Dataset"

    (project_files / "config.yaml").write_text(
        yaml.safe_dump(config)
    )

    manifest = generate_manifest(packages=[])

    issues = manifest[
        "configuration_checks"
    ]["configuration_metadata_mismatches"]

    assert any(
        issue["field"] == "data.dataset_name"
        for issue in issues
    )


def test_input_file_hash_is_recorded(project_files):
    manifest = generate_manifest(packages=[])

    hashes = manifest["input_files"]

    assert any(
        path.endswith("train_processed.csv")
        for path in hashes
    )


def test_hash_changes_when_file_changes(project_files):
    path = project_files / "data/train_processed.csv"

    first = _sha256(path)

    path.write_text("different content")

    second = _sha256(path)

    assert first != second


def test_custom_input_paths(project_files):
    manifest = generate_manifest(
        input_paths=["data/train_processed.csv"],
        packages=[]
    )

    paths = list(manifest["input_files"])

    assert any(path.endswith("train_processed.csv") for path in paths)


def test_experiment_parameters_are_recorded(project_files):
    params = {
        "membership_terms": ["Low", "Medium", "High"],
        "pruning_top_n": [10, 25, 50]
    }

    manifest = generate_manifest(
        experiment_parameters=params,
        packages=[]
    )

    assert manifest["experiment_parameters"] == params


def test_package_versions_are_recorded(project_files):
    manifest = generate_manifest(
        packages=["numpy", "package-that-does-not-exist-123"]
    )

    assert "numpy" in manifest["environment"]["packages"]
    assert (
        manifest["environment"]["packages"][
            "package-that-does-not-exist-123"
        ] is None
    )


def test_missing_config_raises_error(project_files):
    with pytest.raises(FileNotFoundError):
        generate_manifest(
            config_path="missing.yaml",
            packages=[]
        )


def test_missing_metadata_raises_error(project_files):
    with pytest.raises(FileNotFoundError):
        generate_manifest(
            metadata_path="missing.json",
            packages=[]
        )


def test_invalid_yaml_raises_error(project_files):
    (project_files / "invalid.yaml").write_text("- item1\n- item2\n")

    with pytest.raises(ValueError):
        _load_yaml("invalid.yaml")


def test_invalid_json_raises_error(project_files):
    (project_files / "invalid.json").write_text("{invalid}")

    with pytest.raises(json.JSONDecodeError):
        _load_json("invalid.json")


def test_timestamp_is_recorded(project_files):
    manifest = generate_manifest(packages=[])

    assert manifest["generated_at_utc"].endswith("+00:00")
