# tests/test_preprocessing.py
import json
import pandas as pd
import pytest
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from qubo_project.preprocessing import fit_normalize


# ----------------------------------------------------------------------
# Fixture: load the mandatory test dataset from data/sample_test_dataset.csv
# ----------------------------------------------------------------------
@pytest.fixture(scope="module")
def sample_dataset_path():
    base_dir = Path(__file__).parent.parent
    csv_path = base_dir / "data" / "sample_test_dataset.csv"
    if not csv_path.exists():
        pytest.fail(
            f"Required test dataset not found: {csv_path}\n"
            "Please create it by extracting a subset from the provided dataset."
        )
    return csv_path


# ----------------------------------------------------------------------
# Test 1: Preprocessing produces only numeric columns
# ----------------------------------------------------------------------
def test_preprocessing_produces_numeric_columns(sample_dataset_path, tmp_path):
    input_path = sample_dataset_path
    out_csv = tmp_path / "normalized.csv"
    out_json = tmp_path / "stats.json"

    fit_normalize(
        input_csv=str(input_path),
        target_column="target",          # adjust if your sample uses a different name
        normalized_csv=str(out_csv),
        outInitialRes_json=str(out_json),
        minPercValid=0.05,
    )

    df = pd.read_csv(out_csv)
    for col in df.columns:
        if col != "target":
            assert pd.api.types.is_float_dtype(df[col]), f"Column {col} is not float!"
    assert pd.api.types.is_integer_dtype(df["target"]) or pd.api.types.is_float_dtype(df["target"])


# ----------------------------------------------------------------------
# Test 2: Preprocessing handles missing / zero columns gracefully
# (checks that the function runs without error and JSON reports
#  reasonable values, even if no columns are dropped)
# ----------------------------------------------------------------------
def test_preprocessing_handles_missing_values(sample_dataset_path, tmp_path):
    input_path = sample_dataset_path
    out_csv = tmp_path / "normalized.csv"
    out_json = tmp_path / "stats.json"

    # Use a stricter threshold to force at least some columns to be dropped
    fit_normalize(
        input_csv=str(input_path),
        target_column="target",
        normalized_csv=str(out_csv),
        outInitialRes_json=str(out_json),
        minPercValid=0.8,  # High threshold – will drop most columns
    )

    with open(out_json) as f:
        stats = json.load(f)

    assert "dropped_feature_names" in stats
    assert isinstance(stats["dropped_feature_names"], list)
    assert stats["n_kept_features"] <= stats["n_input_features"]


# ----------------------------------------------------------------------
# Test 3: Normalisation produces mean ≈ 0 and std = 0 for constant,
#          std ≈ 1 for non‑constant columns.
# ----------------------------------------------------------------------
def test_preprocessing_normalization(sample_dataset_path, tmp_path):
    input_path = sample_dataset_path

    # ---- Read original sample to identify constant features ----
    original_df = pd.read_csv(input_path)
    original_features = [c for c in original_df.columns if c != "target"]
    constant_features = [col for col in original_features if original_df[col].std(ddof=0) == 0]

    # ---- Run preprocessing ----
    out_csv = tmp_path / "normalized.csv"
    out_json = tmp_path / "stats.json"
    fit_normalize(
        input_csv=str(input_path),
        target_column="target",
        normalized_csv=str(out_csv),
        outInitialRes_json=str(out_json),
        minPercValid=0.05,
    )

    # ---- Check normalised output ----
    df = pd.read_csv(out_csv)
    features = [c for c in df.columns if c != "target"]
    if not features:
        pytest.skip("No features left after preprocessing – cannot test normalisation.")

    for col in features:
        mean = df[col].mean()
        std_pop = df[col].std(ddof=0)   # population std to match StandardScaler

        # Mean must always be ~0
        assert abs(mean) < 1e-6, f"Mean of {col} is {mean}, expected ~0"

        if col in constant_features:
            # Constant column → after normalisation, all zeros → std = 0
            assert abs(std_pop - 0.0) < 1e-6, f"Constant column {col} has std {std_pop}, expected 0"
        else:
            # Non‑constant → std must be ~1
            assert abs(std_pop - 1.0) < 1e-6, f"Non‑constant column {col} has std {std_pop}, expected ~1"


# ----------------------------------------------------------------------
# Test 4: JSON contains all required fields
# ----------------------------------------------------------------------
def test_preprocessing_json_structure(sample_dataset_path, tmp_path):
    input_path = sample_dataset_path
    out_csv = tmp_path / "normalized.csv"
    out_json = tmp_path / "stats.json"

    fit_normalize(
        input_csv=str(input_path),
        target_column="target",
        normalized_csv=str(out_csv),
        outInitialRes_json=str(out_json),
        minPercValid=0.05,
    )

    with open(out_json) as f:
        stats = json.load(f)

    required_keys = [
        "n_input_features",
        "n_kept_features",
        "dataset_size",
        "dataset_input_time",
        "dataset_processing_time",
        "dropped_feature_names",
    ]
    for key in required_keys:
        assert key in stats, f"Missing key: {key}"
    assert isinstance(stats["dropped_feature_names"], list)