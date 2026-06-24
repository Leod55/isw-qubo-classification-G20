# tests/test_preprocessing.py
import os
import json
import pandas as pd
import numpy as np
import pytest
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from qubo_project.preprocessing import fit_normalize


# ----------------------------------------------------------------------
# Fixture: tiny dataset for normalisation tests
# ----------------------------------------------------------------------
@pytest.fixture
def tiny_dataset_path(tmp_path):
    data = {
        "feat1": [1, 2, 0, 4, 5],
        "feat2": [0, 0, 0, 0, 0],          # all zeros → will be dropped
        "feat3": [1, 0, 3, 0, 5],
        "target": [0, 1, 0, 1, 0],
    }
    df = pd.DataFrame(data)
    path = tmp_path / "tiny_input.csv"
    df.to_csv(path, index=False)
    return path, tmp_path


# ----------------------------------------------------------------------
# Test 1: Preprocessing produces only numeric columns
# ----------------------------------------------------------------------
def test_preprocessing_produces_numeric_columns(tiny_dataset_path):
    input_path, tmp_path = tiny_dataset_path
    out_csv = tmp_path / "normalized.csv"
    out_json = tmp_path / "stats.json"

    fit_normalize(
        input_csv=str(input_path),
        target_column="target",
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
# Test 2: Preprocessing correctly drops columns with too few valid values
# ----------------------------------------------------------------------
def test_preprocessing_handles_missing_values(tmp_path):
    # Dataset: feat1 has all valid, feat2 all zeros, feat3 has only 1 valid out of 5 (20%)
    data = {
        "feat1": [1, 2, 3, 4, 5],
        "feat2": [0, 0, 0, 0, 0],          # 0% valid → dropped
        "feat3": [1, 0, 0, 0, 0],          # only first value is non‑zero → 20% valid → dropped
        "target": [0, 1, 0, 1, 0],
    }
    df = pd.DataFrame(data)
    input_path = tmp_path / "missing.csv"
    df.to_csv(input_path, index=False)

    out_csv = tmp_path / "normalized.csv"
    out_json = tmp_path / "stats.json"

    # minPercValid = 0.3 → only feat1 should survive
    fit_normalize(
        input_csv=str(input_path),
        target_column="target",
        normalized_csv=str(out_csv),
        outInitialRes_json=str(out_json),
        minPercValid=0.3,
    )

    with open(out_json) as f:
        stats = json.load(f)

    # Both feat2 and feat3 should be in dropped list
    assert "feat2" in stats["dropped_feature_names"]
    assert "feat3" in stats["dropped_feature_names"]
    assert stats["n_kept_features"] == 1  # only feat1


# ----------------------------------------------------------------------
# Test 3: Normalisation produces mean ≈ 0 and population std ≈ 1
# ----------------------------------------------------------------------
def test_preprocessing_normalization(tiny_dataset_path):
    input_path, tmp_path = tiny_dataset_path
    out_csv = tmp_path / "normalized.csv"
    out_json = tmp_path / "stats.json"

    fit_normalize(
        input_csv=str(input_path),
        target_column="target",
        normalized_csv=str(out_csv),
        outInitialRes_json=str(out_json),
        minPercValid=0.05,
    )

    df = pd.read_csv(out_csv)
    features = [c for c in df.columns if c != "target"]
    for col in features:
        mean = df[col].mean()
        std_pop = df[col].std(ddof=0)          # use population std to match StandardScaler
        assert abs(mean) < 1e-6, f"Mean of {col} is {mean}, expected ~0"
        assert abs(std_pop - 1.0) < 1e-6, f"Population std of {col} is {std_pop}, expected ~1"


# ----------------------------------------------------------------------
# Test 4: JSON contains all required fields
# ----------------------------------------------------------------------
def test_preprocessing_json_structure(tiny_dataset_path):
    input_path, tmp_path = tiny_dataset_path
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