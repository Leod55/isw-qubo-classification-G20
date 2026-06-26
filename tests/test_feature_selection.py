# tests/test_feature_selection.py
import json
import pandas as pd
import pytest
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from qubo_project.feature_selection import select_features


@pytest.fixture(scope="module")
def sample_dataset_path():
    base_dir = Path(__file__).parent.parent
    csv_path = base_dir / "data" / "sample_test_dataset.csv"
    if not csv_path.exists():
        pytest.fail(f"Required test dataset not found: {csv_path}")
    return csv_path


# ----------------------------------------------------------------------
# Test 1: Feature selection produces a binary vector
# ----------------------------------------------------------------------
def test_feature_selection_produces_binary_vector(sample_dataset_path, tmp_path):
    # First, preprocess the sample to get a normalised dataset
    from qubo_project.preprocessing import fit_normalize

    in_csv = sample_dataset_path
    norm_csv = tmp_path / "normalized.csv"
    norm_json = tmp_path / "preproc.json"

    fit_normalize(
        input_csv=str(in_csv),
        target_column="target",
        normalized_csv=str(norm_csv),
        outInitialRes_json=str(norm_json),
        minPercValid=0.05,
    )

    # Now run feature selection
    out_train = tmp_path / "train_reduced.csv"
    out_test = tmp_path / "test_reduced.csv"
    out_opt = tmp_path / "optimizations.csv"
    out_json = tmp_path / "fs_stats.json"

    select_features(
        normalized_csv=str(norm_csv),
        reducedTrain_csv=str(out_train),
        reducedTest_csv=str(out_test),
        output_ottim_csv=str(out_opt),
        output_json=str(out_json),
        target_column="target",
        percTest=0.30,
        percSelected=0.20,
        allowance=1,
        seed=42,
        alpha_computations=20,
    )

    # Check JSON contains selected_vector
    with open(out_json) as f:
        stats = json.load(f)
    assert "selected_vector" in stats
    assert isinstance(stats["selected_vector"], list)
    assert all(v in [0, 1] for v in stats["selected_vector"])


# ----------------------------------------------------------------------
# Test 2: Number of selected features is within allowance
# ----------------------------------------------------------------------
def test_feature_selection_count_within_allowance(sample_dataset_path, tmp_path):
    from qubo_project.preprocessing import fit_normalize

    # Preprocess
    norm_csv = tmp_path / "normalized.csv"
    norm_json = tmp_path / "preproc.json"
    fit_normalize(
        input_csv=str(sample_dataset_path),
        target_column="target",
        normalized_csv=str(norm_csv),
        outInitialRes_json=str(norm_json),
        minPercValid=0.05,
    )

    # Run FS
    out_train = tmp_path / "train_reduced.csv"
    out_test = tmp_path / "test_reduced.csv"
    out_opt = tmp_path / "optimizations.csv"
    out_json = tmp_path / "fs_stats.json"

    perc_selected = 0.20
    allowance = 1

    select_features(
        normalized_csv=str(norm_csv),
        reducedTrain_csv=str(out_train),
        reducedTest_csv=str(out_test),
        output_ottim_csv=str(out_opt),
        output_json=str(out_json),
        target_column="target",
        percTest=0.30,
        percSelected=perc_selected,
        allowance=allowance,
        seed=42,
        alpha_computations=20,
    )

    with open(out_json) as f:
        stats = json.load(f)

    n_features = stats["n_features"]
    K_target = int(round(perc_selected * n_features))
    n_selected = stats["n_selected"]

    assert K_target - allowance <= n_selected <= K_target + allowance


# ----------------------------------------------------------------------
# Test 3: JSON contains all required fields
# ----------------------------------------------------------------------
def test_feature_selection_json_structure(sample_dataset_path, tmp_path):
    from qubo_project.preprocessing import fit_normalize

    norm_csv = tmp_path / "normalized.csv"
    norm_json = tmp_path / "preproc.json"
    fit_normalize(
        input_csv=str(sample_dataset_path),
        target_column="target",
        normalized_csv=str(norm_csv),
        outInitialRes_json=str(norm_json),
        minPercValid=0.05,
    )

    out_train = tmp_path / "train_reduced.csv"
    out_test = tmp_path / "test_reduced.csv"
    out_opt = tmp_path / "optimizations.csv"
    out_json = tmp_path / "fs_stats.json"

    select_features(
        normalized_csv=str(norm_csv),
        reducedTrain_csv=str(out_train),
        reducedTest_csv=str(out_test),
        output_ottim_csv=str(out_opt),
        output_json=str(out_json),
        target_column="target",
        percTest=0.30,
        percSelected=0.20,
        allowance=1,
        seed=42,
        alpha_computations=20,
    )

    with open(out_json) as f:
        stats = json.load(f)

    required_keys = [
        "n_features",
        "target_ratio",
        "target_k",
        "allowance",
        "n_selected",
        "alpha",
        "selected_vector",
        "selected_feature_names",
        "algorithm",
        "seed",
        "alpha_computations",
        "percTest",
        "training_dataset_size",
        "test_dataset_size",
        "q_matrix_creation_time",
        "mean_optimization_time",
        "std_dev_optimization_time",
    ]
    for key in required_keys:
        assert key in stats, f"Missing key: {key}"


# ----------------------------------------------------------------------
# Test 4: Reduced datasets have only selected features + target
# ----------------------------------------------------------------------
def test_feature_selection_reduced_datasets(sample_dataset_path, tmp_path):
    from qubo_project.preprocessing import fit_normalize

    norm_csv = tmp_path / "normalized.csv"
    norm_json = tmp_path / "preproc.json"
    fit_normalize(
        input_csv=str(sample_dataset_path),
        target_column="target",
        normalized_csv=str(norm_csv),
        outInitialRes_json=str(norm_json),
        minPercValid=0.05,
    )

    out_train = tmp_path / "train_reduced.csv"
    out_test = tmp_path / "test_reduced.csv"
    out_opt = tmp_path / "optimizations.csv"
    out_json = tmp_path / "fs_stats.json"

    select_features(
        normalized_csv=str(norm_csv),
        reducedTrain_csv=str(out_train),
        reducedTest_csv=str(out_test),
        output_ottim_csv=str(out_opt),
        output_json=str(out_json),
        target_column="target",
        percTest=0.30,
        percSelected=0.20,
        allowance=1,
        seed=42,
        alpha_computations=20,
    )

    with open(out_json) as f:
        stats = json.load(f)

    selected = stats["selected_feature_names"]

    # Check training set
    df_train = pd.read_csv(out_train)
    assert set(df_train.columns) == set(selected + ["target"])

    # Check test set
    df_test = pd.read_csv(out_test)
    assert set(df_test.columns) == set(selected + ["target"])