# tests/test_model.py
import json
import pandas as pd
import joblib
import pytest
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from qubo_project.model import train, predict
from qubo_project.preprocessing import fit_normalize
from qubo_project.feature_selection import select_features


@pytest.fixture(scope="module")
def sample_dataset_path():
    base_dir = Path(__file__).parent.parent
    csv_path = base_dir / "data" / "sample_test_dataset.csv"
    if not csv_path.exists():
        pytest.fail(f"Required test dataset not found: {csv_path}")
    return csv_path


@pytest.fixture(scope="module")
def reduced_dataset(sample_dataset_path, tmp_path_factory):
    """Preprocess and run feature selection once for all model tests."""
    tmp_path = tmp_path_factory.mktemp("model_fixture")

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

    # Feature selection
    train_csv = tmp_path / "train_reduced.csv"
    test_csv = tmp_path / "test_reduced.csv"
    opt_csv = tmp_path / "optimizations.csv"
    fs_json = tmp_path / "fs_stats.json"
    select_features(
        normalized_csv=str(norm_csv),
        reducedTrain_csv=str(train_csv),
        reducedTest_csv=str(test_csv),
        output_ottim_csv=str(opt_csv),
        output_json=str(fs_json),
        target_column="target",
        percTest=0.30,
        percSelected=0.20,
        allowance=1,
        seed=42,
        alpha_computations=20,
    )

    return train_csv, test_csv, fs_json


# ----------------------------------------------------------------------
# Test 1: Training produces a saved model file
# ----------------------------------------------------------------------
def test_model_training_saves_model(reduced_dataset, tmp_path):
    train_csv, test_csv, fs_json = reduced_dataset
    model_path = tmp_path / "model.joblib"
    metrics_path = tmp_path / "metrics.json"

    train(
        classifier="random_forest",
        reducedTrain_csv=str(train_csv),
        target_column="target",
        model_path=str(model_path),
        metrics_json=str(metrics_path),
        seed=42,
    )

    assert model_path.exists(), "Model file was not created"
    # Verify it can be loaded
    model = joblib.load(model_path)
    assert hasattr(model, "predict"), "Model does not have predict method"


# ----------------------------------------------------------------------
# Test 2: Training produces a JSON with required fields
# ----------------------------------------------------------------------
def test_model_training_metrics_json(reduced_dataset, tmp_path):
    train_csv, test_csv, fs_json = reduced_dataset
    model_path = tmp_path / "model.joblib"
    metrics_path = tmp_path / "metrics.json"

    train(
        classifier="logistic_regression",
        reducedTrain_csv=str(train_csv),
        target_column="target",
        model_path=str(model_path),
        metrics_json=str(metrics_path),
        seed=42,
    )

    with open(metrics_path) as f:
        metrics = json.load(f)

    required_keys = [
        "classifier",
        "seed",
        "training_dataset",
        "target_column",
        "model_path",
        "n_samples",
        "n_features",
        "target_1_percentage",
        "dataset_input_time",
        "training_time",
    ]
    for key in required_keys:
        assert key in metrics, f"Missing key: {key}"


# ----------------------------------------------------------------------
# Test 3: Prediction produces a CSV with required columns
# ----------------------------------------------------------------------
def test_model_prediction_produces_csv(reduced_dataset, tmp_path):
    train_csv, test_csv, fs_json = reduced_dataset

    # First, train a model
    model_path = tmp_path / "model.joblib"
    metrics_path = tmp_path / "metrics.json"
    train(
        classifier="decision_tree",
        reducedTrain_csv=str(train_csv),
        target_column="target",
        model_path=str(model_path),
        metrics_json=str(metrics_path),
        seed=42,
    )

    # Then predict on the test set
    pred_csv = tmp_path / "predictions.csv"
    stats_json = tmp_path / "stats.json"
    predict(
        reduced_Test_csv=str(test_csv),
        target_column="target",
        model_path=str(model_path),
        predictions_csv=str(pred_csv),
        classif_stats_json=str(stats_json),
    )

    # Check CSV exists and has required columns
    assert pred_csv.exists()
    df = pd.read_csv(pred_csv)
    expected_columns = ["row_n", "target", "prediction", "score"]
    for col in expected_columns:
        assert col in df.columns, f"Missing column: {col}"
    assert len(df) > 0, "Predictions CSV is empty"


# ----------------------------------------------------------------------
# Test 4: Prediction stats JSON has required fields
# ----------------------------------------------------------------------
def test_model_prediction_stats_json(reduced_dataset, tmp_path):
    train_csv, test_csv, fs_json = reduced_dataset

    model_path = tmp_path / "model.joblib"
    metrics_path = tmp_path / "metrics.json"
    train(
        classifier="random_forest",
        reducedTrain_csv=str(train_csv),
        target_column="target",
        model_path=str(model_path),
        metrics_json=str(metrics_path),
        seed=42,
    )

    pred_csv = tmp_path / "predictions.csv"
    stats_json = tmp_path / "stats.json"
    predict(
        reduced_Test_csv=str(test_csv),
        target_column="target",
        model_path=str(model_path),
        predictions_csv=str(pred_csv),
        classif_stats_json=str(stats_json),
    )

    with open(stats_json) as f:
        stats = json.load(f)

    required_keys = [
        "classifier",
        "n_samples",
        "target_1_count",
        "target_1_percentage",
        "accuracy",
        "class_0",
        "class_1",
        "confusion_matrix",
    ]
    for key in required_keys:
        assert key in stats, f"Missing key: {key}"
    assert "roc_auc" in stats or stats.get("roc_auc") is None
    assert "precision" in stats["class_0"]
    assert "recall" in stats["class_1"]