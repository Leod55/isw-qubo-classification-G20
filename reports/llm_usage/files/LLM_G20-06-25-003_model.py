# src/qubo_project/model.py
import argparse
import sys
import time
import json
from pathlib import Path
from typing import Dict, Any, Tuple

import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
    confusion_matrix,
)

# Add parent to path for local imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from qubo_project.utils import setup_logger, save_json, load_json

logger = setup_logger(__name__)

# Mapping from classifier name to sklearn class and default parameters
CLASSIFIERS = {
    "random_forest": {
        "class": RandomForestClassifier,
        "params": {"n_estimators": 100, "max_depth": 10, "random_state": 42},
    },
    "logistic_regression": {
        "class": LogisticRegression,
        "params": {"C": 1.0, "max_iter": 1000, "random_state": 42},
    },
    "decision_tree": {
        "class": DecisionTreeClassifier,
        "params": {"max_depth": 10, "random_state": 42},
    },
}


def train(
    classifier: str,
    reducedTrain_csv: str,
    target_column: str,
    model_path: str,
    metrics_json: str,
    seed: int = 42,
) -> None:
    """
    Train the selected classifier on the reduced training set and save the model.

    Args:
        classifier: One of ['random_forest', 'logistic_regression', 'decision_tree'].
        reducedTrain_csv: Training dataset with selected features and target.
        target_column: Name of the target column.
        model_path: Output path for the trained model (.joblib).
        metrics_json: Output path for training statistics JSON.
        seed: Random seed for reproducibility.
    """
    start_total = time.perf_counter()
    logger.info(f"train called: classifier={classifier}, seed={seed}, data={reducedTrain_csv}")

    # --------------------------------------------------------------------
    # 1. Load the reduced training dataset
    # --------------------------------------------------------------------
    read_start = time.perf_counter()
    df = pd.read_csv(reducedTrain_csv)
    n_samples = len(df)
    feature_cols = [c for c in df.columns if c != target_column]
    n_features = len(feature_cols)
    read_time = time.perf_counter() - read_start

    # Separate features and target
    X = df[feature_cols].values
    y = df[target_column].values

    # Check target is binary and compute class distribution
    unique_targets = np.unique(y)
    if not np.array_equal(unique_targets, [0, 1]):
        logger.warning(f"Target contains values {unique_targets}; ensure it's binary 0/1.")
    target_1_percentage = 100.0 * np.mean(y)

    # --------------------------------------------------------------------
    # 2. Instantiate and train the classifier
    # --------------------------------------------------------------------
    if classifier not in CLASSIFIERS:
        raise ValueError(f"Unknown classifier: {classifier}. Choose from {list(CLASSIFIERS.keys())}")

    clf_info = CLASSIFIERS[classifier]
    # Override random_state with the provided seed
    clf_params = clf_info["params"].copy()
    if "random_state" in clf_params:
        clf_params["random_state"] = seed

    clf = clf_info["class"](**clf_params)

    training_start = time.perf_counter()
    clf.fit(X, y)
    training_time = time.perf_counter() - training_start

    # --------------------------------------------------------------------
    # 3. Save the model
    # --------------------------------------------------------------------
    Path(model_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, model_path)
    logger.info(f"Model saved to {model_path}")

    # --------------------------------------------------------------------
    # 4. Compute training (in‑sample) metrics (optional but useful)
    # --------------------------------------------------------------------
    y_pred = clf.predict(X)
    y_score = clf.predict_proba(X)[:, 1] if hasattr(clf, "predict_proba") else y_pred

    # Basic metrics
    accuracy = accuracy_score(y, y_pred)
    precision, recall, f1, support = precision_recall_fscore_support(
        y, y_pred, labels=[0, 1], zero_division=0
    )

    metrics = {
        "classifier": classifier,
        "seed": seed,
        "training_dataset": reducedTrain_csv,
        "target_column": target_column,
        "model_path": model_path,
        "n_samples": n_samples,
        "n_features": n_features,
        "target_1_percentage": round(float(target_1_percentage), 4),
        "dataset_input_time": round(read_time, 4),
        "training_time": round(training_time, 4),
        "in_sample_accuracy": round(float(accuracy), 6),
        "in_sample_precision_class0": round(float(precision[0]), 6),
        "in_sample_recall_class0": round(float(recall[0]), 6),
        "in_sample_f1_class0": round(float(f1[0]), 6),
        "in_sample_precision_class1": round(float(precision[1]), 6),
        "in_sample_recall_class1": round(float(recall[1]), 6),
        "in_sample_f1_class1": round(float(f1[1]), 6),
    }

    save_json(metrics, metrics_json)
    logger.info(f"Training metrics saved to {metrics_json}")

    total_elapsed = time.perf_counter() - start_total
    logger.info(f"train completed in {total_elapsed:.2f}s")


def predict(
    reduced_Test_csv: str,
    target_column: str,
    model_path: str,
    predictions_csv: str,
    classif_stats_json: str,
) -> None:
    """
    Load a trained model and generate predictions on the reduced test set.

    Args:
        reduced_Test_csv: Test dataset with selected features and target.
        target_column: Name of the target column.
        model_path: Path to the saved model (.joblib).
        predictions_csv: Output CSV with row_n, target, prediction, score.
        classif_stats_json: Output JSON with classification metrics.
    """
    start_total = time.perf_counter()
    logger.info(f"predict called: model={model_path}, test_data={reduced_Test_csv}")

    # --------------------------------------------------------------------
    # 1. Load the model
    # --------------------------------------------------------------------
    clf = joblib.load(model_path)
    logger.info("Model loaded successfully.")

    # --------------------------------------------------------------------
    # 2. Read the reduced test dataset
    # --------------------------------------------------------------------
    read_start = time.perf_counter()
    df = pd.read_csv(reduced_Test_csv)
    n_samples = len(df)
    feature_cols = [c for c in df.columns if c != target_column]
    read_time = time.perf_counter() - read_start

    X = df[feature_cols].values
    y_true = df[target_column].values

    # --------------------------------------------------------------------
    # 3. Generate predictions and scores
    # --------------------------------------------------------------------
    pred_start = time.perf_counter()
    y_pred = clf.predict(X)
    if hasattr(clf, "predict_proba"):
        y_score = clf.predict_proba(X)[:, 1]
    else:
        # Fallback: for classifiers without probability, use decision function or dummy
        y_score = y_pred.astype(float)
    pred_time = time.perf_counter() - pred_start

    # --------------------------------------------------------------------
    # 4. Build output predictions CSV
    # --------------------------------------------------------------------
    pred_df = pd.DataFrame({
        "row_n": range(n_samples),
        "target": y_true,
        "prediction": y_pred,
        "score": y_score,
    })
    pred_df.to_csv(predictions_csv, index=False)
    logger.info(f"Predictions saved to {predictions_csv}")

    # --------------------------------------------------------------------
    # 5. Compute classification statistics
    # --------------------------------------------------------------------
    # Accuracy
    accuracy = accuracy_score(y_true, y_pred)

    # Per‑class metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1], zero_division=0
    )

    # ROC‑AUC (only if we have probabilities and both classes)
    roc_auc = None
    if hasattr(clf, "predict_proba") and len(np.unique(y_true)) == 2:
        try:
            roc_auc = roc_auc_score(y_true, y_score)
        except ValueError:
            roc_auc = None

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    # Build the stats dict
    stats: Dict[str, Any] = {
        "classifier": str(clf.__class__.__name__),
        "n_samples": n_samples,
        "target_1_count": int(np.sum(y_true == 1)),
        "target_1_percentage": round(100.0 * np.mean(y_true), 4),
        "accuracy": round(float(accuracy), 6),
        "class_0": {
            "precision": round(float(precision[0]), 6),
            "recall": round(float(recall[0]), 6),
            "f1": round(float(f1[0]), 6),
            "support": int(support[0]),
        },
        "class_1": {
            "precision": round(float(precision[1]), 6),
            "recall": round(float(recall[1]), 6),
            "f1": round(float(f1[1]), 6),
            "support": int(support[1]),
        },
        "confusion_matrix": {
            "labels": [0, 1],
            "matrix": cm.tolist(),
        },
    }
    if roc_auc is not None:
        stats["roc_auc"] = round(float(roc_auc), 6)
    else:
        stats["roc_auc"] = None

    # Add timing info (optional but helpful)
    stats["dataset_input_time"] = round(read_time, 4)
    stats["prediction_time"] = round(pred_time, 4)

    save_json(stats, classif_stats_json)
    logger.info(f"Classification statistics saved to {classif_stats_json}")

    total_elapsed = time.perf_counter() - start_total
    logger.info(f"predict completed in {total_elapsed:.2f}s")


# ============================================================================
# CLI Entry Point (subcommands)
# ============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Model training and prediction.")
    subparsers = parser.add_subparsers(dest="command", required=True, help="Subcommand")

    # --- train subcommand ---
    parser_train = subparsers.add_parser("train")
    parser_train.add_argument(
        "--classifier",
        required=True,
        choices=list(CLASSIFIERS.keys()),
        help="Classifier to train",
    )
    parser_train.add_argument("-in-reduced", required=True, dest="in_reduced")
    parser_train.add_argument("-target", required=True)
    parser_train.add_argument("-out-model", required=True, dest="out_model")
    parser_train.add_argument("-out-metrics", required=True, dest="out_metrics")
    parser_train.add_argument("-seed", type=int, default=42)

    # --- predict subcommand ---
    parser_predict = subparsers.add_parser("predict")
    parser_predict.add_argument("-input-testset", required=True, dest="input_testset")
    parser_predict.add_argument("-target", required=True)
    parser_predict.add_argument("-model", required=True)
    parser_predict.add_argument("-out-predictions", required=True, dest="out_predictions")
    parser_predict.add_argument("-out-stats", required=True, dest="out_stats")

    args = parser.parse_args()

    try:
        if args.command == "train":
            train(
                classifier=args.classifier,
                reducedTrain_csv=args.in_reduced,
                target_column=args.target,
                model_path=args.out_model,
                metrics_json=args.out_metrics,
                seed=args.seed,
            )
        else:  # predict
            predict(
                reduced_Test_csv=args.input_testset,
                target_column=args.target,
                model_path=args.model,
                predictions_csv=args.out_predictions,
                classif_stats_json=args.out_stats,
            )
        sys.exit(0)
    except Exception as e:
        logger.error(f"Model command failed: {e}", exc_info=True)
        sys.exit(1)