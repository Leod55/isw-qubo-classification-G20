# src/qubo_project/model.py
import argparse
import sys
import time
from pathlib import Path
from typing import Tuple

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from qubo_project.utils import setup_logger, save_json, load_json

logger = setup_logger(__name__)


def train(
    classifier: str,
    reducedTrain_csv: str,
    target_column: str,
    model_path: str,
    metrics_json: str,
    seed: int = 42,
) -> None:
    """
    Train the selected classifier on the reduced training set.

    Args:
        classifier: One of ['random_forest', 'logistic_regression', 'decision_tree'].
        reducedTrain_csv: Training dataset with selected features.
        target_column: Target column name.
        model_path: Output path for the trained model (.joblib).
        metrics_json: Output path for training statistics JSON.
        seed: Random seed.
    """
    logger.info(f"train called: classifier={classifier}, seed={seed}")

    # --- STUB: Replace with actual training logic ---
    logger.warning("Using stub implementation – no real training performed.")

    # Simulate reading the dataset
    import pandas as pd
    df = pd.read_csv(reducedTrain_csv)
    n_samples, n_features = df.shape[0], df.shape[1] - 1

    # Simulate training time
    time.sleep(0.5)

    # Write dummy model file (empty file, just to satisfy path)
    Path(model_path).parent.mkdir(parents=True, exist_ok=True)
    with open(model_path, 'w') as f:
        f.write("STUB_MODEL")  # Replace with joblib.dump in real implementation

    # Dummy metrics
    metrics = {
        "classifier": classifier,
        "seed": seed,
        "training_dataset": reducedTrain_csv,
        "target_column": target_column,
        "model_path": model_path,
        "n_samples": n_samples,
        "n_features": n_features,
        "target_1_percentage": 1.5,
        "dataset_input_time": 0.1,
        "training_time": 0.5,
    }
    save_json(metrics, metrics_json)
    # --- END STUB ---

    logger.info("train stub completed.")


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
        reduced_Test_csv: Test dataset with selected features.
        target_column: Target column name.
        model_path: Path to the saved model (.joblib).
        predictions_csv: Output CSV with row_n, target, prediction, score.
        classif_stats_json: Output JSON with classification metrics.
    """
    logger.info(f"predict called: model_path={model_path}")

    # --- STUB: Replace with actual prediction logic ---
    logger.warning("Using stub implementation – dummy predictions.")

    import pandas as pd
    df = pd.read_csv(reduced_Test_csv)
    n_samples = len(df)

    # Dummy predictions
    pred_df = pd.DataFrame({
        "row_n": range(n_samples),
        "target": df[target_column],
        "prediction": [0] * n_samples,   # all zeros for stub
        "score": [0.5] * n_samples,
    })
    pred_df.to_csv(predictions_csv, index=False)

    # Dummy classification stats
    stats = {
        "classifier": "random_forest_stub",
        "n_samples": n_samples,
        "target_1_count": int(n_samples * 0.1),
        "target_1_percentage": 10.0,
        "accuracy": 0.90,
        "class_0": {"precision": 0.91, "recall": 0.92, "f1": 0.915, "support": int(n_samples * 0.9)},
        "class_1": {"precision": 0.80, "recall": 0.75, "f1": 0.774, "support": int(n_samples * 0.1)},
        "roc_auc": 0.88,
        "confusion_matrix": {"labels": [0, 1], "matrix": [[900, 80], [20, 80]]},
    }
    save_json(stats, classif_stats_json)
    # --- END STUB ---

    logger.info("predict stub completed.")


# ========================
# CLI Entry Point (subcommands)
# ========================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Model training and prediction.")
    subparsers = parser.add_subparsers(dest="command", required=True, help="Subcommand")

    # --- train subcommand ---
    parser_train = subparsers.add_parser("train")
    parser_train.add_argument("--classifier", required=True,
                              choices=["random_forest", "logistic_regression", "decision_tree"])
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