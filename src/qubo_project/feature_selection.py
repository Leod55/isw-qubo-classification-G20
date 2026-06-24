# src/qubo_project/feature_selection.py
import argparse
import sys
import time
from pathlib import Path
from typing import List, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))

from qubo_project.utils import setup_logger, save_json, load_json

logger = setup_logger(__name__)


def select_features(
    normalized_csv: str,
    reducedTrain_csv: str,
    reducedTest_csv: str,
    output_ottim_csv: str,
    output_json: str,
    target_column: str,
    percTest: float = 0.30,
    percSelected: float = 0.20,
    allowance: int = 1,
    seed: int = 42,
    alpha_computations: int = 100,
) -> None:
    """
    Run QUBO feature selection, varying alpha to reach desired feature count.

    Args:
        normalized_csv: Input normalised dataset.
        reducedTrain_csv: Output training set with only selected features.
        reducedTest_csv: Output test set with only selected features.
        output_ottim_csv: CSV of alpha vs. optimisations (alpha, time, n_feat, cost).
        output_json: Statistics JSON (selected vector, alpha, times, etc.).
        target_column: Target column name.
        percTest: Fraction of rows to use as test set (cut-off).
        percSelected: Desired percentage of features to keep.
        allowance: Absolute tolerance (±) around the target number of features.
        seed: Random seed for reproducibility.
        alpha_computations: Max number of alpha attempts.
    """
    logger.info(
        f"select_features called: percSelected={percSelected}, allowance={allowance}, "
        f"alpha_computations={alpha_computations}"
    )

    # --- STUB: Replace with real QUBO solver ---
    logger.warning("Using stub implementation – dummy QUBO results.")

    # Simulate reading the normalised data to know number of features.
    import pandas as pd
    df = pd.read_csv(normalized_csv)
    feature_cols = [c for c in df.columns if c != target_column]
    n_features = len(feature_cols)

    # Determine target K
    target_k = int(round(percSelected * n_features))
    logger.info(f"Target K = {target_k} ± {allowance}")

    # Dummy selected vector (first 'target_k' features selected)
    selected_vector = [1] * target_k + [0] * (n_features - target_k)

    # Dummy optimisation trace (for output_ottim_csv)
    trace_df = pd.DataFrame({
        "alpha": [0.1, 0.2, 0.3, 0.4],
        "time": [0.5, 0.6, 0.7, 0.8],
        "n_features": [10, 15, target_k, 50],
        "cost": [-0.5, -0.6, -0.8, -0.9],
    })
    trace_df.to_csv(output_ottim_csv, index=False)

    # Dummy JSON
    stats = {
        "n_features": n_features,
        "target_ratio": percSelected,
        "target_k": target_k,
        "allowance": allowance,
        "n_selected": target_k,
        "alpha": 0.3,
        "selected_vector": selected_vector,
        "selected_feature_names": feature_cols[:target_k],
        "algorithm": "simulated_annealing_stub",
        "seed": seed,
        "alpha_computations": alpha_computations,
        "percTest": percTest,
        "training_dataset_size": int(len(df) * (1 - percTest)),
        "test_dataset_size": int(len(df) * percTest),
        "q_matrix_creation_time": 1.0,
        "mean_optimization_time": 0.2,
        "std_dev_optimization_time": 0.05,
    }
    save_json(stats, output_json)

    # Dummy reduced datasets (copy of original for now – real impl will filter columns)
    df.to_csv(reducedTrain_csv, index=False)
    df.to_csv(reducedTest_csv, index=False)
    logger.info("Stub: saved placeholder reduced CSVs.")
    # --- END STUB ---

    logger.info("select_features stub completed.")


# ========================
# CLI Entry Point
# ========================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="QUBO feature selection.")
    parser.add_argument("--in-normalized", required=True)
    parser.add_argument("--out-train", required=True)
    parser.add_argument("--out-test", required=True)
    parser.add_argument("--out-optimizations", required=True)
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--perc-selected", type=float, default=0.20)
    parser.add_argument("--allowance", type=int, default=1)
    parser.add_argument("--perc-test", type=float, default=0.30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--alpha-computations", type=int, default=100)

    args = parser.parse_args()

    try:
        select_features(
            normalized_csv=args.in_normalized,
            reducedTrain_csv=args.out_train,
            reducedTest_csv=args.out_test,
            output_ottim_csv=args.out_optimizations,
            output_json=args.out_json,
            target_column=args.target,
            percTest=args.perc_test,
            percSelected=args.perc_selected,
            allowance=args.allowance,
            seed=args.seed,
            alpha_computations=args.alpha_computations,
        )
        sys.exit(0)
    except Exception as e:
        logger.error(f"Feature selection failed: {e}", exc_info=True)
        sys.exit(1)