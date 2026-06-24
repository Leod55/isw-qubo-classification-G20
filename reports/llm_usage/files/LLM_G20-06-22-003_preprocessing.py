# src/qubo_project/preprocessing.py
import argparse
import sys
import time
from pathlib import Path
from typing import Optional

# Add parent to path for local imports (when run as script)
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from qubo_project.utils import setup_logger, save_json  # type: ignore

logger = setup_logger(__name__)


def fit_normalize(
    input_csv: str,
    target_column: str,
    normalized_csv: str,
    outInitialRes_json: str,
    minPercValid: float = 0.05,
) -> None:
    """
    Preprocess the dataset: drop near-zero columns, z-score normalise features.

    Args:
        input_csv: Path to input dataset CSV.
        target_column: Name of the binary target column.
        normalized_csv: Output path for the normalised dataset.
        outInitialRes_json: Output path for preprocessing statistics JSON.
        minPercValid: Minimum fraction of non-zero/non-null values per column
                      to keep that column. Default 0.05 (5%).
    """
    logger.info(
        f"fit_normalize called with input_csv={input_csv}, "
        f"target={target_column}, minPercValid={minPercValid}"
    )

    # --- STUB: Replace with real implementation ---
    # For now, we just log and write placeholder outputs.

    # 1. Simulate reading the dataset (you will replace this with pandas)
    logger.warning("Using stub implementation – no real preprocessing performed.")

    # 2. Simulate dropping columns and normalising (placeholder data)
    dummy_stats = {
        "n_input_features": 145,
        "n_kept_features": 120,
        "dataset_size": 20000,
        "dataset_input_time": 0.1,
        "dataset_processing_time": 0.2,
        "dropped_feature_names": ["feature_dummy_1", "feature_dummy_2"],
    }

    # 3. Write dummy JSON
    save_json(dummy_stats, outInitialRes_json)
    logger.info(f"Saved dummy JSON to {outInitialRes_json}")

    # 4. Write a dummy normalised CSV (just a copy of the input for now)
    #    In the real implementation, you will create a new DataFrame.
    import pandas as pd
    df = pd.read_csv(input_csv)
    df.to_csv(normalized_csv, index=False)
    logger.info(f"Copied input to {normalized_csv} as placeholder.")
    # --- END STUB ---

    logger.info("fit_normalize stub completed.")


# ========================
# CLI Entry Point
# ========================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Preprocess dataset for QUBO classification."
    )
    parser.add_argument("--input", required=True, help="Input CSV file")
    parser.add_argument("--target", required=True, help="Target column name")
    parser.add_argument("--out-data", required=True, help="Output normalised CSV")
    parser.add_argument("--out-json", required=True, help="Output preprocessing JSON")
    parser.add_argument(
        "--min-perc-valid",
        type=float,
        default=0.05,
        help="Min fraction of valid non-zero data per column (default 0.05)",
    )

    args = parser.parse_args()

    try:
        fit_normalize(
            input_csv=args.input,
            target_column=args.target,
            normalized_csv=args.out_data,
            outInitialRes_json=args.out_json,
            minPercValid=args.min_perc_valid,
        )
        sys.exit(0)
    except Exception as e:
        logger.error(f"Preprocessing failed: {e}", exc_info=True)
        sys.exit(1)