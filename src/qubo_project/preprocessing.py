# src/qubo_project/preprocessing.py
import argparse
import sys
import time
from pathlib import Path
from typing import List, Dict

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# Add parent to path for local imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from qubo_project.utils import setup_logger, save_json

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

    Uses a two-pass chunked approach to handle datasets with millions of rows
    without exceeding memory limits.
    """
    start_total = time.perf_counter()
    logger.info(
        f"fit_normalize called: input={input_csv}, target={target_column}, "
        f"minPercValid={minPercValid}"
    )

    # =========================================================================
    # PASS 1: Chunked read to count valid (non-zero, non-null) values per column
    # =========================================================================
    read_start = time.perf_counter()
    chunk_size = 100000  # tune this based on your RAM (50k-200k is safe)
    total_rows = 0
    valid_counts: Dict[str, int] = {}
    all_columns = set()

    logger.info("Starting first pass (chunked) to count valid values...")

    # Use low_memory=False to avoid mixed-type warnings; we know data is numeric.
    for chunk in pd.read_csv(input_csv, chunksize=chunk_size, low_memory=False):
        total_rows += len(chunk)
        # Update valid counts for each feature column (exclude target)
        for col in chunk.columns:
            if col == target_column:
                continue
            all_columns.add(col)
            # Count where value is NOT null AND NOT zero
            valid_mask = chunk[col].notna() & (chunk[col] != 0)
            valid_counts[col] = valid_counts.get(col, 0) + valid_mask.sum()

    read_end = time.perf_counter()
    read_time = read_end - read_start
    logger.info(f"First pass complete. Total rows: {total_rows}, columns: {len(all_columns)}")

    # =========================================================================
    # DROP columns that fall below the threshold
    # =========================================================================
    dropped_features = []
    kept_features = []
    for col in all_columns:
        ratio = valid_counts.get(col, 0) / total_rows
        if ratio < minPercValid:
            dropped_features.append(col)
            logger.debug(f"Dropping {col}: valid ratio = {ratio:.4f}")
        else:
            kept_features.append(col)
            logger.debug(f"Keeping {col}: valid ratio = {ratio:.4f}")

    logger.info(
        f"Kept {len(kept_features)} features, dropped {len(dropped_features)} features."
    )

    # =========================================================================
    # PASS 2: Read only the kept features + target into memory
    # =========================================================================
    usecols = kept_features + [target_column]
    logger.info("Starting second pass: reading selected columns...")
    df = pd.read_csv(input_csv, usecols=usecols, low_memory=False)

    # Safety: fill any remaining NaNs with 0 (should be rare after filtering)
    # StandardScaler requires finite values.
    feature_cols = kept_features
    if df[feature_cols].isna().any().any():
        logger.warning("Found NaN values in kept features. Filling with 0.")
        df[feature_cols] = df[feature_cols].fillna(0)

    # =========================================================================
    # Z-SCORE NORMALISATION (excluding target)
    # =========================================================================
    process_start = time.perf_counter()
    scaler = StandardScaler()
    normalized_features = scaler.fit_transform(df[feature_cols].values)

    # Reconstruct DataFrame with normalised features and target
    normalised_df = pd.DataFrame(
        normalized_features,
        columns=feature_cols,
        index=df.index,
    )
    normalised_df[target_column] = df[target_column].values

    # =========================================================================
    # SAVE OUTPUTS
    # =========================================================================
    normalised_df.to_csv(normalized_csv, index=False)
    logger.info(f"Saved normalised dataset to {normalized_csv}")

    process_end = time.perf_counter()
    processing_time = process_end - process_start

    # =========================================================================
    # BUILD & SAVE JSON STATISTICS
    # =========================================================================
    stats = {
        "n_input_features": len(all_columns),
        "n_kept_features": len(kept_features),
        "dataset_size": total_rows,
        "dataset_input_time": round(read_time, 4),
        "dataset_processing_time": round(read_time + processing_time, 4),
        "dropped_feature_names": dropped_features,
    }
    save_json(stats, outInitialRes_json)
    logger.info(f"Saved preprocessing stats to {outInitialRes_json}")

    total_elapsed = time.perf_counter() - start_total
    logger.info(f"fit_normalize completed in {total_elapsed:.2f} seconds.")


# ========================
# CLI Entry Point (unchanged from stub)
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