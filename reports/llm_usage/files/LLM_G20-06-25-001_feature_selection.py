# src/qubo_project/feature_selection.py
import argparse
import sys
import time
import math
import random
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

# Add parent to path for local imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from qubo_project.utils import setup_logger, save_json

logger = setup_logger(__name__)


# ============================================================================
# Simulated Annealing Solver for Dense QUBO
# ============================================================================
class SimulatedAnnealingQUBO:
    """
    Simulated Annealing solver for dense QUBO problems.
    Uses a bit-flip neighbour and geometric cooling schedule.
    """

    def __init__(
        self,
        Q: np.ndarray,
        seed: int = 42,
        max_steps: int = 2000,
        temp_start: float = 1.0,
        temp_end: float = 0.01,
    ):
        """
        Args:
            Q: Dense n×n QUBO matrix (numpy array).
            seed: Random seed for reproducibility.
            max_steps: Number of iterations.
            temp_start: Initial temperature.
            temp_end: Final temperature.
        """
        self.Q = Q
        self.n = Q.shape[0]
        self.seed = seed
        self.max_steps = max_steps
        self.temp_start = temp_start
        self.temp_end = temp_end

        random.seed(seed)
        np.random.seed(seed)

    def _compute_energy(self, x: np.ndarray) -> float:
        """Compute x^T Q x efficiently using vectorised operations."""
        return x @ self.Q @ x

    def _delta_energy(self, x: np.ndarray, idx: int) -> float:
        """
        Compute ΔE if we flip bit at index `idx`.
        ΔE = (1 - 2*x_i) * ( Q[i,i] + 2 * Σ_{j≠i} Q[i,j] * x_j )
        """
        xi = x[idx]
        # linear term: sum of Q[idx][j] * x_j for j != idx
        lin = self.Q[idx, :] @ x
        # remove self contribution (since Q[idx][idx]*x_i is included, but we need it separately)
        # Actually, the standard formula for flipping bit i:
        # delta = (1 - 2*x_i) * ( Q[i,i] + 2 * Σ_{j≠i} Q[i,j] * x_j )
        # Let's compute sum_{j≠i} Q[i,j] * x_j
        sum_offdiag = (self.Q[idx, :] @ x) - self.Q[idx, idx] * x[idx]
        delta = (1 - 2 * xi) * (self.Q[idx, idx] + 2 * sum_offdiag)
        return delta

    def solve(self) -> Tuple[np.ndarray, float, List[float]]:
        """
        Run simulated annealing and return the best solution found.

        Returns:
            best_x: Binary vector (numpy array).
            best_energy: Energy of the best solution.
            energy_history: List of energies for each step (for trace).
        """
        # Initialise randomly
        x = np.random.randint(0, 2, size=self.n)
        current_energy = self._compute_energy(x)
        best_x = x.copy()
        best_energy = current_energy

        energy_history = [current_energy]

        # Temperature schedule (geometric)
        temp = self.temp_start
        cooling_rate = (self.temp_end / self.temp_start) ** (1.0 / self.max_steps)

        for step in range(self.max_steps):
            # Pick a random bit to flip
            idx = random.randint(0, self.n - 1)
            delta = self._delta_energy(x, idx)

            # Accept or reject
            if delta < 0 or random.random() < math.exp(-delta / temp):
                # Flip the bit
                x[idx] = 1 - x[idx]
                current_energy += delta  # update energy efficiently

                # Update best
                if current_energy < best_energy:
                    best_x = x.copy()
                    best_energy = current_energy

            # Cool down
            temp *= cooling_rate
            energy_history.append(current_energy)

        return best_x, best_energy, energy_history


# ============================================================================
# Main Feature Selection Function
# ============================================================================
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
    Run QUBO feature selection with Simulated Annealing, varying α to hit the target count.
    """
    start_total = time.perf_counter()
    logger.info(
        f"select_features called: percSelected={percSelected}, allowance={allowance}, "
        f"alpha_computations={alpha_computations}"
    )

    # ------------------------------------------------------------------------
    # 1. Load the normalised dataset
    # ------------------------------------------------------------------------
    df_full = pd.read_csv(normalized_csv)
    feature_cols = [c for c in df_full.columns if c != target_column]
    n_features = len(feature_cols)
    n_total_rows = len(df_full)

    # Determine the split point (cut-off)
    train_size = int(n_total_rows * (1 - percTest))
    test_size = n_total_rows - train_size
    logger.info(f"Dataset: {n_total_rows} rows, {n_features} features. "
                f"Train: {train_size}, Test: {test_size}")

    df_train = df_full.iloc[:train_size].copy()
    df_test = df_full.iloc[train_size:].copy()

    # ------------------------------------------------------------------------
    # 2. Compute Spearman correlations on the TRAINING set only
    # ------------------------------------------------------------------------
    logger.info("Computing Spearman correlations on training set...")
    corr_start = time.perf_counter()

    # Feature-target correlations (absolute)
    target_vec = df_train[target_column].values
    feature_target_corr = np.zeros(n_features)
    for i, col in enumerate(feature_cols):
        # Spearmanr returns (correlation, p-value)
        rho, _ = spearmanr(df_train[col].values, target_vec, nan_policy="omit")
        feature_target_corr[i] = abs(rho)

    # Feature-feature correlation matrix (absolute, dense)
    # We compute only the upper triangle for efficiency.
    feature_data = df_train[feature_cols].values.T  # shape (n_features, train_size)
    feature_feature_corr = np.zeros((n_features, n_features))
    for i in range(n_features):
        for j in range(i + 1, n_features):
            rho, _ = spearmanr(feature_data[i], feature_data[j], nan_policy="omit")
            val = abs(rho)
            feature_feature_corr[i, j] = val
            feature_feature_corr[j, i] = val  # symmetric

    corr_time = time.perf_counter() - corr_start
    logger.info(f"Correlation matrix computed in {corr_time:.2f}s")

    # ------------------------------------------------------------------------
    # 3. Determine target number of features K
    # ------------------------------------------------------------------------
    K_target = int(round(percSelected * n_features))
    K_min = max(0, K_target - allowance)
    K_max = min(n_features, K_target + allowance)
    logger.info(f"Target K = {K_target} ± {allowance} → range [{K_min}, {K_max}]")

    # ------------------------------------------------------------------------
    # 4. Define QUBO builder and solver wrapper
    # ------------------------------------------------------------------------
    def build_Q(alpha: float) -> np.ndarray:
        """Construct the dense QUBO matrix Q for a given alpha."""
        Q = np.zeros((n_features, n_features))
        # Diagonal: -alpha * |corr(feature, target)|
        np.fill_diagonal(Q, -alpha * feature_target_corr)
        # Off-diagonal: (1-alpha) * |corr(feature_i, feature_j)|
        # We use the full symmetric matrix; the double sum in the objective
        # naturally accounts for both directions.
        Q += (1 - alpha) * feature_feature_corr
        return Q

    def solve_alpha(alpha: float) -> Tuple[np.ndarray, float, int, float]:
        """
        Run SA for a given alpha and return:
            vector, energy, number_of_ones, runtime_seconds
        """
        Q = build_Q(alpha)
        solver = SimulatedAnnealingQUBO(Q, seed=seed)
        start_solve = time.perf_counter()
        vec, energy, _ = solver.solve()
        solve_time = time.perf_counter() - start_solve
        n_ones = int(vec.sum())
        return vec, energy, n_ones, solve_time

    # ------------------------------------------------------------------------
    # 5. Binary search over α to find a solution in [K_min, K_max]
    # ------------------------------------------------------------------------
    trace_records = []
    best_solution = None
    best_alpha = None
    best_error = float("inf")

    # We'll perform a binary search, but we also need to handle the case where
    # no α yields exactly the range. We'll keep the closest.
    alpha_low = 0.0
    alpha_high = 1.0
    attempts = 0

    # We also sample a few points to ensure we have a monotonic trend.
    # Since n_selected is monotonic w.r.t. α (should be), binary search works.
    while attempts < alpha_computations:
        alpha = (alpha_low + alpha_high) / 2.0
        attempts += 1

        logger.debug(f"Attempt {attempts}: α = {alpha:.4f}")
        vec, energy, n_ones, solve_time = solve_alpha(alpha)

        trace_records.append({
            "alpha": alpha,
            "time": solve_time,
            "n_features": n_ones,
            "cost": energy,
        })

        error = abs(n_ones - K_target)
        if error < best_error:
            best_error = error
            best_solution = vec
            best_alpha = alpha

        if K_min <= n_ones <= K_max:
            logger.info(f"Found solution: α={alpha:.4f}, n_ones={n_ones}, energy={energy:.4f}")
            best_solution = vec
            best_alpha = alpha
            break

        # Update bounds: if n_ones < K_min, we need more features → increase α
        if n_ones < K_min:
            alpha_low = alpha
        else:  # n_ones > K_max
            alpha_high = alpha

    # If we didn't find a solution within tolerance, use the best we found.
    if not (K_min <= best_solution.sum() <= K_max):
        logger.warning(
            f"No α found within tolerance. Best: n_ones={best_solution.sum()}, "
            f"α={best_alpha:.4f}, error={best_error}"
        )

    # ------------------------------------------------------------------------
    # 6. Prepare outputs
    # ------------------------------------------------------------------------
    # Extract selected feature names
    selected_indices = np.where(best_solution == 1)[0].tolist()
    selected_features = [feature_cols[i] for i in selected_indices]
    selected_count = len(selected_features)

    logger.info(f"Selected {selected_count} features out of {n_features}")

    # Build reduced datasets
    keep_cols = selected_features + [target_column]
    df_train_reduced = df_train[keep_cols]
    df_test_reduced = df_test[keep_cols]

    # Save reduced CSVs
    df_train_reduced.to_csv(reducedTrain_csv, index=False)
    df_test_reduced.to_csv(reducedTest_csv, index=False)
    logger.info(f"Saved reduced training set to {reducedTrain_csv}")
    logger.info(f"Saved reduced test set to {reducedTest_csv}")

    # Save optimisation trace CSV
    trace_df = pd.DataFrame(trace_records)
    trace_df.to_csv(output_ottim_csv, index=False)
    logger.info(f"Saved optimisation trace to {output_ottim_csv}")

    # ------------------------------------------------------------------------
    # 7. Build and save JSON statistics
    # ------------------------------------------------------------------------
    q_matrix_creation_time = corr_time  # approximation (correlation matrix is the bulk)
    opt_times = [r["time"] for r in trace_records]
    mean_opt_time = np.mean(opt_times) if opt_times else 0.0
    std_opt_time = np.std(opt_times) if opt_times else 0.0

    stats = {
        "n_features": n_features,
        "target_ratio": percSelected,
        "target_k": K_target,
        "allowance": allowance,
        "n_selected": int(selected_count),
        "alpha": best_alpha,
        "selected_vector": [int(v) for v in best_solution.tolist()],
        "selected_feature_names": selected_features,
        "algorithm": "simulated_annealing",
        "seed": seed,
        "alpha_computations": attempts,
        "percTest": percTest,
        "training_dataset_size": train_size,
        "test_dataset_size": test_size,
        "q_matrix_creation_time": round(q_matrix_creation_time, 4),
        "mean_optimization_time": round(mean_opt_time, 4),
        "std_dev_optimization_time": round(std_opt_time, 4),
    }
    save_json(stats, output_json)
    logger.info(f"Saved JSON statistics to {output_json}")

    total_elapsed = time.perf_counter() - start_total
    logger.info(f"select_features completed in {total_elapsed:.2f}s")


# ============================================================================
# CLI Entry Point
# ============================================================================
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