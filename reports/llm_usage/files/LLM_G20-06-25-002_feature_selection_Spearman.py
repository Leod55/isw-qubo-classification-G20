# ------------------------------------------------------------------------
# 2. Compute Spearman correlations on the TRAINING set only
# ------------------------------------------------------------------------
logger.info("Computing Spearman correlations on training set...")
corr_start = time.perf_counter()

# Feature-target correlations (absolute)
target_vec = df_train[target_column].values
feature_target_corr = np.zeros(n_features)
for i, col in enumerate(feature_cols):
    rho, _ = spearmanr(df_train[col].values, target_vec, nan_policy="omit")
    feature_target_corr[i] = abs(rho) if not np.isnan(rho) else 0.0

# Feature-feature correlation matrix (absolute, dense)
feature_data = df_train[feature_cols].values.T  # shape (n_features, train_size)
feature_feature_corr = np.zeros((n_features, n_features))
for i in range(n_features):
    for j in range(i + 1, n_features):
        rho, _ = spearmanr(feature_data[i], feature_data[j], nan_policy="omit")
        val = abs(rho) if not np.isnan(rho) else 0.0
        feature_feature_corr[i, j] = val
        feature_feature_corr[j, i] = val  # symmetric

# SAFETY: neutralize any remaining NaNs (e.g., due to all-zero columns)
feature_target_corr = np.nan_to_num(feature_target_corr, nan=0.0)
feature_feature_corr = np.nan_to_num(feature_feature_corr, nan=0.0)

corr_time = time.perf_counter() - corr_start
logger.info(f"Correlation matrix computed in {corr_time:.2f}s")