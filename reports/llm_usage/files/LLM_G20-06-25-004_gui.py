# src/qubo_project/gui.py
import streamlit as st
import subprocess
import os
import sys
import json
import pandas as pd
from pathlib import Path
import tempfile
import time

# ============================================================================
# Page configuration
# ============================================================================
st.set_page_config(page_title="QUBO Classification Pipeline", layout="wide")
st.title("🧠 QUBO Feature Selection & Binary Classification")

# ============================================================================
# Session state initialization
# ============================================================================
if "logs" not in st.session_state:
    st.session_state.logs = []
if "current_step" not in st.session_state:
    st.session_state.current_step = 0  # 0=idle, 1=preprocess, 2=fs, 3=train, 4=predict
if "output_dir" not in st.session_state:
    st.session_state.output_dir = None

def log_message(msg: str, level: str = "info"):
    """Append a log message to the session state."""
    st.session_state.logs.append(f"[{level.upper()}] {msg}")

def clear_logs():
    st.session_state.logs = []

def run_command(cmd: list, description: str) -> bool:
    """
    Run a subprocess command, capture output, and update logs.
    Returns True if successful, False otherwise.
    """
    log_message(f"Starting: {description}", "info")
    log_message(f"Command: {' '.join(cmd)}", "debug")
    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        # Capture output line by line
        for line in process.stdout:
            st.session_state.logs.append(line.strip())
        process.wait()
        if process.returncode == 0:
            log_message(f"✅ {description} completed successfully.", "success")
            return True
        else:
            log_message(f"❌ {description} failed with return code {process.returncode}.", "error")
            return False
    except Exception as e:
        log_message(f"Exception during {description}: {e}", "error")
        return False

# ============================================================================
# Sidebar: Configuration
# ============================================================================
with st.sidebar:
    st.header("⚙️ Configuration")
    # Input dataset
    dataset_path = st.text_input(
        "Dataset CSV",
        value="data/input_dataset.csv",
        help="Path to the input CSV file (relative to project root)."
    )
    target_column = st.text_input("Target Column", value="target")
    min_perc_valid = st.slider(
        "Min % Valid (preprocessing)",
        min_value=0.0,
        max_value=0.2,
        value=0.05,
        step=0.01,
        format="%.2f",
    )
    perc_selected = st.slider(
        "Features to select (%)",
        min_value=0.05,
        max_value=0.95,
        value=0.20,
        step=0.01,
    )
    allowance = st.number_input("Allowance (± features)", min_value=0, max_value=10, value=1)
    perc_test = st.slider("Test split (%)", min_value=0.1, max_value=0.5, value=0.30)
    classifier = st.selectbox(
        "Classifier",
        options=["random_forest", "logistic_regression", "decision_tree"],
        index=0,
    )
    seed = st.number_input("Random Seed", min_value=0, max_value=9999, value=42)

    st.divider()
    if st.button("🗑️ Clear Logs", use_container_width=True):
        clear_logs()
        st.rerun()

    if st.button("🧹 Reset All", use_container_width=True):
        st.session_state.current_step = 0
        st.session_state.output_dir = None
        clear_logs()
        st.rerun()

# ============================================================================
# Main area: Steps
# ============================================================================
col1, col2, col3, col4, col5 = st.columns(5)

# ---- Step 1: Preprocess ----
with col1:
    if st.button("1️⃣ Preprocess", use_container_width=True):
        clear_logs()
        # Create a temporary output directory or use a fixed one
        out_dir = Path("outputs")
        out_dir.mkdir(exist_ok=True)
        st.session_state.output_dir = str(out_dir)

        normalized_csv = out_dir / "normalized.csv"
        preproc_json = out_dir / "preprocessing_result.json"

        cmd = [
            sys.executable,
            "src/qubo_project/preprocessing.py",
            "--input", dataset_path,
            "--target", target_column,
            "--out-data", str(normalized_csv),
            "--out-json", str(preproc_json),
            "--min-perc-valid", str(min_perc_valid),
        ]
        success = run_command(cmd, "Preprocessing")
        if success:
            st.session_state.current_step = 1
            st.session_state.normalized_csv = str(normalized_csv)
            st.session_state.preproc_json = str(preproc_json)
        st.rerun()

# ---- Step 2: Feature Selection ----
with col2:
    if st.button("2️⃣ Feature Selection", use_container_width=True):
        if st.session_state.current_step < 1:
            st.error("Please run Preprocessing first.")
        else:
            clear_logs()
            out_dir = Path(st.session_state.output_dir)
            reduced_train = out_dir / "training_reduced.csv"
            reduced_test = out_dir / "test_reduced.csv"
            opt_csv = out_dir / "optimizations.csv"
            fs_json = out_dir / "feature_selection_result.json"

            cmd = [
                sys.executable,
                "src/qubo_project/feature_selection.py",
                "--in-normalized", st.session_state.normalized_csv,
                "--out-train", str(reduced_train),
                "--out-test", str(reduced_test),
                "--out-optimizations", str(opt_csv),
                "--out-json", str(fs_json),
                "--target", target_column,
                "--perc-selected", str(perc_selected),
                "--allowance", str(allowance),
                "--perc-test", str(perc_test),
                "--seed", str(seed),
                "--alpha-computations", "50",
            ]
            success = run_command(cmd, "Feature Selection")
            if success:
                st.session_state.current_step = 2
                st.session_state.reduced_train = str(reduced_train)
                st.session_state.reduced_test = str(reduced_test)
                st.session_state.fs_json = str(fs_json)
            st.rerun()

# ---- Step 3: Train ----
with col3:
    if st.button("3️⃣ Train Model", use_container_width=True):
        if st.session_state.current_step < 2:
            st.error("Please run Feature Selection first.")
        else:
            clear_logs()
            out_dir = Path(st.session_state.output_dir)
            model_path = out_dir / "model.joblib"
            metrics_json = out_dir / "training_metrics.json"

            cmd = [
                sys.executable,
                "src/qubo_project/model.py",
                "train",
                "--classifier", classifier,
                "-in-reduced", st.session_state.reduced_train,
                "-target", target_column,
                "-out-model", str(model_path),
                "-out-metrics", str(metrics_json),
                "-seed", str(seed),
            ]
            success = run_command(cmd, "Training")
            if success:
                st.session_state.current_step = 3
                st.session_state.model_path = str(model_path)
                st.session_state.metrics_json = str(metrics_json)
            st.rerun()

# ---- Step 4: Predict ----
with col4:
    if st.button("4️⃣ Predict", use_container_width=True):
        if st.session_state.current_step < 3:
            st.error("Please train a model first.")
        else:
            clear_logs()
            out_dir = Path(st.session_state.output_dir)
            pred_csv = out_dir / "predictions.csv"
            stats_json = out_dir / "classification_stats.json"

            cmd = [
                sys.executable,
                "src/qubo_project/model.py",
                "predict",
                "-input-testset", st.session_state.reduced_test,
                "-target", target_column,
                "-model", st.session_state.model_path,
                "-out-predictions", str(pred_csv),
                "-out-stats", str(stats_json),
            ]
            success = run_command(cmd, "Prediction")
            if success:
                st.session_state.current_step = 4
                st.session_state.pred_csv = str(pred_csv)
                st.session_state.stats_json = str(stats_json)
            st.rerun()

# ---- Step 5: View Outputs ----
with col5:
    if st.button("5️⃣ View Outputs", use_container_width=True):
        st.session_state.show_outputs = True
        st.rerun()

# ============================================================================
# Display Logs
# ============================================================================
st.divider()
st.subheader("📋 Pipeline Logs")
log_container = st.container(height=400)
with log_container:
    for line in st.session_state.logs[-200:]:
        st.text(line)

# ============================================================================
# Display Outputs (if requested)
# ============================================================================
if st.session_state.get("show_outputs", False):
    st.divider()
    st.subheader("📊 Results")
    cols = st.columns(3)

    # Show preprocessing stats
    if st.session_state.get("preproc_json"):
        with cols[0]:
            st.write("**Preprocessing**")
            try:
                with open(st.session_state.preproc_json) as f:
                    data = json.load(f)
                    st.json(data)
            except:
                st.write("No data yet.")

    # Show feature selection stats
    if st.session_state.get("fs_json"):
        with cols[1]:
            st.write("**Feature Selection**")
            try:
                with open(st.session_state.fs_json) as f:
                    data = json.load(f)
                    st.json(data)
            except:
                st.write("No data yet.")

    # Show classification stats
    if st.session_state.get("stats_json"):
        with cols[2]:
            st.write("**Classification Metrics**")
            try:
                with open(st.session_state.stats_json) as f:
                    data = json.load(f)
                    st.json(data)
            except:
                st.write("No data yet.")

    # Show predictions preview
    if st.session_state.get("pred_csv"):
        st.write("**Predictions Preview**")
        try:
            df = pd.read_csv(st.session_state.pred_csv, nrows=10)
            st.dataframe(df)
        except:
            st.write("Could not load predictions CSV.")

    # Show reduced dataset previews
    if st.session_state.get("reduced_train"):
        with st.expander("Training Set (reduced)"):
            df = pd.read_csv(st.session_state.reduced_train, nrows=5)
            st.dataframe(df)

    if st.session_state.get("reduced_test"):
        with st.expander("Test Set (reduced)"):
            df = pd.read_csv(st.session_state.reduced_test, nrows=5)
            st.dataframe(df)