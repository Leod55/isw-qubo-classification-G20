# src/qubo_project/gui.py
import streamlit as st
import subprocess
import os
from pathlib import Path

st.set_page_config(page_title="QUBO Classification GUI", layout="wide")
st.title("🧠 QUBO Feature Selection & Classification")

# Sidebar for common configs
st.sidebar.header("Configuration")
target_col = st.sidebar.text_input("Target Column", value="target")
min_perc_valid = st.sidebar.slider("Min % Valid", 0.0, 0.2, 0.05, 0.01)

# Main buttons
col1, col2, col3 = st.columns(3)
col4, col5, col6 = st.columns(3)

with col1:
    if st.button("1. Select Dataset", use_container_width=True):
        st.info("Dataset selection: you can provide a path in the sidebar or via file upload.")
        # Placeholder: in a real implementation, use st.file_uploader or a text input.

with col2:
    if st.button("2. Preprocess", use_container_width=True):
        st.info("Running preprocessing... (stub)")
        # In future: subprocess.run(["python", "preprocessing.py", ...])
        st.success("Preprocessing completed (stub).")

with col3:
    if st.button("3. Feature Selection (QUBO)", use_container_width=True):
        st.info("Running QUBO feature selection... (stub)")
        st.success("Feature selection completed (stub).")

with col4:
    if st.button("4. Train Model", use_container_width=True):
        st.info("Training classifier... (stub)")
        st.success("Training completed (stub).")

with col5:
    if st.button("5. Predict", use_container_width=True):
        st.info("Running predictions... (stub)")
        st.success("Predictions completed (stub).")

with col6:
    if st.button("6. View Outputs", use_container_width=True):
        st.info("Displaying results (stub).")
        # Show dummy metrics or CSV preview

# Placeholder for displaying results
st.subheader("Latest Results (Stub)")
st.json({
    "accuracy": 0.90,
    "f1_class_1": 0.774,
    "roc_auc": 0.88,
})

# --- How to run: streamlit run src/qubo_project/gui.py ---