# QUBO Feature Selection & Binary Classification

**Project:** ISW 2025-26 – Python QUBO Classification  
**Group:** G20  
**Repository:** [isw-qubo-classification-G20](https://github.com/Leod55/isw-qubo-classification-G20)

---

## 📌 Project Overview

This project implements a complete **binary classification pipeline** with **QUBO‑based feature reduction**. It is designed for credit‑risk assessment (or any binary classification task) and supports datasets with **>1.5 million rows**. The pipeline consists of four phases:

1. **Preprocessing** – Z‑score normalisation and removal of near‑empty columns.
2. **Feature Selection (QUBO)** – Spearman‑correlation‑based QUBO formulation with Simulated Annealing, driven by an `α` parameter to select a target percentage of features (± allowance).
3. **Model Training** – Three classifiers (mandatory Random Forest, plus Logistic Regression and Decision Tree).
4. **Prediction & Evaluation** – Generates per‑record predictions (CSV) and comprehensive metrics (JSON).

The system provides both a **command‑line interface (CLI)** for automated evaluation and a **Streamlit GUI** for interactive use.

---

## 🚀 Quick Start

### Prerequisites
- **Python 3.11** or higher
- **pip** (package installer)

### 1. Clone the Repository
```bash
git clone https://github.com/Leod55/isw-qubo-classification-G20.git
cd isw-qubo-classification-G20
```

### 2. Set Up a Virtual Environment (Recommended)
```bash
python3.11 -m venv .venv
source .venv/bin/activate          # On Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Verify the Installation
```bash
python -c "import pandas, numpy, sklearn, scipy, joblib, streamlit; print('OK')"
pytest tests/ -v
```

All tests should pass.

---

## 📂 Project Structure

```
isw-qubo-classification-G20/
├── data/
│   └── sample_test_dataset.csv      # Small dataset for pytest (mandatory)
├── src/
│   └── qubo_project/
│       ├── __init__.py
│       ├── utils.py                 # Logging & JSON helpers
│       ├── preprocessing.py         # fit_normalize (CLI + function)
│       ├── feature_selection.py     # select_features (QUBO + SA)
│       ├── model.py                 # train & predict (CLI subcommands)
│       └── gui.py                   # Streamlit GUI (6 actions)
├── tests/
│   ├── __init__.py
│   ├── test_feature_selection.py
│   ├── test_model.py
│   └── test_preprocessing.py
├── reports/
│   ├── llm_usage/                   # LLM interaction logs & generated files
│   │   ├── files/                   # Prompts files
│   │   ├── LOG-G20-01.md
│   │   ├── LOG-G20-02.md
│   │   ├── LOG-G20-03.md
│   │   └── LOG-G20-04.md
│   └── project_report.yaml          # Final project report
├── outputs/                         # Runtime outputs (created on execution)
├── logs/                            # Runtime logs (created on execution)
├── requirements.txt                 # Production dependencies
├── README.md                        # This file
├── group_info.yaml                  # Group metadata
└── .gitignore
```

---

## 🧪 Running Tests

Execute the full test suite (must use `data/sample_test_dataset.csv`):

```bash
pytest tests/ -v
```

To run a specific test file:
```bash
pytest tests/test_preprocessing.py -v
```

---

## 🖥️ Command‑Line Interface (CLI)

All CLI commands conform to the specification (Sections 11–12). Outputs are written to the `outputs/` directory.

### 1. Preprocessing
```bash
python src/qubo_project/preprocessing.py \
  --input data/input_dataset.csv \
  --target target \
  --out-data outputs/normalized.csv \
  --out-json outputs/preprocessing_result.json \
  --min-perc-valid 0.05
```

### 2. Feature Selection (QUBO)
```bash
python src/qubo_project/feature_selection.py \
  --in-normalized outputs/normalized.csv \
  --out-train outputs/training_reduced.csv \
  --out-test outputs/test_reduced.csv \
  --out-optimizations outputs/optimizations.csv \
  --out-json outputs/feature_selection_result.json \
  --target target \
  --perc-selected 0.20 \
  --allowance 1 \
  --perc-test 0.30 \
  --seed 42 \
  --alpha-computations 50
```

### 3. Train a Classifier
```bash
python src/qubo_project/model.py train \
  --classifier random_forest \
  -in-reduced outputs/training_reduced.csv \
  -target target \
  -out-model outputs/model.joblib \
  -out-metrics outputs/training_metrics.json \
  -seed 42
```

### 4. Predict on Test Set
```bash
python src/qubo_project/model.py predict \
  -input-testset outputs/test_reduced.csv \
  -target target \
  -model outputs/model.joblib \
  -out-predictions outputs/predictions.csv \
  -out-stats outputs/classification_stats.json
```

---

## 🖥️ Graphical User Interface (GUI)

The GUI provides a user‑friendly way to run the entire pipeline step‑by‑step.

### Launch the GUI
```bash
streamlit run src/qubo_project/gui.py
```

The interface opens in your browser at `http://localhost:8501`.

### GUI Actions
1. **Select Dataset** – via sidebar text input.
2. **Preprocess** – runs `preprocessing.py` with sidebar parameters.
3. **Feature Selection (QUBO)** – runs `feature_selection.py`.
4. **Train Model** – trains the selected classifier.
5. **Predict** – evaluates the model on the test set.
6. **View Outputs** – displays JSON stats and CSV previews.

All logs are displayed in the main panel.

---

## 📊 Output Files

| File | Description |
|------|-------------|
| `normalized.csv` | Full dataset after z‑score normalisation. |
| `preprocessing_result.json` | Statistics: `n_input_features`, `n_kept_features`, `dataset_size`, timings, `dropped_feature_names`. |
| `training_reduced.csv` | Training set with only selected features + target. |
| `test_reduced.csv` | Test set with only selected features + target. |
| `optimizations.csv` | Trace of α‑search: `alpha`, `time`, `n_features`, `cost`. |
| `feature_selection_result.json` | QUBO results: `selected_vector`, `selected_feature_names`, `alpha`, timings. |
| `model.joblib` | Trained classifier (scikit‑learn). |
| `training_metrics.json` | Training statistics (samples, features, time, in‑sample metrics). |
| `predictions.csv` | Per‑record predictions: `row_n`, `target`, `prediction`, `score`. |
| `classification_stats.json` | Evaluation metrics: accuracy, precision/recall/f1 per class, ROC‑AUC, confusion matrix. |

---

## 🔧 Dependencies

All dependencies are listed in `requirements.txt`. Key packages:

- `pandas` – Data loading and manipulation.
- `numpy` – Numerical operations for QUBO.
- `scikit-learn` – Classifiers, metrics, `StandardScaler`, `joblib`.
- `scipy` – Spearman correlation (`spearmanr`).
- `streamlit` – GUI framework.
- `pytest` – Test execution.
- `joblib` – Model serialisation.

Install all with:
```bash
pip install -r requirements.txt
```

---

## 📄 Group Information

The `group_info.yaml` file contains:

```yaml
group_id: "G20"
repository_url: "https://github.com/Leod55/isw-qubo-classification-G20.git"
students:
  - matricola: "66413"
    name: "Leonardo Dessanay"
  - matricola: "66528"
    name: "Francesca Del Zompo"
```

---

## 📝 LLM Interaction Logs

All interactions with the Large Language Model (DeepSeek) are documented in the `reports/llm_usage/` directory:

- `LOG-G20-01.md` – Setup, stubs, and CLI verification.
- `LOG-G20-02.md` – Preprocessing implementation and test fixes.
- `LOG-G20-03.md` – Feature selection, model training, and GUI integration.
- `LOG-G20-04.md` – Final comprehensive code review and verification.

Each log includes prompts, responses, generated files, and honest reporting of issues encountered.

---

## ✅ Compliance with Specifications

| Section | Requirement | Status |
|---------|-------------|--------|
| 7 | Preprocessing (z‑score, drop empty columns) | ✅ |
| 8 | QUBO feature selection (Spearman, α, SA) | ✅ |
| 9 | Three classifiers (Random Forest mandatory) | ✅ |
| 10 | Predictions CSV (`row_n, target, prediction, score`) | ✅ |
| 11 | Mandatory functions (`fit_normalize`, `select_features`, `train`, `predict`) | ✅ |
| 12 | CLI arguments exactly as specified | ✅ |
| 13 | `pytest` with `data/sample_test_dataset.csv` | ✅ |
| 14 | GUI (6 actions, Streamlit) | ✅ |
| 15 | `group_info.yaml` and project report | ✅ |
| 16 | Complete LLM interaction logs | ✅ |
| 17 | Reproducibility (requirements.txt, seed control) | ✅ |

---

## 🤝 Contributors

- **Leonardo Dessanay** (matricola 66413)
- **Francesca Del Zompo** (matricola 66528)

---

## 📧 Contact

For any issues, please open a GitHub issue or contact the group directly.

---

**© 2026 – ISW Project, University of Cagliari**