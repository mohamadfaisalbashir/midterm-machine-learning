# 🔍 Fraud Detection — End-to-End MLOps Pipeline on Azure ML

> **Author:** Mohamad Faisal Bashir · Class: TK-47-04 · NIM: 101032300036

An end-to-end machine learning pipeline for detecting fraudulent financial transactions using the [IEEE-CIS Fraud Detection](https://www.kaggle.com/c/ieee-fraud-detection) dataset. The project covers the full MLOps lifecycle: data preprocessing, feature engineering, model training with LightGBM, automated hyperparameter tuning via Optuna, and experiment tracking with MLflow — deployed on **Azure Machine Learning**.

---

## 📋 Table of Contents

- [Project Overview](#-project-overview)
- [Pipeline Workflow](#-pipeline-workflow)
- [Tech Stack](#-tech-stack)
- [Repository Structure](#-repository-structure)
- [Azure ML Setup Guide](#-azure-ml-setup-guide)
- [Running the Notebook](#-running-the-notebook)
- [Results & Metrics](#-results--metrics)
- [MLflow Experiment Tracking](#-mlflow-experiment-tracking)
- [Cost Management](#-cost-management)

---

## 📌 Project Overview

Payment fraud is a severe imbalanced classification problem — fraudulent transactions represent only **~3.5%** of all records. This project builds a robust detection system that handles:

- **Severe class imbalance** via LightGBM's `is_unbalance` parameter
- **High-dimensional sparse data** from merged transaction + identity tables (700+ features)
- **Missing value patterns** treated as informative signals (sentinel imputation with `-999`)
- **Automated hyperparameter search** across 50 Optuna trials with Bayesian optimization
- **Full experiment reproducibility** via MLflow nested run tracking on Azure ML

**Dataset files:**

| File | Description |
|------|-------------|
| `train_transaction.csv` | Financial transaction features — training set |
| `train_identity.csv` | Device & network metadata — training set |
| `test_transaction.csv` | Financial transaction features — test set |
| `test_identity.csv` | Device & network metadata — test set |

**Target variable:** `isFraud` — binary (0 = legitimate, 1 = fraudulent)

---

## 🔄 Pipeline Workflow

```
1. Environment Setup & Library Imports
2. Data Ingestion & Memory Optimization
3. Exploratory Data Analysis (EDA)
4. Data Preprocessing
5. Feature Engineering
6. Train / Validation Split (80/20, stratified)
7. Hyperparameter Tuning with Optuna (50 trials) + MLflow Tracking
8. Model Evaluation (AUC-ROC, PR-AUC, F1, Confusion Matrix)
9. Final Prediction & Submission
```

### Key Design Decisions

| Step | Decision | Rationale |
|------|----------|-----------|
| Memory optimization | Downcast numeric dtypes (`float64 → float32/16`, `int64 → int8/16/32`) | Reduces memory footprint by 50–70% on the raw 500MB+ dataset |
| Missing value imputation | `fillna(-999)` sentinel | Lets LightGBM trees explicitly learn that missingness is itself informative |
| Label encoding | Fit on union of train + test values | Prevents unseen-label errors when transforming test data |
| Feature engineering | Extract `Transaction_Day` and `Transaction_Hour` from `TransactionDT` delta | Encodes temporal fraud patterns a tree cannot recover from a raw continuous delta |
| Class imbalance | `is_unbalance=True` in LightGBM | Auto-adjusts class weights; no manual resampling required |
| Hyperparameter search | Optuna Bayesian optimization, 50 trials, early stopping (`rounds=30`) | More sample-efficient than grid/random search; prevents overfitting within each trial |

---

## 🛠 Tech Stack

| Category | Library / Tool |
|----------|---------------|
| Core ML | `lightgbm` |
| Hyperparameter Tuning | `optuna` |
| Experiment Tracking | `mlflow`, `azureml-mlflow` |
| Data Processing | `pandas`, `numpy`, `scikit-learn` |
| Visualization | `matplotlib`, `seaborn`, `plotly` |
| Infrastructure | Azure Machine Learning (Compute Instance) |

---

## 📁 Repository Structure

```
/
├── fraud_detection.ipynb    # Main notebook
├── folder-data/             # Dataset directory (not tracked in Git)
│   ├── train_transaction.csv
│   ├── train_identity.csv
│   ├── test_transaction.csv
│   └── test_identity.csv
└── README.md
```

> **Note:** Dataset files are not included due to size. Download from [Kaggle IEEE-CIS Fraud Detection](https://www.kaggle.com/c/ieee-fraud-detection/data) and place them inside `folder-data/`.

---

## ☁️ Azure ML Setup Guide

### Step 1 — Create Azure ML Workspace

1. Go to [portal.azure.com](https://portal.azure.com) and search for **Azure Machine Learning**.
2. Click **+ Create > New Workspace** and fill in:
   - **Resource group:** e.g., `rg-mlops-faisal`
   - **Workspace name:** e.g., `ml-workspace-faisal`
   - **Region:** Southeast Asia (Singapore)
3. Click **Review + Create → Go to resource → Launch studio** → opens [ml.azure.com](https://ml.azure.com).

### Step 2 — Create a Compute Instance (VM)

1. In Azure ML Studio, go to **Compute > Compute instances > + New**.
2. Name the instance and select size:
   - **Recommended:** `Standard_E4ds_v4` (4 cores, 32 GB RAM)
   - **Alternative:** `Standard_D8s_v3` or `Standard_DS4_v2`
3. Click **Create** and wait ~2–5 minutes for status to become **Running** (green dot).

### Step 3 — Upload Files via JupyterLab

1. On the Compute page, click **JupyterLab** on your running instance.
2. In the file explorer, create a folder named exactly `folder-data`.
3. Upload all four CSV files **inside** `folder-data/`.
4. Return to the root directory and upload `fraud_detection.ipynb`.

Expected structure on the VM:

```
/ (Root)
 ├── folder-data/
 │    ├── train_transaction.csv
 │    ├── train_identity.csv
 │    ├── test_transaction.csv
 │    └── test_identity.csv
 └── fraud_detection.ipynb
```

---

## ▶️ Running the Notebook

1. Open `fraud_detection.ipynb` in JupyterLab.
2. Run **Cell 0** (the `pip install` cell) with `Shift + Enter` and wait for it to complete.
3. Restart the kernel: **Kernel > Restart Kernel... > Restart**.
4. Run all cells: **Run > Run All Cells**.

> ⚠️ The kernel restart after installation is mandatory — newly installed libraries won't be available until the kernel reloads.

---

## 📊 Results & Metrics

The model is evaluated on the held-out 20% validation set using:

- **AUC-ROC** — primary metric optimized by Optuna
- **PR-AUC** (Precision-Recall AUC) — better suited for imbalanced datasets
- **F1 Score**
- **Confusion Matrix**
- **ROC Curve**

Optuna searches across the following hyperparameter space:

| Hyperparameter | Search Range |
|----------------|-------------|
| `learning_rate` | 0.01 – 0.3 (log-uniform) |
| `num_leaves` | 31 – 256 |
| `max_depth` | -1 – 15 |
| `feature_fraction` | 0.5 – 1.0 |
| `n_estimators` | 100 – 1000 |

---

## 📈 MLflow Experiment Tracking

After running the notebook, view results in Azure ML Studio:

1. Go to **Jobs** (or **Experiments**) in the left panel.
2. Select **`Fraud_Detection_LGBM_Optuna`**.
3. Click any run and navigate to the **Metrics** or **Images** tab to view:
   - Optuna optimization history (AUC vs. trial number)
   - Parameter importance chart
   - Per-trial hyperparameter and metric logs

Each Optuna trial is logged as a **nested child run** under the parent experiment for clean, hierarchical comparison across all 50 trials.

---

## 💡 Cost Management

> ⚠️ **Always stop your VM when not in use.**

1. Go to **Compute** in Azure ML Studio.
2. Check the box next to your VM name and click **Stop**.
3. Confirm status changes to **Stopped** — you will not be charged for compute while stopped.

---

## 📄 License

Submitted as academic coursework. Dataset usage is subject to [Kaggle's competition rules](https://www.kaggle.com/c/ieee-fraud-detection/rules).
