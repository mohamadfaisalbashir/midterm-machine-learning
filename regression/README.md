# 📈 Midterm Regression — MLOps Pipeline on Azure ML

> **Author:** Mohamad Faisal Bashir · Class: TK-47-04 · NIM: 101032300036

An end-to-end regression pipeline built with MLOps best practices, featuring automated hyperparameter tuning via Optuna and full experiment tracking with MLflow — deployed on **Azure Machine Learning**.

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

This project implements a supervised regression pipeline to predict a continuous target variable from the provided dataset. The pipeline follows MLOps principles to ensure reproducibility, scalability, and observability at every step.

**Dataset file:**

| File | Description |
|------|-------------|
| `midterm-regresi-dataset.csv` | Tabular dataset with features and continuous target variable |

---

## 🔄 Pipeline Workflow

```
1. Environment Setup & Library Imports
2. Data Loading & Exploratory Data Analysis (EDA)
3. Data Cleaning & Preprocessing
4. Feature Engineering & Selection
5. Train / Validation Split
6. Model Training with Hyperparameter Tuning (Optuna)
7. MLflow Experiment Tracking
8. Model Evaluation (RMSE, MAE, R²)
9. Final Prediction
```

---

## 🛠 Tech Stack

| Category | Library / Tool |
|----------|---------------|
| Core ML | `lightgbm` / `scikit-learn` |
| Hyperparameter Tuning | `optuna` |
| Experiment Tracking | `mlflow`, `azureml-mlflow` |
| Data Processing | `pandas`, `numpy`, `scikit-learn` |
| Visualization | `matplotlib`, `seaborn`, `plotly` |
| Infrastructure | Azure Machine Learning (Compute Instance) |

---

## 📁 Repository Structure

```
/
├── midterm_regression_mlops.ipynb    # Main notebook
├── folder-data/                      # Dataset directory (not tracked in Git)
│   └── midterm-regresi-dataset.csv
└── README.md
```

> **Note:** The dataset file is not included in this repository. Place it inside `folder-data/` before running the notebook.

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
3. Upload `midterm-regresi-dataset.csv` **inside** `folder-data/`.
4. Return to the root directory and upload `midterm_regression_mlops.ipynb`.

Expected structure on the VM:

```
/ (Root)
 ├── folder-data/
 │    └── midterm-regresi-dataset.csv
 └── midterm_regression_mlops.ipynb
```

---

## ▶️ Running the Notebook

1. Open `midterm_regression_mlops.ipynb` in JupyterLab.
2. Run **Cell 0** (the `pip install` cell) with `Shift + Enter` and wait for it to complete.
3. Restart the kernel: **Kernel > Restart Kernel... > Restart**.
4. Run all cells: **Run > Run All Cells**.

> ⚠️ The kernel restart after installation is mandatory — newly installed libraries won't be available until the kernel reloads.

---

## 📊 Results & Metrics

The model is evaluated on the held-out validation set using:

- **RMSE** (Root Mean Squared Error) — primary metric
- **MAE** (Mean Absolute Error)
- **R²** (Coefficient of Determination)

---

## 📈 MLflow Experiment Tracking

After running the notebook, view results in Azure ML Studio:

1. Go to **Jobs** (or **Experiments**) in the left panel.
2. Select **`Midterm_Regression_Pipeline`**.
3. Click any run and navigate to the **Metrics** or **Images** tab to view:
   - Optuna optimization history
   - Parameter importance chart
   - Per-trial hyperparameter and metric logs

---

## 💡 Cost Management

> ⚠️ **Always stop your VM when not in use.**

1. Go to **Compute** in Azure ML Studio.
2. Check the box next to your VM name and click **Stop**.
3. Confirm status changes to **Stopped** — you will not be charged for compute while stopped.

---

## 📄 License

Submitted as academic coursework.
