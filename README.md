👤 **Author Profile**
* **Name:** Mohamad Faisal Bashir
* **Class:** TK-47-04
* **NIM:** 101032300036
* **Institution:** Computer Engineering, Telkom University  

This repository showcases a range of solutions from unsupervised behavioral segmentation to high-stakes fraud detection, with a heavy emphasis on **MLOps practices**, **experiment tracking**, and **cloud-native development**.

---

## 📁 Projects Overview

### 1. 🧠 Customer Clustering Pipeline
**Platform:** ![Google Colab](https://img.shields.io/badge/Google%20Colab-F9AB00?style=flat&logo=googlecolab&logoColor=white)  
**File:** `customer_clustering_pipeline.ipynb`

A production-grade segmentation workflow designed to identify customer archetypes through unsupervised learning.

* **Objective:** Transform raw transactional behavior into actionable marketing segments.
* **Methodology:** * **Preprocessing:** Robust outlier handling and feature scaling.
    * **Dimensionality Reduction:** PCA implementation to capture 90% variance.
    * **Optimal k Selection:** Multi-metric consensus using Elbow Method, Silhouette Score, and Davies-Bouldin Index.
* **Algorithms:** K-Means (Primary), Hierarchical Clustering (Validation), and DBSCAN (Anomalies).
* **Key Results:** Identified distinct clusters such as *VIP Spenders*, *Dormant Users*, and *Transactors*, enabling targeted CRM strategies.

---

### 2. 📈 End-to-End Regression Pipeline with MLOps
**Platform:** ![Azure Machine Learning](https://img.shields.io/badge/Azure%20ML-0078D4?style=flat&logo=microsoftazure&logoColor=white)  
**File:** `midterm_regression_mlops.ipynb`

A highly reproducible regression framework integrated with the Azure ML ecosystem for professional lifecycle management.

* **Objective:** Predict continuous target variables with high precision while maintaining strict experiment tracking.
* **Models:** Evaluated a suite of regressors including XGBoost, LightGBM, CatBoost, and Gradient Boosting.
* **MLOps Integration (Azure ML + MLflow):** * Centralized logging of MSE, RMSE, MAE, R², and MAPE.
    * Artifact versioning and model registration.
* **Interpretability:** Integrated **LIME** (Local Interpretable Model-agnostic Explanations) to provide transparency for individual predictions, crucial for business decision-making.

---

### 3. 🔍 End-to-End Fraud Detection Pipeline
**Platform:** ![Azure Machine Learning](https://img.shields.io/badge/Azure%20ML-0078D4?style=flat&logo=microsoftazure&logoColor=white)  
**File:** `fraud_detection.ipynb`

An advanced MLOps pipeline tackling the complex problem of financial fraud detection in highly imbalanced datasets.

* **Objective:** Predict the probability of fraudulent transactions (`isFraud`) using a data-centric approach.
* **Strategy:**
    * **Handling Imbalance:** Optimized LightGBM parameters (`is_unbalance=True`) and specialized sampling.
    * **Auto-ML & Tuning:** Utilized **Optuna** for Bayesian hyperparameter optimization over 50+ trials.
* **Workflow:** * Experiment tracking on **Azure ML Studio** via MLflow.
    * Evaluation focusing on **ROC-AUC** and **Precision-Recall** curves to ensure robust detection of minority fraud classes.
* **Performance:** Scalable implementation capable of processing 700+ features from merged transaction/identity tables.

---

## 🛠️ Tech Stack

| Category | Tools & Libraries |
| :--- | :--- |
| **Cloud Platforms** | Microsoft Azure Machine Learning, Google Colab |
| **MLOps & Tracking** | MLflow, Azure ML Studio |
| **Optimization** | Optuna |
| **Explainability** | LIME |
| **Algorithms** | LightGBM, XGBoost, CatBoost, Scikit-Learn |
| **Data Science** | Pandas, NumPy, Matplotlib, Seaborn |

---

## 🚀 Environment & Setup

### For Azure ML Projects (Regression & Fraud)
1.  **Workspace:** Ensure you have access to an Azure Machine Learning workspace.
2.  **Tracking URI:** The notebooks are configured to log to the Azure ML tracking URI.
3.  **Compute:** Tested on `Standard_DS3_v2` (4 cores, 14 GB RAM).

### For Google Colab Projects (Clustering)
1.  **Drive Mounting:** The notebook includes cells to mount Google Drive for data persistence.
2.  **GPU Acceleration:** While not strictly required for K-Means, T4 GPU is recommended for faster PCA processing.
