# 🧠 Customer Clustering Pipeline
> **Author:** Mohamad Faisal Bashir · Class: TK-47-04 · NIM: 10103230003
### Unsupervised Learning — K-Means, Hierarchical & DBSCAN
---

## 📌 Overview
This repository contains a complete, production-grade machine learning pipeline for customer segmentation using unsupervised learning techniques. The project focuses on discovering natural groupings within customer data to uncover actionable behavioral archetypes for targeted marketing, CRM, and business strategy.

The pipeline rigorously handles end-to-end data processing, from missing value imputation and outlier treatment to feature engineering, dimensionality reduction, and parallel evaluation of three distinct clustering algorithms.

---

## 🛠️ Tech Stack & Libraries
* **Data Manipulation:** `pandas`, `numpy`
* **Machine Learning:** `scikit-learn` (K-Means, DBSCAN, Agglomerative Clustering, PCA, Imputers, Scalers)
* **Statistical Analysis:** `scipy` (Hierarchical linkages and dendrograms)
* **Data Visualization:** `matplotlib`, `seaborn`

---

## ⚙️ Pipeline Architecture

This notebook follows a structured, robust methodology designed for production readiness:

| Stage | Decision & Methodology | Rationale |
|---|---|---|
| **Missing Values** | Tiered: Median (<5%) / KNN (5–30%) / Drop (>30%) | Median is robust to skew; KNN preserves feature correlations. |
| **Outlier Treatment** | IQR Winsorization (1st–99th percentile) | Capping extreme values prevents centroid distortion without losing valuable rows. |
| **Feature Engineering** | Behavioral ratios (e.g., Utilization, Repayment) | Derived features often reveal richer behavioral signals than raw metrics. |
| **Scaling** | `RobustScaler` | Scales data by IQR, making it highly resistant to any remaining outliers. |
| **Dimensionality Reduction**| PCA (Principal Component Analysis) | Reduces noise and speeds up clustering by keeping components that explain ≥ 90% of the variance. |
| **Optimal *k* Selection** | Elbow Method, Silhouette Score, Davies-Bouldin | Uses multi-metric consensus to find the mathematical "sweet spot" for cluster count. |
| **Clustering Algorithms** | K-Means, Hierarchical (Ward), DBSCAN | Triangulates results across centroid-based, connectivity-based, and density-based paradigms. |

---

## 📂 Notebook Structure

1. **Data Loading & Initial Exploration:** Data profiling and structural auditing.
2. **Data Cleaning & Missing Value Handling:** Safe, tiered imputation.
3. **Outlier Detection & Treatment:** Visualizing and Winsorizing extremes.
4. **Feature Engineering:** Crafting financial/behavioral ratios (e.g., `BALANCE_TO_LIMIT_RATIO`, `CASH_ADVANCE_RATIO`).
5. **Preprocessing:** Scaling and PCA projection.
6. **K-Means Clustering:** Sweeping for optimal *k* and evaluating cluster cohesion.
7. **Hierarchical Clustering:** Building dendrograms to validate natural breaks.
8. **DBSCAN Clustering:** Detecting arbitrary shapes and noise/anomaly customers.
9. **Algorithm Comparison:** Side-by-side metric evaluation (Silhouette, Davies-Bouldin).
10. **Cluster Profiling & Business Interpretation:** Generating radar charts, box plots, and dynamic customer personas.
11. **Export Results:** Saving clustered datasets and profiles for downstream use.

---

## 💡 Key Takeaways

* **K-Means** delivered the most interpretable and stable clusters, making it the primary choice for CRM targeting.
* **Hierarchical clustering** visually confirmed the cluster structure via the dendrogram.
* **DBSCAN** effectively identified potential "noise" or anomaly customers, which warrant separate fraud or churn reviews.
* **Feature Engineering** (specifically calculated ratios) improved cluster separation significantly more than using raw features alone.
* Each identified cluster represents a distinct behavioral archetype with clear, actionable marketing implications (e.g., Transactors, Revolvers, VIP Spenders).

---

## 🚀 How to Run

1. Clone the repository and ensure you have the required libraries installed.
```bash
pip install pandas numpy scikit-learn matplotlib seaborn scipy
```
2. Open the Jupyter Notebook:
```bash
jupyter notebook customer_clustering_pipeline.ipynb
```
3. Dataset Path: The notebook is currently configured to load data from Google Drive (`/content/drive/MyDrive/ML/clusteringmidterm.csv`). If running locally, update the `DATA_PATH` variable in Section 1 to point to your local dataset.
4. Run all cells sequentially.

📈 Recommended Next Steps
- **Domain Validation:** Validate the resulting clusters and personas with domain experts and business stakeholders.
- **Naming Convention:** Assign concrete business names to each cluster (e.g., "High-Value Transactors", "Dormant Accounts").
- **Predictive Modeling:** Build a supervised classifier (e.g., XGBoost, LightGBM) using the cluster labels to predict segments for new customers in real-time.
- **Monitor Drift:** Schedule quarterly clustering re-runs to observe how spending patterns and segment sizes evolve over time.
- **A/B Testing:** Deploy targeted campaigns per segment and measure the business uplift compared to a generic baseline.
