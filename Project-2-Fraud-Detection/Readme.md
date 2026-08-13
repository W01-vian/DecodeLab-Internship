# 🚀 Project 2: Fraud Detection Pipeline

**DecodeLabs – Batch 2026**  
*Data Science Intern – Javeria Faisal*
---
## 📌 Overview

Build a **fraud detection** model for imbalanced transaction data.  
Use **SMOTE** to handle class imbalance and evaluate with **Precision, Recall, F1, ROC‑AUC** – never Accuracy.
---
## 📦 Requirements

```bash
pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn
```
> If `pip` not found, use `python -m pip install ...`
---
## ▶️ How to Run

1. Ensure `Project1-Output.csv` is in the same folder.
2. Run the script:
```bash
python Project2.py
```
3. Output: `fraud_detection_results.png` (visualizations) and console reports.
---
## 📊 Results (Example)

| Model               | Precision | Recall | F1    | ROC‑AUC |
|---------------------|-----------|--------|-------|---------|
| Logistic Regression | 0.82      | 0.78   | 0.80  | 0.88    |
| **Random Forest**   | **0.88**  | **0.85**| **0.86**| **0.92** |

*Best model: Random Forest* – higher recall (catches more fraud).
---
## 🔍 Key Features

- Leak‑free pipelines using `imblearn.pipeline.Pipeline`
- SMOTE applied **only on training folds** inside cross‑validation
- Feature importance from Random Forest
- Confusion matrices, ROC/PR curves, and model comparison
---
## ✅ What We Learned

- Accuracy is misleading → use precision & recall.
- Never apply SMOTE or scaling before train/test split.
- Tree‑based models don’t need scaling.
- Always resample **inside** cross‑validation.
---
