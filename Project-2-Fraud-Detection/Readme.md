# 🚀 Project 2: Fraud Detection Pipeline
---

## 📌 The Vibe

Building a **fraud detection** model that actually works on imbalanced data.  
We use **SMOTE** to balance the classes and evaluate with **Precision, Recall, F1, ROC‑AUC** – because **Accuracy is a lie** (no cap).

---

## 🧰 Tech Stack

```bash
pandas · numpy · matplotlib · seaborn · scikit‑learn · imbalanced‑learn
```

## ⚡ Quick Start

1. Make sure `Project1-Output.csv` is in the same folder.
2. Fire up the script:
   ```bash
   python Project2-DecodeLabs.py
   ```
3. Boom – you get:
   - Console logs with model scores.
   - `fraud_detection_results.png` – dope visualizations (confusion matrices, ROC/PR curves, feature importance, model showdown).

---

## 📊 Results (Spoiler: Random Forest Wins)

| Model               | Precision | Recall | F1    | ROC‑AUC |
|---------------------|-----------|--------|-------|---------|
| Logistic Regression | 0.82      | 0.78   | 0.80  | 0.88    |
| **Random Forest**   | **0.88**  | **0.85**| **0.86**| **0.92** |

> **MVP:** Random Forest – catches more fraud (higher recall) and keeps a solid precision.

---

## 🔥 Key Moves

- **Leak‑proof pipelines** using `imblearn.pipeline.Pipeline` – SMOTE only touches training folds.
- SMOTE **never** sees test data – zero leakage.
- Feature importance from Random Forest tells us what really matters.
- Full suite of confusion matrices, ROC/PR curves, and side‑by‑side comparisons.

---

## 💡 Lessons Learned (Drop the Mic)

- Accuracy is for amateurs – use precision & recall.
- **Never** SMOTE or scale before train/test split – that’s data cheating.
- Tree‑based models? They don’t need scaling – flex.
- Always resample **inside** cross‑validation – keep it clean.

---

## 🛠️ Built With

- Python 3.14
- pandas, numpy – data wrangling
- scikit‑learn – modeling & metrics
- imbalanced‑learn – SMOTE & pipelines
- matplotlib, seaborn – viz game strong

---

## 👩‍💻 Author

**Javeria Faisal**  
Data Science Intern @ DecodeLabs

---
