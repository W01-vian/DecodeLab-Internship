# 🚀 Project 4: NLP & Sentiment Analysis
---

## 📌 Overview

This pipeline ingests raw customer reviews, scrubs them clean, converts words into numeric vectors (TF‑IDF), and trains a **Naïve Bayes** classifier to detect sentiment (Positive / Negative).  
**No accuracy metric** – we evaluate using **Precision, Recall, F1, and ROC‑AUC** (the real deal for imbalanced data).

---

## 🧠 Tech Stack

- Python 3.8+  
- pandas, numpy, matplotlib, seaborn  
- scikit‑learn (TfidfVectorizer, Naive Bayes, cross‑validation)  
---

## ⚙️ Pipeline (5 Steps to Mathematical Certainty)

1. **Preprocess** – lowercasing, punctuation removal, tokenization.  
2. **Stop‑word removal** – **keeps negations** (`not`, `no`, `never`) because they flip sentiment.  
3. **Lemmatisation** – with POS tags (e.g., “went” → “go”) to preserve meaning.  
4. **TF‑IDF Vectorisation** – unigrams + bigrams, stored as **sparse CSR** matrix to save memory.  
5. **Naïve Bayes** – MultinomialNB (balanced) or ComplementNB (imbalanced) with Laplace smoothing.

---

## 🚦 How to Run

```bash
# 1. Install dependencies (if using full version with NLTK)
pip install pandas numpy matplotlib seaborn scikit-learn nltk

# 2. Place your dataset as 'reviews.csv' with columns: 'text', 'sentiment' (positive/negative or 1/0)
#    If missing, the script generates a synthetic dataset automatically.

# 3. Run the script
python Project4-DecodeLabs.py
```

> 💡 **No NLTK?** Use the fallback script – pure Python tokenizer + stemmer.

---

## 📦 Outputs

| File | Description |
|------|-------------|
| `nlp_sentiment_results.png` | 4‑panel visualisation: Confusion Matrix, ROC Curve, Top Features, Confidence Distribution |
| `test_predictions.csv` | Predictions + probabilities on the test set |
| `sentiment_model.pkl` | Pickled model + TF‑IDF vectoriser for deployment |

---

## 📊 Sample Results (Synthetic Data)

```
Precision: 0.92  ← When we say Positive, we're right 92% of the time.
Recall:    0.88   ← We caught 88% of actual Positives.
F1-Score:  0.90   ← Harmonic mean of Precision & Recall.
ROC-AUC:   0.95   ← Model separates classes excellently.
```

---

## 👩‍💻 Author

**Javeria Faisal** – Data Science Intern @ DecodeLabs  
---

📌 **Project 4 – Optional Mastery Phase**  
*“You’ve earned your certificate – this is the cherry on top.”*
