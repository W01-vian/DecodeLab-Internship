"""
PROJECT 4: NLP & SENTIMENT ANALYSIS (Fallback – No NLTK required)
Batch: 2026 | Powered by DecodeLabs
Data Science Intern – Javeria Faisal
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import string
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.naive_bayes import MultinomialNB, ComplementNB
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
)
import warnings
import pickle

warnings.filterwarnings("ignore")

print("=" * 80)
print("PROJECT 4: NLP & SENTIMENT ANALYSIS")
print("=" * 80)

# ----------------------------------------------------------------------
# 1. DATA LOADING
# ----------------------------------------------------------------------
print("\nSTEP 1: DATA LOADING")
print("-" * 40)

try:
    df = pd.read_csv("reviews.csv")
    print(f"Loaded {df.shape[0]} reviews from 'reviews.csv'")
    if df["sentiment"].dtype == object:
        df["sentiment"] = df["sentiment"].map({"positive": 1, "negative": 0})
    print(f"Class distribution: Positive = {df['sentiment'].sum()}, "
          f"Negative = {len(df) - df['sentiment'].sum()}")
except FileNotFoundError:
    print("No 'reviews.csv' found. Generating a synthetic dataset for demonstration.")
    np.random.seed(42)
    n_samples = 200
    positive_templates = [
        "This product is absolutely amazing and works perfectly.",
        "I love this item, it exceeds my expectations.",
        "Great quality, highly recommend to everyone.",
        "Excellent service and fast delivery.",
        "Very satisfied with my purchase.",
    ]
    negative_templates = [
        "This product is terrible, does not work at all.",
        "I hate this item, very poor quality.",
        "Waste of money, do not buy this.",
        "Disappointed with the service and delivery.",
        "Not worth the price, poor build.",
    ]
    texts, labels = [], []
    for _ in range(n_samples // 2):
        texts.append(np.random.choice(positive_templates))
        labels.append(1)
        texts.append(np.random.choice(negative_templates))
        labels.append(0)
    for i in range(20):
        texts.append("Not bad, but could be better." if i % 2 == 0 else "Not good at all.")
        labels.append(1 if i % 2 == 0 else 0)
    df = pd.DataFrame({"text": texts, "sentiment": labels})
    print(f"Generated {len(df)} synthetic reviews (Positive = {df['sentiment'].sum()}, "
          f"Negative = {len(df) - df['sentiment'].sum()})")

# ----------------------------------------------------------------------
# 2. TEXT PREPROCESSING (Pure Python – no NLTK)
# ----------------------------------------------------------------------
print("\nSTEP 2: TEXT PREPROCESSING (Pure Python)")
print("-" * 40)

# Negations to keep
negations = {
    "not", "no", "never", "nor", "neither", "none", "nobody",
    "nothing", "nowhere", "cannot", "don't", "doesn't", "didn't",
    "won't", "wouldn't", "shouldn't", "couldn't"
}

# Common stopwords (from NLTK) – we exclude negations
default_stopwords = {
    "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours",
    "yourself", "he", "him", "his", "himself", "she", "her", "hers", "herself", "it",
    "its", "itself", "they", "them", "their", "theirs", "themselves", "am", "is", "are",
    "was", "were", "be", "been", "being", "have", "has", "had", "having", "do", "does",
    "did", "doing", "will", "would", "shall", "should", "may", "might", "must", "can",
    "could", "a", "an", "the", "and", "but", "or", "for", "of", "with", "without",
    "so", "than", "then", "hence", "thus", "etc", "etc.", "i.e.", "e.g.", "vs", "vs."
}
stop_words = default_stopwords - negations
print(f"Stopwords after removing negations: {len(stop_words)}")
print(f"Negations kept: {negations}")

def simple_stem(word):
    """A very basic stemmer (suffix removal) to mimic lemmatisation."""
    if len(word) > 3:
        if word.endswith("ing"):
            return word[:-3]
        if word.endswith("ed"):
            return word[:-2]
        if word.endswith("es"):
            return word[:-2]
        if word.endswith("s") and not word.endswith("ss"):
            return word[:-1]
    return word

def preprocess_text(text):
    """Lowercase, remove punctuation/numbers, tokenize, remove stopwords (except negations), apply stemming."""
    text = text.lower()
    # Remove punctuation and digits
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    tokens = text.split()
    filtered = []
    for token in tokens:
        if token in stop_words:
            continue
        stemmed = simple_stem(token)
        filtered.append(stemmed)
    return " ".join(filtered)

print("Applying preprocessing to all reviews...")
df["processed_text"] = df["text"].apply(preprocess_text)
print("Preprocessing complete.\n")
print("Sample original text:")
print(df["text"].iloc[0])
print("Processed text:")
print(df["processed_text"].iloc[0])

# ----------------------------------------------------------------------
# 3. TRAIN-TEST SPLIT
# ----------------------------------------------------------------------
print("\nSTEP 3: TRAIN-TEST SPLIT")
print("-" * 40)

X = df["processed_text"]
y = df["sentiment"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Training samples: {len(X_train)} (Pos: {y_train.sum()}, Neg: {len(y_train) - y_train.sum()})")
print(f"Test samples:     {len(X_test)} (Pos: {y_test.sum()}, Neg: {len(y_test) - y_test.sum()})")

# ----------------------------------------------------------------------
# 4. TF-IDF VECTORISATION
# ----------------------------------------------------------------------
print("\nSTEP 4: TF-IDF VECTORISATION")
print("-" * 40)

max_features = 5000
vectorizer = TfidfVectorizer(
    ngram_range=(1, 2),
    max_features=max_features,
    stop_words=None,          # we already removed stopwords manually
    sublinear_tf=True,
    use_idf=True,
    smooth_idf=True,
)

print(f"Vectorising with max_features={max_features}, ngram_range=(1,2)")
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

print(f"Training TF‑IDF shape: {X_train_vec.shape} (sparse CSR)")
print(f"Test TF‑IDF shape:     {X_test_vec.shape}")
print(f"Memory usage: {X_train_vec.data.nbytes / 1024**2:.2f} MB")

# ----------------------------------------------------------------------
# 5. MODEL TRAINING
# ----------------------------------------------------------------------
print("\nSTEP 5: MODEL TRAINING")
print("-" * 40)

pos_ratio = y_train.mean()
neg_ratio = 1 - pos_ratio
print(f"Training class balance: Positive = {pos_ratio:.2%}, Negative = {neg_ratio:.2%}")

use_complement = (pos_ratio > 0.6) or (neg_ratio > 0.6)
print(f"Using {'ComplementNB' if use_complement else 'MultinomialNB'} "
      f"(imbalance detected: {use_complement})")

model = ComplementNB(alpha=1.0) if use_complement else MultinomialNB(alpha=1.0)
model.fit(X_train_vec, y_train)
print("Model training complete.")

# ----------------------------------------------------------------------
# 6. EVALUATION
# ----------------------------------------------------------------------
print("\nSTEP 6: MODEL EVALUATION")
print("-" * 40)

y_pred = model.predict(X_test_vec)
y_proba = model.predict_proba(X_test_vec)[:, 1]

precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_proba)

print(f"Precision: {precision:.4f}  ← When we predict Positive, are we right?")
print(f"Recall:    {recall:.4f}     ← Did we catch all true Positives?")
print(f"F1-Score:  {f1:.4f}        ← Harmonic mean of Precision & Recall")
print(f"ROC-AUC:   {roc_auc:.4f}    ← Model's ability to separate classes")

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=["Negative", "Positive"]))

cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(f"  True Negatives:  {cm[0,0]}")
print(f"  False Positives: {cm[0,1]}  ← Type I Error")
print(f"  False Negatives: {cm[1,0]}  ← Type II Error (Costly!)")
print(f"  True Positives:  {cm[1,1]}")

cv_scores = cross_val_score(model, X_train_vec, y_train, cv=5, scoring="f1")
print(f"\n5‑Fold Cross‑Validation F1 Scores: {cv_scores}")
print(f"Mean F1: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")

# ----------------------------------------------------------------------
# 7. VISUALISATIONS
# ----------------------------------------------------------------------
print("\nSTEP 7: VISUALISATIONS")
print("-" * 40)

fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# (1) Confusion Matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Negative", "Positive"])
disp.plot(ax=axes[0, 0], cmap="Blues", values_format="d")
axes[0, 0].set_title(f"Confusion Matrix\nF1: {f1:.3f}")

# (2) ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_proba)
axes[0, 1].plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC (AUC = {roc_auc:.3f})")
axes[0, 1].plot([0, 1], [0, 1], "k--", lw=1, label="Random Classifier")
axes[0, 1].set_xlabel("False Positive Rate")
axes[0, 1].set_ylabel("True Positive Rate")
axes[0, 1].set_title("ROC Curve – Sentiment Analysis")
axes[0, 1].legend(loc="lower right")
axes[0, 1].grid(True, alpha=0.3)

# (3) Top Features per Class
feature_names = vectorizer.get_feature_names_out()
if hasattr(model, "feature_log_prob_"):
    log_probs = model.feature_log_prob_
    diff = log_probs[1] - log_probs[0]
    top_pos_idx = np.argsort(diff)[-10:][::-1]
    top_neg_idx = np.argsort(diff)[:10]
    top_pos_terms = [feature_names[i] for i in top_pos_idx]
    top_neg_terms = [feature_names[i] for i in top_neg_idx]

    axes[1, 0].barh(range(10), diff[top_pos_idx][::-1], color="green", alpha=0.7)
    axes[1, 0].set_yticks(range(10))
    axes[1, 0].set_yticklabels(top_pos_terms[::-1])
    axes[1, 0].set_title("Top 10 Words Indicative of Positive Sentiment")
    axes[1, 0].grid(True, alpha=0.3)
else:
    axes[1, 0].text(0.5, 0.5, "Feature importance not available\nfor this model",
                    ha="center", va="center")
    axes[1, 0].set_title("Feature Importance")

# (4) Confidence Distribution
axes[1, 1].hist(y_proba[y_test == 0], bins=20, alpha=0.7, label="Actual Negative", color="red")
axes[1, 1].hist(y_proba[y_test == 1], bins=20, alpha=0.7, label="Actual Positive", color="green")
axes[1, 1].set_xlabel("Predicted Probability of Positive")
axes[1, 1].set_ylabel("Frequency")
axes[1, 1].set_title("Prediction Confidence Distribution")
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("nlp_sentiment_results.png", dpi=300, bbox_inches="tight")
print("Saved visualisation: nlp_sentiment_results.png")
plt.close()

# ----------------------------------------------------------------------
# 8. SAVE OUTPUTS
# ----------------------------------------------------------------------
print("\nSTEP 8: SAVING OUTPUTS")
print("-" * 40)

df_test = pd.DataFrame({
    "text": X_test,
    "true_sentiment": y_test,
    "predicted_sentiment": y_pred,
    "probability_positive": y_proba,
})
df_test.to_csv("test_predictions.csv", index=False)
print("Saved: test_predictions.csv")

with open("sentiment_model.pkl", "wb") as f:
    pickle.dump((vectorizer, model), f)
print("Saved: sentiment_model.pkl")

# ----------------------------------------------------------------------
# 9. FINAL SUMMARY
# ----------------------------------------------------------------------
print("\n" + "=" * 80)
print("PROJECT 4 COMPLETED SUCCESSFULLY (Fallback version)")
print("=" * 80)
print("""
KEY ACHIEVEMENTS:
 1. Built a complete NLP pipeline from raw text to predictive model.
 2. Preprocessed text with stopword removal (excluded negations) and basic stemming.
 3. Vectorised using TF‑IDF with unigrams + bigrams, stored as sparse CSR matrix.
 4. Trained Multinomial/Complement Naive Bayes with Laplace smoothing.
 5. Evaluated using Precision, Recall, F1, ROC‑AUC (accuracy deliberately omitted).
 6. Generated visualisations: confusion matrix, ROC curve, top features, confidence distribution.
""")
print("=" * 80)