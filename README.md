# 🚀 DecodeLabs Data Science Portfolio

## A Comprehensive Data Science Journey

Welcome to my data science portfolio! This repository showcases four end-to-end data science projects that demonstrate the complete lifecycle of data analysis, from raw data to actionable insights. Each project builds upon the previous one, creating a cohesive narrative of advanced analytics techniques.

---

## 📊 Project Overview

| Project | Focus | Key Techniques |
|---------|-------|---------------|
| **Project 1** | Data Cleaning & EDA | Missing value imputation, outlier detection, feature engineering |
| **Project 2** | Fraud Detection | SMOTE, Logistic Regression, Random Forest, Model Evaluation |
| **Project 3** | Customer Segmentation | K-Means Clustering, PCA, Silhouette Analysis, Persona Creation |
| **Project 4** | NLP & Sentiment Analysis | TF-IDF, Naive Bayes, Text Preprocessing, Sentiment Classification |

---

## 🛠️ Technologies Used

<p align="center">
  <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python"/>
  <img src="https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white" alt="Pandas"/>
  <img src="https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white" alt="NumPy"/>
  <img src="https://img.shields.io/badge/Scikit_Learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white" alt="Scikit-Learn"/>
  <img src="https://img.shields.io/badge/Matplotlib-11557C?style=for-the-badge&logo=matplotlib&logoColor=white" alt="Matplotlib"/>
  <img src="https://img.shields.io/badge/Seaborn-3776AB?style=for-the-badge&logo=seaborn&logoColor=white" alt="Seaborn"/>
</p>

---

## 📁 Project 1: Advanced EDA & Feature Engineering

### 🎯 Objective
Build a robust, production-ready dataset through comprehensive data cleaning, feature engineering, and statistical analysis.

### 🔑 Key Achievements

#### ✅ Data Quality Enhancement
- **Handled Missing Data**: Applied statistical imputation (Mean/Median) for numerical columns
- **Outlier Management**: Identified and capped outliers using IQR method (Winsorization)
- **Categorical Encoding**: One-hot encoded 6+ categorical features

#### ✅ Feature Engineering (8 New Features Created)
```python
# New features engineered
DayOfWeek          → Weekly patterns
IsWeekend          → Weekend behavior
OrderMonth         → Seasonal trends
IsDelivered        → Delivery status
HasDiscount        → Promotional impact
AvgItemValue       → Per-item value
OrderYear          → Annual trends
DiscountCode       → Discount categories
```

### 📊 Results
- **Final Dataset**: 1000+ rows, 50+ features
- **Memory Usage**: Optimized to ~2.3 MB
- **No Missing Values**: Complete dataset ready for modeling

### 📁 Output Files
- `processed_data_[timestamp].csv` - Ready-to-use dataset

---

## 🔍 Project 2: Fraud Detection Pipeline

### 🎯 Objective
Build a machine learning pipeline to detect fraudulent transactions, prioritizing recall over accuracy.

### 🔑 Key Achievements

#### ✅ Imbalanced Data Handling
- **Original Fraud Rate**: ~20% of transactions
- **Applied SMOTE**: Oversampled minority class to balance training data
- **Leak-Free Pipeline**: Used `imblearn.pipeline.Pipeline` to prevent data leakage

#### ✅ Model Training & Tuning

**Logistic Regression Pipeline:**
```
StandardScaler → SMOTE → LogisticRegression
```

**Random Forest Pipeline:**
```
SMOTE → RandomForestClassifier (no scaling needed)
```

**Hyperparameter Tuning:**
```python
# Grid Search Parameters
lr_params = {'C': [0.1, 1, 10], 'solver': ['liblinear']}
rf_params = {'n_estimators': [50, 100], 'max_depth': [10, None]}
```

### 📊 Model Performance

| Metric | Logistic Regression | Random Forest |
|--------|-------------------|---------------|
| **Precision** | 0.4234 | 0.8134 |
| **Recall** | 0.7112 | 0.8813 |
| **F1-Score** | 0.5306 | 0.8461 |
| **ROC-AUC** | 0.7842 | 0.9421 |

**Key Insight**: Random Forest outperforms Logistic Regression significantly, especially in identifying fraudulent transactions (higher recall).

### 📁 Output Files
- `fraud_detection_results.png` - Comprehensive visualization dashboard

---

## 👥 Project 3: Customer Segmentation (Unsupervised Learning)

### 🎯 Objective
Segment customers into distinct groups and create actionable business personas for targeted marketing.

### 🔑 Key Achievements

#### ✅ Customer Aggregation
Created customer-level features from transaction data:
```python
Recency          → Days since last purchase
Frequency        → Total orders
Monetary         → Total spend
AvgOrderValue    → Average transaction value
DiscountRate     → Proportion of discounted orders
DeliveryRate     → Successful delivery rate
UniqueProducts   → Product variety
```

#### ✅ Clustering Analysis
- **Optimal K**: Determined using Elbow Method and Silhouette Score
- **Best K**: 3 clusters (highest silhouette score)
- **Dimensionality Reduction**: PCA for visualization (95% variance explained)

### 🎭 Business Personas Identified

| Cluster | Persona | Size | Key Traits |
|---------|---------|------|------------|
| **0** | Average Shoppers | 63.9% | Balanced behavior, moderate spend |
| **1** | VIP Customers | 35.4% | High monetary value, frequent purchases |
| **2** | High-Value Loyal | 0.7% | Exceptional frequency, high product diversity |

### 📊 Cluster Centroids (Key Metrics)

| Metric | Cluster 0 | Cluster 1 | Cluster 2 |
|--------|-----------|-----------|-----------|
| Monetary | $545.82 | $1,981.98 | $1,891.99 |
| AvgOrderValue | $548.55 | $1,972.59 | $909.71 |
| Frequency | 1.0 | 1.0 | 2.09 |
| AvgQuantity | 2.34 | 4.04 | 3.04 |
| DiscountRate | 74.4% | 74.1% | 68.5% |

### 📁 Output Files
- `customer_segments_with_personas.csv` - Full customer segmentation
- `cluster_centroids_summary.csv` - Centroids in original units
- `cluster_feature_stats.csv` - Feature statistics by cluster

### 📊 Visualizations
1. **Elbow & Silhouette** - Optimal K determination
2. **2D Cluster Visualization** - PCA projection
3. **Feature Distributions** - Boxplots by cluster
4. **Silhouette Plot** - Cluster cohesion analysis
5. **Radar Chart** - Persona comparison
6. **Cluster Summary** - Quick reference table

---

## 💬 Project 4: NLP & Sentiment Analysis

### 🎯 Objective
Build a text classification pipeline to analyze customer sentiment from reviews.

### 🔑 Key Achievements

#### ✅ Text Preprocessing (Pure Python - No NLTK)
```python
def preprocess_text(text):
    # Lowercase
    text = text.lower()
    # Remove punctuation & numbers
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    # Tokenize
    tokens = text.split()
    # Remove stopwords (except negations)
    filtered = [t for t in tokens if t not in stop_words]
    # Basic stemming
    filtered = [simple_stem(t) for t in filtered]
    return " ".join(filtered)
```

**Negations Preserved**: `not`, `no`, `never`, `cannot`, `don't`, etc.

#### ✅ Feature Engineering
- **TF-IDF Vectorization**: 1,000 unigrams & bigrams
- **Max Features**: 1,000 most important terms
- **Sparse Matrix**: Efficient memory usage

### 📊 Model Performance

| Metric | Score |
|--------|-------|
| **Precision** | 0.9459 |
| **Recall** | 0.9211 |
| **F1-Score** | 0.9333 |
| **ROC-AUC** | 0.9840 |

### 📁 Output Files
- `nlp_sentiment_results.png` - Visualization dashboard
- `sentiment_model.pkl` - Serialized model & vectorizer
- `test_predictions.csv` - Sample predictions

---

## 📈 Key Insights Summary

### 1️⃣ Data Quality Matters
- Proper handling of missing values and outliers significantly improves model performance
- Feature engineering adds valuable business context

### 2️⃣ Imbalanced Data Challenges
- SMOTE effectively balances classes for fraud detection
- Random Forest outperforms Logistic Regression on imbalanced data
- **Recall is more important than accuracy** for fraud detection

### 3️⃣ Customer Segmentation Power
- 3 distinct customer segments with clear personas
- **VIP Customers** (35.4%) drive the most revenue
- **Average Shoppers** (63.9%) represent the largest segment with cross-sell potential

### 4️⃣ NLP Performance
- Simple preprocessing (no heavy libraries) achieves excellent results
- **TF-IDF with bigrams** captures important phrases
- **Negations** are critical for sentiment analysis

---

## 🚀 Next Steps & Recommendations

### For Fraud Detection:
- ✅ Deploy Random Forest model (better overall performance)
- ✅ Monitor model weekly for performance drift
- ✅ Retrain monthly with new transaction data
- ✅ Implement human-in-the-loop for high-value transactions

### For Customer Segmentation:
- ✅ Use personas for targeted marketing campaigns
- ✅ Develop VIP loyalty program
- ✅ Re-engagement campaigns for dormant customers
- ✅ Cross-sell recommendations based on product variety

### For Sentiment Analysis:
- ✅ Deploy model for real-time review classification
- ✅ Identify negative reviews for immediate action
- ✅ Track sentiment trends over time

---

## 📚 Project Structure

```
DecodeLabs-Portfolio/
├── Project1-EDA/
│   ├── Project1-DecodeLabs.py
│   └── processed_data_[timestamp].csv
├── Project2-Fraud/
│   ├── Project2-DecodeLabs.py
│   └── fraud_detection_results.png
├── Project3-Segmentation/
│   ├── Project3-DecodeLabs.py
│   ├── customer_segments_with_personas.csv
│   ├── cluster_centroids_summary.csv
│   ├── cluster_feature_stats.csv
│   ├── 1_elbow_silhouette.png
│   ├── 2_clusters_2d.png
│   ├── 3_feature_distributions.png
│   ├── 4_silhouette_plot.png
│   ├── 5_radar_chart.png
│   └── 6_cluster_summary.png
├── Project4-NLP/
│   ├── Project4-DecodeLabs.py
│   ├── nlp_sentiment_results.png
│   ├── sentiment_model.pkl
│   └── test_predictions.csv
└── README.md
```

---

## 🎯 Key Skills Demonstrated

| Skill | Project |
|-------|---------|
| **Data Cleaning** | Project 1 |
| **Feature Engineering** | Project 1, 2, 3, 4 |
| **Statistical Analysis** | Project 1 |
| **Supervised Learning** | Project 2, 4 |
| **Unsupervised Learning** | Project 3 |
| **Imbalanced Data Handling** | Project 2 |
| **NLP & Text Processing** | Project 4 |
| **Model Evaluation** | Project 2, 4 |
| **Visualization** | All Projects |
| **Business Insights** | All Projects |

---

## 📝 Final Thoughts

This portfolio demonstrates the complete data science workflow:
1. **Data Preparation** → Clean, engineer, and understand data
2. **Predictive Modeling** → Build and evaluate machine learning models
3. **Unsupervised Learning** → Discover hidden patterns and segments
4. **Natural Language Processing** → Extract insights from text data

**Accuracy was deliberately omitted** from all model evaluations in favor of metrics that better reflect real-world business impact:
- **Precision**: When we make a prediction, are we right?
- **Recall**: Did we catch all the important cases?
- **F1-Score**: Balance between precision and recall
- **ROC-AUC**: Model's ability to separate classes

---

**Powered by DecodeLabs** | **Data Science Intern - Javeria Faisal** | **Batch 2026**
