---

# Project 3: Customer Segmentation

## 🎯 The Vibe
Unsupervised Learning to find hidden customer tribes. No labels, just pure data vibes. PCA + K-Means = Market Insights.

---

## 📊 What We Built

### The Stack
- **PCA** - Squeezed 12 features into 7 components (97% variance kept, we don't do noise)
- **K-Means** - Found the optimal K using Elbow + Silhouette Score
- **Business Personas** - Turned clusters into actual strategies

### The Flow
```
Raw Orders → Customer Aggregation → Standardization → PCA → K-Means → Personas
```

---

## 🔍 Key Results

| Cluster | Size | Persona | Vibe Check |
|---------|------|---------|------------|
| 0 | 63.9% | Average Shoppers | Casual buyers, standard behavior |
| 1 | 35.4% | VIP Customers | High spend, consistent purchases |
| 2 | 0.7% | Product Explorers | Diverse purchases, cross-sell goldmine |

**Optimal K = 3** (Silhouette Score: 0.2725)

---

## 📁 Output Files

### Visuals (6 PNGs)
```
1_elbow_silhouette.png      - Find the sweet spot K
2_clusters_2d.png           - See the tribes
3_feature_distributions.png - Who buys what
4_silhouette_plot.png       - Quality check
5_radar_chart.png           - Persona comparison
6_cluster_summary.png       - TL;DR table
```

### Data (3 CSVs)
```
customer_segments_with_personas.csv  - Every customer with their tribe
cluster_centroids_summary.csv        - What makes each tribe tick
cluster_feature_stats.csv            - Feature breakdown per cluster
```

---

## 🏷️ Personas & Strategy

### Cluster 0: Average Shoppers (64%)
> "The Mainstream"
- Balanced behavior, largest group
- **Move**: Test promotions, nudge to higher value

### Cluster 1: VIP Customers (35%)
> "The Heavy Hitters"  
- High frequency × High monetary
- **Move**: Loyalty program, exclusive perks

### Cluster 2: Product Explorers (0.7%)
> "The Curious Ones"
- Diverse purchases, small but mighty
- **Move**: Cross-sell, curated collections

---

## 🚀 Quick Start

```bash
# Clone the repo
git clone [your-repo-url]

# Install dependencies
pip install pandas numpy matplotlib seaborn scikit-learn

# Run the magic
python Project3-DecodeLabs.py
```

---

## 💡 Key Takeaways

- **Unsupervised Learning** = Finding patterns without labels
- **PCA** = Dimensionality reduction, keeping the signal
- **K-Means** = Grouping similar customers together
- **Business Personas** = Data → Actionable Strategy

---

## 🛠️ Tech Stack

```
Python 3.14
├── Pandas (data wrangling)
├── NumPy (math)
├── Scikit-learn (ML)
│   ├── StandardScaler
│   ├── PCA
│   └── KMeans
├── Matplotlib (visuals)
└── Seaborn (pretty visuals)
```

---

## 📈 Next Moves

1. Deploy segmentation in CRM
2. Personalize marketing campaigns by cluster
3. Monitor cluster drift monthly
4. Test DBSCAN for non-spherical clusters
5. A/B test persona-specific strategies

---

## 👩‍💻 Author

**Javeria Faisal**  
Data Science Intern @ DecodeLabs  
Batch 2026

---

> *"Data without strategy is just noise. Strategy without data is just guessing."*

---
