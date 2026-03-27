# 🧵 Vastraa — Data-Driven Textile Intelligence Dashboard

**Vastraa** is a D2C (Direct-to-Consumer) brand bringing India's rich regional handloom and artisan textile heritage to modern consumers. This Streamlit dashboard provides end-to-end data analytics for data-driven decision making.

## 📊 Dashboard Pages

| Page | Analytics Type | Purpose |
|------|---------------|---------|
| 1. Executive Summary | Descriptive | KPIs, demographics, product interest overview |
| 2. EDA Deep Dive | Descriptive + Diagnostic | Correlations, bivariate analysis, textile & behavioral patterns |
| 3. Customer Segmentation | Prescriptive (Clustering) | K-Means & Hierarchical clustering with radar profiles |
| 4. Purchase Prediction | Predictive (Classification) | 7 algorithms compared — LR, DT, RF, KNN, SVM, NB, GB |
| 5. Spend Prediction | Predictive (Regression) | Linear, Ridge & Lasso regression on 3 spend targets |
| 6. Market Basket Analysis | Prescriptive (Association Rules) | Apriori algorithm for bundle & cross-sell strategy |
| 7. Prescriptive Strategy | Prescriptive | Actionable recommendations per customer segment |
| 8. New Customer Prediction | Predictive | Single form + bulk CSV upload for scoring new leads |

## 🚀 Deployment

### Run Locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

### Deploy on Streamlit Cloud
1. Push this repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repo
4. Set main file as `app.py`
5. Deploy!

## 📁 Files

- `app.py` — Main Streamlit application (all 8 pages)
- `vastraa_cleaned_dataset.csv` — Cleaned synthetic dataset (2,000 respondents, 98 columns)
- `requirements.txt` — Python dependencies

## 🔬 Algorithms Used

- **Classification:** Logistic Regression, Decision Tree, Random Forest, KNN, SVM, Naive Bayes, Gradient Boosting
- **Clustering:** K-Means (with Elbow + Silhouette), Hierarchical (Ward linkage with Dendrogram)
- **Association Rules:** Apriori with support, confidence, and lift metrics
- **Regression:** Linear, Ridge (L2), Lasso (L1)

## 📋 Dataset

- 2,000 synthetic survey respondents (Pan-India)
- 30 survey questions across 8 sections
- 98 columns (binary multi-select + ordinal codes + derived features)
- Realistic correlations, persona-driven patterns, ~6% noise/outliers

---

*Built by Samvit | Data Analytics — MGB | Project Based Learning*
