# Behavioral Customer Segmentation and Repeat-Purchase Prediction

**Can pre-purchase browsing behavior identify distinct customer segments, and does it predict who buys again within 30 days?**

A team project for the University of Michigan MADS program (SIADS 685/696 milestone), analyzing ~20 million e-commerce events from an online cosmetics store (Oct 2019 – Feb 2020).

## Research questions

- **RQ1 — Segments:** Are there distinct behavioral segments among purchasers, based only on what they did *before* their first observed purchase?
- **RQ2 — Prediction:** Does pre-purchase behavior predict a repeat purchase within 30 days?
- **RQ3 — Added value:** Do the discovered segments add predictive value beyond the raw behavioral features themselves?

## Data

[eCommerce Events History in Cosmetics Shop](https://www.kaggle.com/datasets/mkechinov/ecommerce-events-history-in-cosmetics-shop) (REES46, Kaggle): event-level records (`view`, `cart`, `remove_from_cart`, `purchase`) with product, category, brand, price, and session fields. Raw data is **not** included in this repo — see [`data/README.md`](data/README.md) for how to obtain it and reproduce the analysis.

Analysis sample: **91,460 purchasers** with observable pre-purchase activity (13.7% repeat purchasers). Users are split **temporally** — earliest 80% by first-purchase date form the training set (73,168), the latest 20% the test set (18,292) — so evaluation mimics predicting future customers from past ones.

## Part A — Unsupervised learning (segmentation)

Five standardized behavioral-composition features: `view_share`, `cart_share`, `products_per_session`, `views_per_product`, `breadth` (`remove_share` dropped as a closed compositional component; `events_per_session` dropped, r ≈ 0.92 with `products_per_session`; skewed features log1p-transformed).

| Model (k = 2) | Train silhouette | Test silhouette |
| --- | --- | --- |
| **K-Means** | **0.319** | **0.325** |
| GMM (full covariance) | 0.303 | 0.314 |

Both methods peaked at k = 2; assignments were highly stable across seeds (mean pairwise ARI ≈ 0.999) and broadly agreed across models (cross-model ARI 0.701).

| Segment | Share | Profile | 30-day repeat rate |
| --- | --- | --- | --- |
| **Cart-oriented** | 43.5% | High cart share, more products per session, broader browsing | 14.9% |
| **View-oriented** | 56.5% | High view share, repeated viewing of the same products | 13.4% |

## Part B — Supervised learning (repeat-purchase prediction)

Logistic Regression, Gradient Boosting, and Random Forest compared across four feature sets (basic activity → + behavioral ratios → + cluster label → + engineered features), selected by 5-fold CV Average Precision (AP), and evaluated once on the temporal test set.

**Best model: tuned Gradient Boosting on the full 26-feature set**

| Metric | CV (train) | Temporal test |
| --- | --- | --- |
| Average Precision | 0.221 | 0.188 |
| ROC-AUC | — | 0.631 |
| Precision @ Top 10% | — | 0.233 (vs. 12.4% base rate) |

## Key findings

1. **Two stable, interpretable segments exist**, but they are general patterns, not sharply separated customer types (silhouette 0.318).
2. **Pre-purchase activity predicts repeat buying modestly.** How long and how often a user was active before purchasing (`pre_duration_days`, `pre_active_days`) mattered far more than any single action.
3. **Cluster labels add ~nothing for prediction** (GBT AP 0.2184 → 0.2192; cluster-label permutation importance ≈ 0; removing the cluster feature changed AP by ~0.0000). Segmentation is useful for interpretation and marketing hypotheses, not as a predictive feature — an honest negative result for RQ3.
4. **Performance decays over time** (CV AP 0.221 → test AP 0.188, alongside a repeat-rate drift from 14.1% to 12.4%), a reminder that behavioral models need periodic revalidation.

## Repository structure

```
├── Behavioral_Customer_Segmentation_and_Repeat_Purchase_Prediction.ipynb   # full pipeline: EDA → features → clustering → supervised models
├── docs/
│   └── Written_Report.pdf        # full written report
├── data/
│   └── README.md                 # data source and reproduction steps (data not included)
├── requirements.txt
└── README.md
```

## How to run

1. Download the dataset from Kaggle (link above) and place the monthly CSVs in `data/`.
2. Install dependencies: `pip install -r requirements.txt`
3. Open the notebook in Jupyter or Google Colab and run top to bottom. It builds the user-level analysis table, fits the segmentation, and trains/evaluates the prediction models on the temporal split.

## Team

- **Tingting Zhu** — data preparation, exploratory analysis, feature engineering
- **Jason Gan** — unsupervised learning, clustering evaluation, segment interpretation
- **Xiaojun Xun** — supervised learning, model evaluation, prediction analysis

*Coursework project. Results describe one store and one five-month window; segments and models should not be treated as general customer truths.*
