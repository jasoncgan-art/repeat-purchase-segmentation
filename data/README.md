# Data

The raw data is **not included** in this repository.

**Source:** [eCommerce Events History in Cosmetics Shop](https://www.kaggle.com/datasets/mkechinov/ecommerce-events-history-in-cosmetics-shop) (REES46, Kaggle) — event-level records (view, cart, remove_from_cart, purchase) from an online cosmetics store, October 2019 – February 2020.

To reproduce:

1. Download the monthly CSV files from Kaggle: `2019-Oct.csv`, `2019-Nov.csv`, `2019-Dec.csv`, `2020-Jan.csv`, `2020-Feb.csv`.
2. Place them in this `data/` folder.
3. Run the notebook top to bottom. Part 1 builds the user-level analysis table (`analysis_df.csv`); the clustering part adds segment labels (`analysis_df_with_cluster_labels.csv`).
