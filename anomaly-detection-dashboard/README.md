# Data Quality & Anomaly Detection — E-Commerce/Logistics Orders

An automated QA + anomaly-detection pipeline for order-level e-commerce/logistics
data, with an interactive Streamlit dashboard on top.

## What it does

1. **Data quality checks** (`src/quality_checks.py`) — completeness, uniqueness,
   validity, consistency, and timeliness checks over an orders table. Each check
   reports pass/fail, severity, and affected row count/indices.
2. **Anomaly detection** (`src/anomaly_detection.py`) — unsupervised, two methods
   cross-validated against each other:
   - Z-score and IQR (Tukey fence) per numeric column
   - Isolation Forest over scaled numeric features jointly (catches multivariate
     anomalies, e.g. plausible price but abnormal quantity × delivery-time combo)
3. **Dashboard** (`app.py`) — KPIs, a color-coded QA report, a filterable table
   of flagged orders with CSV export, and distribution plots.

## Dataset

Data is **synthetic** (`src/generate_data.py`), generated to match the schema of
public datasets like [Online Retail II](https://www.kaggle.com/datasets/mashlyn/online-retail-ii-uci)
or [DataCo Smart Supply Chain](https://www.kaggle.com/datasets/shashwatwork/dataco-smart-supply-chain-for-big-data-analysis).
Dirty data (nulls, duplicates, negative quantities/prices, extreme outliers,
inconsistent country naming, broken delivery-date logic) is injected on purpose
so the pipeline has something real to catch.

**To use a real dataset:** download one of the datasets above (or your own
logistics export), rename columns to match `src/generate_data.py`'s schema, and
replace the `load_data()` call in `app.py` with `pd.read_csv("your_file.csv")`.
No other code changes required.

## Run it

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Why unsupervised

Real fraud/anomaly labels are rare, delayed, or simply unavailable in
production. This pipeline never uses a label to build anything — only to
validate results if one happens to be available.
