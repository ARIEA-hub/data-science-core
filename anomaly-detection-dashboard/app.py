"""
Streamlit dashboard: Data Quality & Anomaly Detection for E-Commerce/Logistics Orders

Run locally:
    pip install -r requirements.txt
    streamlit run app.py

Swap in a real dataset: edit `load_data()` below to `pd.read_csv("your_file.csv")`
with matching column names (see src/generate_data.py docstring for the schema
and recommended public datasets).
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from generate_data import generate_orders
from quality_checks import run_all_checks, results_to_frame
from anomaly_detection import combine_detections, NUMERIC_FEATURES

st.set_page_config(page_title="Order Data QA & Anomaly Dashboard", layout="wide")


@st.cache_data
def load_data(n_orders: int, seed: int) -> pd.DataFrame:
    return generate_orders(n_orders=n_orders, seed=seed)


@st.cache_data
def compute(df: pd.DataFrame, contamination: float):
    checks = run_all_checks(df)
    checks_df = results_to_frame(checks)
    enriched = combine_detections(df, contamination=contamination)
    return checks_df, enriched


# ---------------- Sidebar controls ----------------
st.sidebar.header("Data & Model Settings")
n_orders = st.sidebar.slider("Number of orders (synthetic)", 1000, 20000, 5000, step=1000)
seed = st.sidebar.number_input("Random seed", value=42, step=1)
contamination = st.sidebar.slider("Isolation Forest contamination", 0.01, 0.10, 0.02, step=0.01)
st.sidebar.caption(
    "Synthetic data mimics a real e-commerce/logistics orders table. "
    "Swap in a real CSV by editing `load_data()` in app.py."
)

df = load_data(n_orders, seed)
checks_df, enriched = compute(df, contamination)

# ---------------- Header / KPIs ----------------
st.title("📦 Order Data Quality & Anomaly Detection Dashboard")

failed_checks = (checks_df["status"] == "FAIL").sum()
critical_fails = ((checks_df["status"] == "FAIL") & (checks_df["severity"] == "critical")).sum()
n_anomalies = int(enriched["iso_anomaly"].sum())
n_both = int(enriched["flagged_by_both"].sum())

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Total Rows", f"{len(df):,}")
k2.metric("Checks Run", len(checks_df))
k3.metric("Checks Failed", int(failed_checks), delta=f"{critical_fails} critical", delta_color="inverse")
k4.metric("Anomalies (Isolation Forest)", f"{n_anomalies:,}")
k5.metric("Flagged by Both Methods", f"{n_both:,}")

st.divider()

# ---------------- Tabs ----------------
tab_qa, tab_anomaly, tab_explore = st.tabs(["🔍 Data Quality Report", "🚨 Anomaly Detection", "📊 Explore Data"])

with tab_qa:
    st.subheader("Automated QA Check Results")

    severity_color = {"critical": "🔴", "warning": "🟡", "info": "🔵"}
    checks_df_display = checks_df.copy()
    checks_df_display["severity"] = checks_df_display["severity"].map(lambda s: f"{severity_color.get(s,'')} {s}")

    def highlight_status(row):
        color = "background-color: #ffe5e5" if row["status"] == "FAIL" else "background-color: #e8f8ee"
        return [color] * len(row)

    st.dataframe(
        checks_df_display.style.apply(highlight_status, axis=1),
        use_container_width=True,
        hide_index=True,
    )

    st.caption(
        f"{failed_checks} of {len(checks_df)} checks failed ({critical_fails} critical). "
        "Critical failures should block downstream reporting until resolved; warnings are worth "
        "investigating but may reflect legitimate business exceptions."
    )

    csv = checks_df.to_csv(index=False).encode("utf-8")
    st.download_button("Download QA report (CSV)", csv, "qa_report.csv", "text/csv")

with tab_anomaly:
    st.subheader("Flagged Anomalous Orders")

    method = st.radio(
        "Detection method",
        ["Statistical (Z-score / IQR)", "Isolation Forest", "Flagged by both (high confidence)"],
        horizontal=True,
    )

    if method == "Statistical (Z-score / IQR)":
        flagged = enriched[enriched["stat_anomaly_score"] > 0]
    elif method == "Isolation Forest":
        flagged = enriched[enriched["iso_anomaly"]]
    else:
        flagged = enriched[enriched["flagged_by_both"]]

    st.write(f"**{len(flagged):,}** orders flagged by this method (of {len(enriched):,} total).")
    display_cols = ["order_id", "customer_id", "order_date", "ship_country",
                     "category", "quantity", "unit_price", "order_value",
                     "delivery_days", "stat_anomaly_score", "iso_anomaly", "iso_score"]
    display_cols = [c for c in display_cols if c in flagged.columns]
    st.dataframe(flagged[display_cols].sort_values("order_value", ascending=False),
                 use_container_width=True, hide_index=True)

    csv2 = flagged.to_csv(index=False).encode("utf-8")
    st.download_button("Download flagged orders (CSV)", csv2, "flagged_orders.csv", "text/csv")

    st.subheader("Order value distribution: normal vs. flagged")
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(enriched.loc[~enriched.index.isin(flagged.index), "order_value"].dropna(),
            bins=50, alpha=0.6, label="normal")
    ax.hist(flagged["order_value"].dropna(), bins=50, alpha=0.8, label="flagged", color="crimson")
    ax.set_xlabel("Order Value")
    ax.set_ylabel("Count")
    ax.legend()
    ax.set_xlim(0, enriched["order_value"].quantile(0.99))
    st.pyplot(fig)

with tab_explore:
    st.subheader("Raw / enriched data explorer")
    st.dataframe(df.head(500), use_container_width=True)

    st.subheader("Numeric feature summary")
    st.dataframe(df[NUMERIC_FEATURES].describe(), use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        fig2, ax2 = plt.subplots()
        df["ship_country"].value_counts().plot(kind="bar", ax=ax2)
        ax2.set_title("Orders by ship_country (note: casing inconsistencies are intentional)")
        st.pyplot(fig2)
    with col2:
        fig3, ax3 = plt.subplots()
        df["delivery_days"].plot(kind="hist", bins=40, ax=ax3)
        ax3.set_title("Delivery days distribution")
        st.pyplot(fig3)
