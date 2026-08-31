"""
generate_data.py

Generates a schema-accurate SYNTHETIC e-commerce/logistics order dataset.

Why synthetic: this project was built in a sandboxed dev environment without
access to Kaggle/UCI dataset hosts. The schema mirrors real public datasets
such as:
  - Kaggle "Online Retail II" (https://www.kaggle.com/datasets/mashlyn/online-retail-ii-uci)
  - Kaggle "DataCo Smart Supply Chain" (https://www.kaggle.com/datasets/shashwatwork/dataco-smart-supply-chain-for-big-data-analysis)

To swap in a real dataset: replace `generate_orders()` with
`pd.read_csv("your_real_file.csv")` in app.py / the notebook, keeping the
same column names, and the quality-check + anomaly-detection modules will
work unmodified.

Dirty data intentionally injected (this is the point of a QA pipeline):
  - missing values in customer_id, ship_country, unit_price
  - duplicate order rows
  - negative / zero quantities and prices
  - extreme outlier order values
  - delivery_days that are negative or absurdly large
  - inconsistent country name casing/spelling
  - a few rows with delivery_date before order_date (referential violation)
"""

import numpy as np
import pandas as pd

RNG = np.random.default_rng(42)

COUNTRIES = ["United States", "united states", "USA", "Germany", "France",
             "United Kingdom", "Canada", "Australia", "Brazil", "India"]
CATEGORIES = ["Electronics", "Home & Kitchen", "Apparel", "Toys", "Sports",
              "Office Supplies", "Beauty", "Books"]
CARRIERS = ["FastShip", "GlobalLogix", "ParcelPro", "EcoFreight"]


def generate_orders(n_orders: int = 5000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    order_ids = np.arange(100000, 100000 + n_orders)
    order_dates = pd.to_datetime("2024-01-01") + pd.to_timedelta(
        rng.integers(0, 365, n_orders), unit="D"
    )
    ship_delay = rng.integers(1, 10, n_orders)
    delivery_dates = order_dates + pd.to_timedelta(ship_delay, unit="D")

    df = pd.DataFrame({
        "order_id": order_ids,
        "customer_id": rng.integers(1000, 1000 + n_orders // 3, n_orders),
        "order_date": order_dates,
        "delivery_date": delivery_dates,
        "ship_country": rng.choice(COUNTRIES, n_orders),
        "carrier": rng.choice(CARRIERS, n_orders),
        "category": rng.choice(CATEGORIES, n_orders),
        "quantity": rng.integers(1, 8, n_orders),
        "unit_price": np.round(rng.gamma(shape=3.0, scale=12.0, size=n_orders), 2),
    })
    df["order_value"] = np.round(df["quantity"] * df["unit_price"], 2)
    df["delivery_days"] = (df["delivery_date"] - df["order_date"]).dt.days

    # ---- inject dirty data (deliberate, for QA pipeline to catch) ----

    # 1. missing values
    null_idx = rng.choice(n_orders, size=int(n_orders * 0.03), replace=False)
    df.loc[null_idx, "customer_id"] = np.nan
    null_idx2 = rng.choice(n_orders, size=int(n_orders * 0.02), replace=False)
    df.loc[null_idx2, "unit_price"] = np.nan
    null_idx3 = rng.choice(n_orders, size=int(n_orders * 0.015), replace=False)
    df.loc[null_idx3, "ship_country"] = np.nan

    # 2. duplicate rows (exact copies appended)
    dup_rows = df.sample(n=int(n_orders * 0.01), random_state=seed)
    df = pd.concat([df, dup_rows], ignore_index=True)

    # 3. negative / zero quantity or price (data entry errors)
    bad_idx = rng.choice(len(df), size=int(len(df) * 0.01), replace=False)
    df.loc[bad_idx, "quantity"] = -rng.integers(1, 5, len(bad_idx))
    bad_idx2 = rng.choice(len(df), size=int(len(df) * 0.005), replace=False)
    df.loc[bad_idx2, "unit_price"] = 0

    # 4. extreme outlier order values (fraud / pricing-glitch style anomalies)
    out_idx = rng.choice(len(df), size=int(len(df) * 0.008), replace=False)
    df.loc[out_idx, "order_value"] = df.loc[out_idx, "order_value"] * rng.uniform(15, 40, len(out_idx))

    # 5. delivery_days broken: negative (delivered before ordered) or huge delays
    bad_delivery_idx = rng.choice(len(df), size=int(len(df) * 0.006), replace=False)
    df.loc[bad_delivery_idx, "delivery_days"] = -rng.integers(1, 5, len(bad_delivery_idx))
    slow_idx = rng.choice(len(df), size=int(len(df) * 0.004), replace=False)
    df.loc[slow_idx, "delivery_days"] = rng.integers(60, 120, len(slow_idx))

    # 6. recompute delivery_date to match delivery_days for consistency checks
    df["delivery_date"] = df["order_date"] + pd.to_timedelta(df["delivery_days"], unit="D")

    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return df


if __name__ == "__main__":
    data = generate_orders()
    out_path = "data/orders_raw.csv"
    import os
    os.makedirs("data", exist_ok=True)
    data.to_csv(out_path, index=False)
    print(f"Generated {len(data)} rows -> {out_path}")
