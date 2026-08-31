"""
quality_checks.py

Automated data-quality checks for the orders dataset. Each check returns a
structured result so the Streamlit dashboard (and any future CI job) can
render or gate on them without re-implementing logic.

Categories covered:
  - Completeness   : missing value rates per column
  - Uniqueness      : duplicate rows / duplicate order_ids
  - Validity        : type & range checks (quantity > 0, price >= 0, etc.)
  - Consistency     : cross-field logic (delivery_date >= order_date, etc.)
  - Timeliness      : delivery_days within a sane operational range
"""

from dataclasses import dataclass, field
import pandas as pd
import numpy as np


@dataclass
class CheckResult:
    name: str
    category: str
    passed: bool
    severity: str  # "info" | "warning" | "critical"
    details: str
    affected_rows: int = 0
    row_indices: list = field(default_factory=list)


REQUIRED_COLUMNS = [
    "order_id", "customer_id", "order_date", "delivery_date",
    "ship_country", "carrier", "category", "quantity",
    "unit_price", "order_value", "delivery_days",
]


def run_all_checks(df: pd.DataFrame) -> list[CheckResult]:
    results: list[CheckResult] = []

    # ---- Schema ----
    missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    results.append(CheckResult(
        name="Required columns present",
        category="Schema",
        passed=len(missing_cols) == 0,
        severity="critical",
        details=f"Missing columns: {missing_cols}" if missing_cols else "All required columns present.",
    ))

    # ---- Completeness ----
    for col in ["customer_id", "ship_country", "unit_price"]:
        if col not in df.columns:
            continue
        null_mask = df[col].isna()
        null_pct = null_mask.mean() * 100
        results.append(CheckResult(
            name=f"Missing values: {col}",
            category="Completeness",
            passed=null_pct < 1.0,
            severity="warning" if null_pct < 5 else "critical",
            details=f"{null_pct:.2f}% missing ({null_mask.sum()} rows).",
            affected_rows=int(null_mask.sum()),
            row_indices=df.index[null_mask].tolist(),
        ))

    # ---- Uniqueness ----
    dup_mask = df.duplicated(keep="first")
    results.append(CheckResult(
        name="Exact duplicate rows",
        category="Uniqueness",
        passed=dup_mask.sum() == 0,
        severity="warning",
        details=f"{dup_mask.sum()} exact duplicate rows found.",
        affected_rows=int(dup_mask.sum()),
        row_indices=df.index[dup_mask].tolist(),
    ))

    dup_id_mask = df["order_id"].duplicated(keep=False) if "order_id" in df.columns else pd.Series(dtype=bool)
    results.append(CheckResult(
        name="Duplicate order_id",
        category="Uniqueness",
        passed=dup_id_mask.sum() == 0,
        severity="critical",
        details=f"{dup_id_mask.sum()} rows share a duplicated order_id.",
        affected_rows=int(dup_id_mask.sum()),
        row_indices=df.index[dup_id_mask].tolist() if len(dup_id_mask) else [],
    ))

    # ---- Validity ----
    if "quantity" in df.columns:
        bad_qty = df["quantity"] <= 0
        results.append(CheckResult(
            name="Non-positive quantity",
            category="Validity",
            passed=bad_qty.sum() == 0,
            severity="critical",
            details=f"{bad_qty.sum()} rows with quantity <= 0.",
            affected_rows=int(bad_qty.sum()),
            row_indices=df.index[bad_qty].tolist(),
        ))

    if "unit_price" in df.columns:
        bad_price = df["unit_price"] <= 0
        results.append(CheckResult(
            name="Non-positive unit price",
            category="Validity",
            passed=bad_price.sum() == 0,
            severity="critical",
            details=f"{bad_price.sum()} rows with unit_price <= 0.",
            affected_rows=int(bad_price.sum()),
            row_indices=df.index[bad_price.fillna(False)].tolist(),
        ))

    if "ship_country" in df.columns:
        # inconsistent casing/spelling, e.g. "USA" vs "United States" vs "united states"
        normalized = df["ship_country"].dropna().str.strip().str.lower()
        variants = normalized.value_counts()
        us_variants = [v for v in variants.index if v in ("usa", "united states", "us")]
        inconsistent = len(us_variants) > 1
        results.append(CheckResult(
            name="Inconsistent country naming",
            category="Validity",
            passed=not inconsistent,
            severity="warning",
            details=f"Found variant spellings for the same country: {us_variants}" if inconsistent else "No obvious naming inconsistencies.",
        ))

    # ---- Consistency ----
    if {"order_date", "delivery_date"}.issubset(df.columns):
        bad_order = pd.to_datetime(df["delivery_date"]) < pd.to_datetime(df["order_date"])
        results.append(CheckResult(
            name="Delivery date before order date",
            category="Consistency",
            passed=bad_order.sum() == 0,
            severity="critical",
            details=f"{bad_order.sum()} rows delivered before they were ordered.",
            affected_rows=int(bad_order.sum()),
            row_indices=df.index[bad_order].tolist(),
        ))

    if {"quantity", "unit_price", "order_value"}.issubset(df.columns):
        expected = (df["quantity"] * df["unit_price"]).round(2)
        mismatch = (expected - df["order_value"]).abs() > 0.5
        mismatch = mismatch.fillna(False)
        results.append(CheckResult(
            name="order_value != quantity * unit_price",
            category="Consistency",
            passed=mismatch.sum() == 0,
            severity="warning",
            details=f"{mismatch.sum()} rows where order_value doesn't reconcile with quantity*price (may indicate discounting, fraud, or entry errors).",
            affected_rows=int(mismatch.sum()),
            row_indices=df.index[mismatch].tolist(),
        ))

    # ---- Timeliness ----
    if "delivery_days" in df.columns:
        bad_delivery = (df["delivery_days"] < 0) | (df["delivery_days"] > 30)
        results.append(CheckResult(
            name="Delivery time out of operational range (0-30 days)",
            category="Timeliness",
            passed=bad_delivery.sum() == 0,
            severity="warning",
            details=f"{bad_delivery.sum()} rows with delivery_days outside 0-30 days.",
            affected_rows=int(bad_delivery.sum()),
            row_indices=df.index[bad_delivery].tolist(),
        ))

    return results


def results_to_frame(results: list[CheckResult]) -> pd.DataFrame:
    return pd.DataFrame([{
        "check": r.name,
        "category": r.category,
        "status": "PASS" if r.passed else "FAIL",
        "severity": r.severity,
        "affected_rows": r.affected_rows,
        "details": r.details,
    } for r in results])
