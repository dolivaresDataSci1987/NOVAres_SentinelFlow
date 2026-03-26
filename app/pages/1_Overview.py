from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.data.load_dashboard_data import (
    extract_kpi_value,
    infer_main_transaction_date_column,
    list_expected_files,
    load_selected_datasets,
    parse_datetime_column,
)


st.set_page_config(
    page_title="Overview | SentinelFlow",
    page_icon="📊",
    layout="wide",
)


@st.cache_data
def get_data():
    return load_selected_datasets(
        [
            "dashboard_scored_transactions",
            "executive_kpis",
            "alert_queue",
            "review_queue_top_1pct",
            "review_queue_top_3pct",
            "review_queue_top_5pct",
            "review_queue_top_10pct",
            "queue_summary",
            "output_inventory",
        ]
    )


def format_metric_value(value, fmt: str = ".4f", pct: bool = False) -> str:
    if value is None or pd.isna(value):
        return "N/A"
    if pct:
        return f"{value:.2%}"
    return format(float(value), fmt)


data = get_data()

scored_df = data.get("dashboard_scored_transactions", pd.DataFrame())
executive_kpis_df = data.get("executive_kpis", pd.DataFrame())
alert_queue_df = data.get("alert_queue", pd.DataFrame())
q1_df = data.get("review_queue_top_1pct", pd.DataFrame())
q3_df = data.get("review_queue_top_3pct", pd.DataFrame())
q5_df = data.get("review_queue_top_5pct", pd.DataFrame())
q10_df = data.get("review_queue_top_10pct", pd.DataFrame())
queue_summary_df = data.get("queue_summary", pd.DataFrame())
output_inventory_df = data.get("output_inventory", pd.DataFrame())

inventory_df = list_expected_files()

st.title("📊 Overview")
st.caption("Executive entry point for the SentinelFlow fraud monitoring dashboard")

# =========================================================
# KPI BLOCK
# =========================================================

total_transactions = len(scored_df) if not scored_df.empty else None
total_alerts = len(alert_queue_df) if not alert_queue_df.empty else None
q1_n = len(q1_df) if not q1_df.empty else None
q3_n = len(q3_df) if not q3_df.empty else None
q5_n = len(q5_df) if not q5_df.empty else None
q10_n = len(q10_df) if not q10_df.empty else None

champion_threshold = extract_kpi_value(
    executive_kpis_df,
    candidate_keys=["champion_threshold", "selected_threshold", "operating_threshold", "model_threshold"],
)

roc_auc = extract_kpi_value(
    executive_kpis_df,
    candidate_keys=["roc_auc", "test_roc_auc", "champion_test_roc_auc"],
)

pr_auc = extract_kpi_value(
    executive_kpis_df,
    candidate_keys=["pr_auc", "test_pr_auc", "champion_test_pr_auc"],
)

precision = extract_kpi_value(
    executive_kpis_df,
    candidate_keys=["precision", "test_precision", "champion_test_precision"],
)

recall = extract_kpi_value(
    executive_kpis_df,
    candidate_keys=["recall", "test_recall", "champion_test_recall"],
)

top3_recall = extract_kpi_value(
    executive_kpis_df,
    candidate_keys=["top_3pct_recall", "top3_recall", "recall_top_3pct"],
)

# Fallback manual con tus valores conocidos
if champion_threshold is None:
    champion_threshold = 0.8987
if roc_auc is None:
    roc_auc = 0.9781
if pr_auc is None:
    pr_auc = 0.6004
if precision is None:
    precision = 0.5985
if recall is None:
    recall = 0.4969
if top3_recall is None:
    top3_recall = 0.7377

k1, k2, k3, k4 = st.columns(4)
with k1:
    st.metric("Scored transactions", f"{total_transactions:,}" if total_transactions is not None else "N/A")
with k2:
    st.metric("Alert queue", f"{total_alerts:,}" if total_alerts is not None else "N/A")
with k3:
    st.metric("Champion threshold", format_metric_value(champion_threshold))
with k4:
    st.metric("Test ROC-AUC", format_metric_value(roc_auc))

k5, k6, k7, k8 = st.columns(4)
with k5:
    st.metric("Test PR-AUC", format_metric_value(pr_auc))
with k6:
    st.metric("Precision", format_metric_value(precision))
with k7:
    st.metric("Recall", format_metric_value(recall))
with k8:
    st.metric("Top 3% recall", format_metric_value(top3_recall, pct=True))

st.markdown("---")

# =========================================================
# REVIEW CAPACITY
# =========================================================

st.markdown("## Review capacity snapshot")

c1, c2, c3, c4 = st.columns(4)
with c1:
    st.metric("Top 1% queue", f"{q1_n:,}" if q1_n is not None else "N/A")
with c2:
    st.metric("Top 3% queue", f"{q3_n:,}" if q3_n is not None else "N/A")
with c3:
    st.metric("Top 5% queue", f"{q5_n:,}" if q5_n is not None else "N/A")
with c4:
    st.metric("Top 10% queue", f"{q10_n:,}" if q10_n is not None else "N/A")

st.markdown("---")

# =========================================================
# EXECUTIVE READING
# =========================================================

st.markdown("## Executive reading")

summary_lines = []

if total_transactions is not None:
    summary_lines.append(f"- Total scored population: **{total_transactions:,} transactions**.")

if total_alerts is not None and total_transactions not in [None, 0]:
    summary_lines.append(
        f"- Current alert queue contains **{total_alerts:,} alerts** "
        f"({total_alerts / total_transactions:.2%} of scored transactions)."
    )

summary_lines.append(f"- Champion threshold is currently set at **{champion_threshold:.4f}**.")
summary_lines.append(
    f"- Champion test performance: **ROC-AUC {roc_auc:.4f}**, **PR-AUC {pr_auc:.4f}**, "
    f"**Precision {precision:.4f}**, **Recall {recall:.4f}**."
)
summary_lines.append(
    f"- Review prioritization remains strong, with **Top 3% recall = {top3_recall:.2%}**."
)

for line in summary_lines:
    st.markdown(line)

st.markdown("---")

# =========================================================
# DATASET PREVIEW
# =========================================================

st.markdown("## Core dataset preview")

if not scored_df.empty:
    preview_df = scored_df.copy()
    date_col = infer_main_transaction_date_column(preview_df)
    if date_col is not None:
        preview_df = parse_datetime_column(preview_df, date_col)

    preview_cols = [
        col for col in [
            "transaction_id",
            "transaction_ts",
            "transaction_date",
            "fraud_score",
            "risk_bucket",
            "alert_flag",
            "prediction_outcome",
            "review_rank",
            "amount",
            "channel",
            "payment_type",
            "merchant_category",
            "customer_segment",
            "transaction_country",
            "ip_country",
        ]
        if col in preview_df.columns
    ]

    if preview_cols:
        st.dataframe(
            preview_df.loc[:, preview_cols].head(25),
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.dataframe(preview_df.head(25), use_container_width=True, hide_index=True)
else:
    st.warning("dashboard_scored_transactions.csv is missing or empty.")
    st.markdown("### File inventory diagnostic")
    st.dataframe(inventory_df, use_container_width=True, hide_index=True)

st.markdown("---")

# =========================================================
# TEMPORARY DIAGNOSTICS
# =========================================================

with st.expander("Temporary diagnostics · Overview dataset loader", expanded=False):
    st.markdown("### dashboard_scored_transactions diagnostics")

    if not scored_df.empty:
        d1, d2, d3 = st.columns(3)
        with d1:
            st.metric("Rows", f"{scored_df.shape[0]:,}")
        with d2:
            st.metric("Columns", f"{scored_df.shape[1]:,}")
        with d3:
            detected_date_col = infer_main_transaction_date_column(scored_df)
            st.metric("Detected date column", detected_date_col if detected_date_col else "None")

        st.markdown("**Detected columns**")
        st.write(list(scored_df.columns))

        st.markdown("**First 5 rows**")
        st.dataframe(scored_df.head(5), use_container_width=True, hide_index=True)
    else:
        st.error("dashboard_scored_transactions loaded as empty DataFrame.")
        st.dataframe(inventory_df, use_container_width=True, hide_index=True)

    st.markdown("### Other dataset status")
    diagnostic_rows = []
    for dataset_name, df in data.items():
        diagnostic_rows.append(
            {
                "dataset": dataset_name,
                "rows": int(df.shape[0]) if isinstance(df, pd.DataFrame) else None,
                "columns": int(df.shape[1]) if isinstance(df, pd.DataFrame) else None,
                "empty": bool(df.empty) if isinstance(df, pd.DataFrame) else True,
            }
        )
    st.dataframe(pd.DataFrame(diagnostic_rows), use_container_width=True, hide_index=True)

st.markdown("---")

# =========================================================
# OUTPUT INVENTORY
# =========================================================

st.markdown("## Output inventory")

if output_inventory_df is not None and not output_inventory_df.empty:
    st.dataframe(output_inventory_df, use_container_width=True, hide_index=True)
else:
    st.info("output_inventory.csv is missing or empty. Falling back to filesystem inventory.")
    st.dataframe(inventory_df, use_container_width=True, hide_index=True)
