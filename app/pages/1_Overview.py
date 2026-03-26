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


data = get_data()

scored_df = data["dashboard_scored_transactions"]
executive_kpis_df = data["executive_kpis"]
alert_queue_df = data["alert_queue"]
q1_df = data["review_queue_top_1pct"]
q3_df = data["review_queue_top_3pct"]
q5_df = data["review_queue_top_5pct"]
q10_df = data["review_queue_top_10pct"]
queue_summary_df = data["queue_summary"]
output_inventory_df = data["output_inventory"]


st.title("📊 Overview")
st.caption("Executive entry point for the SentinelFlow fraud monitoring dashboard")

# -------------------------------------------------------------------
# KPI extraction
# -------------------------------------------------------------------

total_transactions = len(scored_df) if not scored_df.empty else None
total_alerts = len(alert_queue_df) if not alert_queue_df.empty else None
q1_n = len(q1_df) if not q1_df.empty else None
q3_n = len(q3_df) if not q3_df.empty else None
q5_n = len(q5_df) if not q5_df.empty else None
q10_n = len(q10_df) if not q10_df.empty else None

champion_threshold = extract_kpi_value(
    executive_kpis_df,
    candidate_keys=[
        "champion_threshold",
        "selected_threshold",
        "operating_threshold",
        "model_threshold",
    ],
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

# -------------------------------------------------------------------
# Header KPIs
# -------------------------------------------------------------------

k1, k2, k3, k4 = st.columns(4)

with k1:
    st.metric(
        "Scored transactions",
        f"{total_transactions:,}" if total_transactions is not None else "N/A",
    )

with k2:
    st.metric(
        "Alert queue",
        f"{total_alerts:,}" if total_alerts is not None else "N/A",
    )

with k3:
    st.metric(
        "Champion threshold",
        f"{champion_threshold:.4f}" if champion_threshold is not None else "N/A",
    )

with k4:
    st.metric(
        "Test ROC-AUC",
        f"{roc_auc:.4f}" if roc_auc is not None else "N/A",
    )

k5, k6, k7, k8 = st.columns(4)

with k5:
    st.metric(
        "Test PR-AUC",
        f"{pr_auc:.4f}" if pr_auc is not None else "N/A",
    )

with k6:
    st.metric(
        "Precision",
        f"{precision:.4f}" if precision is not None else "N/A",
    )

with k7:
    st.metric(
        "Recall",
        f"{recall:.4f}" if recall is not None else "N/A",
    )

with k8:
    st.metric(
        "Top 3% recall",
        f"{top3_recall:.4f}" if top3_recall is not None else "N/A",
    )

st.markdown("---")

# -------------------------------------------------------------------
# Review capacity summary
# -------------------------------------------------------------------

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

# -------------------------------------------------------------------
# Executive reading
# -------------------------------------------------------------------

st.markdown("## Executive reading")

summary_lines = []

if total_transactions is not None:
    summary_lines.append(f"- Total scored population: **{total_transactions:,} transactions**.")

if total_alerts is not None and total_transactions:
    alert_rate = total_alerts / total_transactions
    summary_lines.append(
        f"- Current production-style alert queue: **{total_alerts:,} alerts** "
        f"({alert_rate:.2%} of scored transactions)."
    )

if champion_threshold is not None:
    summary_lines.append(f"- Operational champion threshold currently set at **{champion_threshold:.4f}**.")

if roc_auc is not None and pr_auc is not None:
    summary_lines.append(
        f"- Champion model discrimination is strong with **ROC-AUC {roc_auc:.4f}** "
        f"and **PR-AUC {pr_auc:.4f}**."
    )

if top3_recall is not None:
    summary_lines.append(
        f"- At reviewer-focused prioritization, the model captures **{top3_recall:.2%}** "
        f"of all frauds within the **top 3% highest-risk transactions**."
    )

if summary_lines:
    st.markdown("\n".join(summary_lines))
else:
    st.warning("No executive KPI summary could be inferred from the current exported files.")

st.markdown("---")

# -------------------------------------------------------------------
# Dataset preview
# -------------------------------------------------------------------

st.markdown("## Core dataset preview")

if scored_df.empty:
    st.warning("dashboard_scored_transactions.csv is missing or empty.")
else:
    preview_df = scored_df.copy()

    date_col = infer_main_transaction_date_column(preview_df)
    if date_col is not None:
        preview_df = parse_datetime_column(preview_df, date_col)

    st.dataframe(preview_df.head(25), use_container_width=True, hide_index=True)

st.markdown("---")

# -------------------------------------------------------------------
# Inventory / traceability
# -------------------------------------------------------------------

st.markdown("## Output inventory")

if output_inventory_df.empty:
    st.info("output_inventory.csv is missing or empty.")
else:
    st.dataframe(output_inventory_df, use_container_width=True, hide_index=True)

st.markdown("---")

st.markdown(
    """
    **Next build recommendation:** continue with **Transaction Monitoring** and make it the first
    analytical page after Overview.
    """
)
