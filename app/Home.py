from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.data.load_dashboard_data import get_dashboard_exports_dir, list_expected_files


st.set_page_config(
    page_title="SentinelFlow",
    page_icon="🛡️",
    layout="wide",
)


st.title("🛡️ SentinelFlow")
st.subheader("Fraud Detection Command Center")

st.markdown(
    """
    **SentinelFlow** es una plataforma demo de detección de fraude transaccional de extremo a extremo,
    construida sobre datos sintéticos, feature engineering profesional, modelado supervisado y salidas
    analíticas listas para operación.

    Este dashboard consume directamente los CSV generados en:

    `artifacts/dashboard_exports/`
    """
)

exports_dir = get_dashboard_exports_dir()
inventory_df = list_expected_files()

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Expected exported datasets", len(inventory_df))
with col2:
    st.metric("Available exported datasets", int(inventory_df["exists"].sum()))
with col3:
    st.metric("Missing exported datasets", int((~inventory_df["exists"]).sum()))

st.markdown("---")

st.markdown("### Project status")

status_col1, status_col2 = st.columns([2, 1])

with status_col1:
    st.info(
        """
        **Current dashboard phase:** Base Streamlit structure initialized.

        **Next target:** Build executive Overview and then expand module by module:
        Transaction Monitoring, Alert Queue, Review Capacity, Suspicious Entities,
        Model Performance, and Case Explorer.
        """
    )

with status_col2:
    st.code(str(exports_dir), language="text")

st.markdown("### Export inventory")
st.dataframe(inventory_df, use_container_width=True, hide_index=True)

st.markdown("---")
st.markdown(
    """
    Usa el menú lateral para navegar por las páginas del dashboard.
    """
)
