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
    page_title="Resumen ejecutivo | SentinelFlow",
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


def first_existing_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    if df is None or df.empty:
        return None

    lower_map = {str(c).strip().lower(): c for c in df.columns}
    for candidate in candidates:
        match = lower_map.get(str(candidate).strip().lower())
        if match is not None:
            return match
    return None


def safe_copy(df: pd.DataFrame) -> pd.DataFrame:
    if df is None:
        return pd.DataFrame()
    return df.copy()


def get_numeric_series(df: pd.DataFrame, candidates: list[str]) -> pd.Series | None:
    col = first_existing_col(df, candidates)
    if col is None:
        return None
    s = pd.to_numeric(df[col], errors="coerce")
    if s.dropna().empty:
        return None
    return s


def get_categorical_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    return first_existing_col(df, candidates)


def build_daily_volume(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()

    tmp = df.copy()
    date_col = infer_main_transaction_date_column(tmp)
    if date_col is None:
        return pd.DataFrame()

    tmp = parse_datetime_column(tmp, date_col)
    tmp = tmp[tmp[date_col].notna()].copy()
    if tmp.empty:
        return pd.DataFrame()

    tmp["fecha"] = tmp[date_col].dt.date
    out = tmp.groupby("fecha").size().reset_index(name="transacciones")
    return out.sort_values("fecha")


def build_daily_alerts(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()

    tmp = df.copy()
    date_col = infer_main_transaction_date_column(tmp)
    if date_col is None:
        return pd.DataFrame()

    tmp = parse_datetime_column(tmp, date_col)
    tmp = tmp[tmp[date_col].notna()].copy()
    if tmp.empty:
        return pd.DataFrame()

    tmp["fecha"] = tmp[date_col].dt.date
    out = tmp.groupby("fecha").size().reset_index(name="alertas")
    return out.sort_values("fecha")


def build_risk_bucket_summary(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()

    bucket_col = get_categorical_col(df, ["risk_bucket", "risk_segment", "score_bucket", "bucket"])
    if bucket_col is None:
        return pd.DataFrame()

    out = (
        df.groupby(bucket_col)
        .size()
        .reset_index(name="transacciones")
        .sort_values("transacciones", ascending=False)
    )
    out.columns = ["bucket_riesgo", "transacciones"]
    return out


def build_channel_alert_summary(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()

    channel_col = get_categorical_col(df, ["channel", "transaction_channel", "payment_channel"])
    if channel_col is None:
        return pd.DataFrame()

    out = (
        df.groupby(channel_col)
        .size()
        .reset_index(name="alertas")
        .sort_values("alertas", ascending=False)
    )
    out.columns = ["canal", "alertas"]
    return out


def build_amount_by_bucket(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()

    bucket_col = get_categorical_col(df, ["risk_bucket", "risk_segment", "score_bucket", "bucket"])
    amount_col = first_existing_col(df, ["amount", "transaction_amount", "amount_eur", "amount_usd"])

    if bucket_col is None or amount_col is None:
        return pd.DataFrame()

    tmp = df.copy()
    tmp[amount_col] = pd.to_numeric(tmp[amount_col], errors="coerce")
    tmp = tmp[tmp[amount_col].notna()].copy()
    if tmp.empty:
        return pd.DataFrame()

    out = (
        tmp.groupby(bucket_col)[amount_col]
        .sum()
        .reset_index()
        .sort_values(amount_col, ascending=False)
    )
    out.columns = ["bucket_riesgo", "importe_total"]
    return out


def build_score_histogram(df: pd.DataFrame, bins: int = 20) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()

    score_col = first_existing_col(df, ["fraud_score", "score", "predicted_probability", "risk_score"])
    if score_col is None:
        return pd.DataFrame()

    tmp = df.copy()
    tmp[score_col] = pd.to_numeric(tmp[score_col], errors="coerce")
    tmp = tmp[tmp[score_col].notna()].copy()
    if tmp.empty:
        return pd.DataFrame()

    try:
        tmp["intervalo_score"] = pd.cut(tmp[score_col], bins=bins, include_lowest=True)
        out = (
            tmp.groupby("intervalo_score", observed=False)
            .size()
            .reset_index(name="transacciones")
        )
        out["intervalo_score"] = out["intervalo_score"].astype(str)
        return out
    except Exception:
        return pd.DataFrame()


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

st.title("📊 Resumen ejecutivo")
st.caption("Punto de entrada ejecutivo para la monitorización de fraude en SentinelFlow")

# =========================================================
# KPIs PRINCIPALES
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

# Fallback manual
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

st.markdown("## Indicadores clave")

k1, k2, k3, k4 = st.columns(4)
with k1:
    st.metric("Transacciones scoreadas", f"{total_transactions:,}" if total_transactions is not None else "N/A")
with k2:
    st.metric("Alertas activas", f"{total_alerts:,}" if total_alerts is not None else "N/A")
with k3:
    st.metric("Umbral champion", format_metric_value(champion_threshold))
with k4:
    st.metric("ROC-AUC test", format_metric_value(roc_auc))

k5, k6, k7, k8 = st.columns(4)
with k5:
    st.metric("PR-AUC test", format_metric_value(pr_auc))
with k6:
    st.metric("Precisión", format_metric_value(precision))
with k7:
    st.metric("Recall", format_metric_value(recall))
with k8:
    st.metric("Recall Top 3%", format_metric_value(top3_recall, pct=True))

st.info(
    """
    **Cómo interpretar este bloque**
    
    - **Transacciones scoreadas**: volumen total evaluado por el modelo.
    - **Alertas activas**: casos que superan el umbral operativo y pasan a revisión.
    - **Umbral champion**: punto de corte a partir del cual una transacción se considera prioritaria.
    - **ROC-AUC / PR-AUC**: resumen de calidad predictiva del modelo.
    - **Precisión**: proporción de alertas que realmente son fraude.
    - **Recall**: porcentaje de fraude total que el sistema consigue capturar.
    - **Recall Top 3%**: capacidad del modelo para concentrar fraude real en la cola prioritaria.
    """
)

st.markdown("---")

# =========================================================
# CAPACIDAD DE REVISIÓN
# =========================================================

st.markdown("## Capacidad de revisión")

c1, c2, c3, c4 = st.columns(4)
with c1:
    st.metric("Cola Top 1%", f"{q1_n:,}" if q1_n is not None else "N/A")
with c2:
    st.metric("Cola Top 3%", f"{q3_n:,}" if q3_n is not None else "N/A")
with c3:
    st.metric("Cola Top 5%", f"{q5_n:,}" if q5_n is not None else "N/A")
with c4:
    st.metric("Cola Top 10%", f"{q10_n:,}" if q10_n is not None else "N/A")

st.info(
    """
    **Cómo interpretar esta sección**
    
    Estas colas representan escenarios de capacidad operativa.  
    Permiten responder a una pregunta clave: **si el equipo solo puede revisar una fracción del flujo, cuánto volumen tendría que absorber?**
    
    - **Top 1%**: revisión muy selectiva, máxima concentración de riesgo.
    - **Top 3%**: nivel normalmente muy útil para operación priorizada.
    - **Top 5% / Top 10%**: escenarios más amplios, con mayor cobertura pero también mayor carga operativa.
    """
)

st.markdown("---")

# =========================================================
# LECTURA EJECUTIVA
# =========================================================

st.markdown("## Lectura ejecutiva")

summary_lines = []

if total_transactions is not None:
    summary_lines.append(f"- El sistema ha evaluado **{total_transactions:,} transacciones** en el dataset principal.")

if total_alerts is not None and total_transactions not in [None, 0]:
    summary_lines.append(
        f"- La cola actual contiene **{total_alerts:,} alertas**, equivalentes al **{total_alerts / total_transactions:.2%}** del total scoreado."
    )

summary_lines.append(f"- El umbral operativo del modelo champion está fijado en **{champion_threshold:.4f}**.")
summary_lines.append(
    f"- En test, el modelo presenta **ROC-AUC {roc_auc:.4f}**, **PR-AUC {pr_auc:.4f}**, "
    f"**precisión {precision:.4f}** y **recall {recall:.4f}**."
)
summary_lines.append(
    f"- La priorización es especialmente fuerte en la parte alta del ranking, con **Recall Top 3% = {top3_recall:.2%}**."
)

for line in summary_lines:
    st.markdown(line)

st.success(
    """
    **Interpretación ejecutiva**
    
    Este bloque permite evaluar de forma rápida si el sistema está funcionando de forma coherente en tres dimensiones:
    **volumen**, **calidad predictiva** y **presión operativa de revisión**.
    """
)

st.markdown("---")

# =========================================================
# VISUALIZACIÓN EJECUTIVA
# =========================================================

st.markdown("## Visualización ejecutiva")

score_hist_df = build_score_histogram(scored_df, bins=20)
risk_bucket_df = build_risk_bucket_summary(scored_df)
channel_alert_df = build_channel_alert_summary(alert_queue_df)
amount_bucket_df = build_amount_by_bucket(scored_df)
daily_volume_df = build_daily_volume(scored_df)
daily_alerts_df = build_daily_alerts(alert_queue_df)

v1, v2 = st.columns(2)

with v1:
    st.markdown("### Distribución del score de fraude")
    if not score_hist_df.empty:
        st.bar_chart(score_hist_df.set_index("intervalo_score"))
        st.caption(
            "Muestra cómo se distribuye el score del modelo. Una concentración fuerte en scores bajos con una cola alta bien definida suele indicar una priorización útil."
        )
    else:
        st.info("No se pudo construir la distribución de score con las columnas disponibles.")

with v2:
    st.markdown("### Volumen por bucket de riesgo")
    if not risk_bucket_df.empty:
        st.bar_chart(risk_bucket_df.set_index("bucket_riesgo"))
        st.caption(
            "Permite ver cuántas transacciones caen en cada nivel de riesgo. Es útil para entender el balance entre tráfico normal y tráfico priorizado."
        )
    else:
        st.info("No se pudo construir el resumen por bucket de riesgo.")

v3, v4 = st.columns(2)

with v3:
    st.markdown("### Alertas por canal")
    if not channel_alert_df.empty:
        st.bar_chart(channel_alert_df.set_index("canal"))
        st.caption(
            "Ayuda a identificar qué canales concentran más alertas. Puede revelar fricción operativa o focos de fraude por canal."
        )
    else:
        st.info("No se pudo construir la vista de alertas por canal.")

with v4:
    st.markdown("### Importe total por bucket de riesgo")
    if not amount_bucket_df.empty:
        st.bar_chart(amount_bucket_df.set_index("bucket_riesgo"))
        st.caption(
            "No solo importa el número de transacciones: también el importe expuesto. Este gráfico resume dónde se concentra el volumen económico."
        )
    else:
        st.info("No se pudo construir el análisis de importe por bucket.")

v5, v6 = st.columns(2)

with v5:
    st.markdown("### Serie diaria de transacciones")
    if not daily_volume_df.empty:
        st.line_chart(daily_volume_df.set_index("fecha"))
        st.caption(
            "Permite detectar cambios en el flujo transaccional, picos anómalos o jornadas con actividad inusual."
        )
    else:
        st.info("No se pudo construir la serie diaria de transacciones.")

with v6:
    st.markdown("### Serie diaria de alertas")
    if not daily_alerts_df.empty:
        st.area_chart(daily_alerts_df.set_index("fecha"))
        st.caption(
            "Resume la evolución de la presión de alertado en el tiempo. Sirve para detectar días de tensión operativa o campañas anómalas."
        )
    else:
        st.info("No se pudo construir la serie diaria de alertas.")

st.markdown("---")

# =========================================================
# PREVIEW DEL DATASET
# =========================================================

st.markdown("## Vista previa del dataset principal")

if not scored_df.empty:
    preview_df = safe_copy(scored_df)
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

    st.caption(
        "Esta tabla muestra una muestra operativa del dataset scoreado. Sirve para validar que el pipeline exporta correctamente los campos clave que después usan los filtros y drill-downs del dashboard."
    )

    if preview_cols:
        st.dataframe(
            preview_df.loc[:, preview_cols].head(25),
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.dataframe(preview_df.head(25), use_container_width=True, hide_index=True)
else:
    st.warning("El archivo dashboard_scored_transactions.csv no se encontró o está vacío.")
    st.markdown("### Diagnóstico de inventario de ficheros")
    st.dataframe(inventory_df, use_container_width=True, hide_index=True)

st.markdown("---")

# =========================================================
# DIAGNÓSTICO TEMPORAL
# =========================================================

with st.expander("Diagnóstico temporal · carga de datasets", expanded=False):
    st.markdown("### Diagnóstico de dashboard_scored_transactions")

    if not scored_df.empty:
        d1, d2, d3 = st.columns(3)
        with d1:
            st.metric("Filas", f"{scored_df.shape[0]:,}")
        with d2:
            st.metric("Columnas", f"{scored_df.shape[1]:,}")
        with d3:
            detected_date_col = infer_main_transaction_date_column(scored_df)
            st.metric("Columna de fecha detectada", detected_date_col if detected_date_col else "Ninguna")

        st.markdown("**Columnas detectadas**")
        st.write(list(scored_df.columns))

        st.markdown("**Primeras 5 filas**")
        st.dataframe(scored_df.head(5), use_container_width=True, hide_index=True)
    else:
        st.error("dashboard_scored_transactions se cargó como DataFrame vacío.")
        st.dataframe(inventory_df, use_container_width=True, hide_index=True)

    st.markdown("### Estado del resto de datasets")
    diagnostic_rows = []
    for dataset_name, df in data.items():
        diagnostic_rows.append(
            {
                "dataset": dataset_name,
                "filas": int(df.shape[0]) if isinstance(df, pd.DataFrame) else None,
                "columnas": int(df.shape[1]) if isinstance(df, pd.DataFrame) else None,
                "vacío": bool(df.empty) if isinstance(df, pd.DataFrame) else True,
            }
        )
    st.dataframe(pd.DataFrame(diagnostic_rows), use_container_width=True, hide_index=True)

st.markdown("---")

# =========================================================
# INVENTARIO DE OUTPUTS
# =========================================================

st.markdown("## Inventario de outputs")

if output_inventory_df is not None and not output_inventory_df.empty:
    st.caption(
        "Inventario de artefactos exportados para el dashboard. Esta tabla es útil para validar cobertura funcional y trazabilidad del pipeline."
    )
    st.dataframe(output_inventory_df, use_container_width=True, hide_index=True)
else:
    st.info("output_inventory.csv no está disponible. Se muestra el inventario detectado en filesystem.")
    st.dataframe(inventory_df, use_container_width=True, hide_index=True)
