from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))


# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="Alert Queue | SentinelFlow",
    page_icon="🚨",
    layout="wide",
)


# =========================================================
# DATA LOADING
# =========================================================
EXPORTS_DIR = ROOT / "artifacts" / "dashboard_exports"


@st.cache_data
def load_dashboard_exports() -> Dict[str, pd.DataFrame]:
    datasets: Dict[str, pd.DataFrame] = {}

    if not EXPORTS_DIR.exists():
        return datasets

    for csv_path in EXPORTS_DIR.glob("*.csv"):
        try:
            df = pd.read_csv(csv_path)
            if df is not None and not df.empty:
                datasets[csv_path.name] = df
        except Exception:
            continue

    return datasets


# =========================================================
# HELPERS
# =========================================================
def normalize_colname(name: str) -> str:
    return (
        str(name)
        .strip()
        .lower()
        .replace(" ", "_")
        .replace("-", "_")
        .replace("/", "_")
        .replace(".", "_")
    )


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [normalize_colname(c) for c in out.columns]
    return out


def first_existing(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    cols = set(df.columns)
    for c in candidates:
        if c in cols:
            return c
    return None


def pick_alert_dataset(datasets: Dict[str, pd.DataFrame]) -> Tuple[pd.DataFrame, str]:
    if not datasets:
        return pd.DataFrame(), ""

    preferred_keywords = [
        "alert",
        "queue",
        "scored_transactions",
        "scored",
        "alerts",
        "transactions",
        "monitor",
    ]

    scored = []
    for name, df in datasets.items():
        if df is None or df.empty:
            continue

        lname = str(name).lower()
        score = 0

        for kw in preferred_keywords:
            if kw in lname:
                score += 2

        ncols = [normalize_colname(c) for c in df.columns]

        if any(c in ncols for c in ["risk_score", "fraud_probability", "model_score", "alert_score", "score"]):
            score += 4
        if any(c in ncols for c in ["transaction_amount", "amount", "amount_eur", "usd_amount"]):
            score += 2
        if any(c in ncols for c in ["transaction_id", "alert_id", "case_id"]):
            score += 2

        scored.append((score, name, df))

    if not scored:
        first_name = list(datasets.keys())[0]
        return datasets[first_name].copy(), first_name

    scored.sort(key=lambda x: x[0], reverse=True)
    _, best_name, best_df = scored[0]
    return best_df.copy(), best_name


def coerce_numeric(df: pd.DataFrame, cols: List[Optional[str]]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c and c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def coerce_datetime(df: pd.DataFrame, cols: List[Optional[str]]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c and c in out.columns:
            out[c] = pd.to_datetime(out[c], errors="coerce")
    return out


def infer_schema(df: pd.DataFrame) -> Dict[str, Optional[str]]:
    schema = {
        "id": first_existing(df, [
            "alert_id", "transaction_id", "case_id", "event_id", "id"
        ]),
        "score": first_existing(df, [
            "alert_score", "risk_score", "fraud_probability", "model_score",
            "fraud_score", "score", "predicted_probability"
        ]),
        "amount": first_existing(df, [
            "transaction_amount", "amount", "amount_eur", "usd_amount",
            "eur_amount", "amount_usd", "tx_amount", "payment_amount"
        ]),
        "country": first_existing(df, [
            "country", "country_code", "issuer_country", "merchant_country",
            "customer_country", "card_country"
        ]),
        "channel": first_existing(df, [
            "channel", "transaction_channel", "entry_channel", "payment_channel"
        ]),
        "payment_type": first_existing(df, [
            "payment_type", "payment_method_type", "payment_method",
            "card_type", "instrument_type"
        ]),
        "merchant_category": first_existing(df, [
            "merchant_category", "merchant_category_name", "mcc_group",
            "merchant_segment", "merchant_type", "category"
        ]),
        "merchant_id": first_existing(df, [
            "merchant_id", "merchant", "merchant_name"
        ]),
        "customer_id": first_existing(df, [
            "customer_id", "client_id", "user_id", "account_id"
        ]),
        "timestamp": first_existing(df, [
            "transaction_ts", "transaction_time", "alert_ts", "event_ts",
            "created_at", "timestamp", "transaction_date", "date"
        ]),
        "rank": first_existing(df, [
            "alert_rank", "priority_rank", "rank", "queue_rank"
        ]),
        "reason": first_existing(df, [
            "alert_reason", "reason", "driver", "top_reason",
            "risk_reason", "explanation"
        ]),
        "risk_bucket": first_existing(df, [
            "risk_bucket", "alert_bucket", "risk_band", "score_band", "priority_bucket"
        ]),
        "status": first_existing(df, [
            "alert_status", "status", "review_status", "case_status"
        ]),
    }
    return schema


def build_risk_bucket(df: pd.DataFrame, score_col: Optional[str]) -> pd.Series:
    if score_col is None or score_col not in df.columns:
        return pd.Series(["Sin score"] * len(df), index=df.index)

    s = pd.to_numeric(df[score_col], errors="coerce")

    if s.dropna().empty:
        return pd.Series(["Sin score"] * len(df), index=df.index)

    if s.max(skipna=True) <= 1.0:
        bins = [-np.inf, 0.40, 0.70, 0.85, np.inf]
        labels = ["Bajo", "Medio", "Alto", "Crítico"]
    else:
        bins = [-np.inf, 40, 70, 85, np.inf]
        labels = ["Bajo", "Medio", "Alto", "Crítico"]

    return pd.cut(s, bins=bins, labels=labels)


def normalize_score_to_100(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    if s.dropna().empty:
        return s
    if s.max(skipna=True) <= 1.0:
        return s * 100
    return s


def normalize_size_index(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce").fillna(0)
    if len(s) == 0:
        return s
    if s.max() == s.min():
        return pd.Series([50] * len(s), index=s.index)
    return 20 + (s - s.min()) / (s.max() - s.min()) * 80


def build_priority_score(df: pd.DataFrame, schema: Dict[str, Optional[str]]) -> pd.Series:
    out = pd.Series(np.zeros(len(df)), index=df.index, dtype=float)

    score_col = schema["score"]
    amount_col = schema["amount"]
    ts_col = schema["timestamp"]

    if score_col and score_col in df.columns:
        s = normalize_score_to_100(df[score_col]).fillna(0)
        out += 0.60 * s

    if amount_col and amount_col in df.columns:
        a = pd.to_numeric(df[amount_col], errors="coerce").fillna(0)
        if a.max() > a.min():
            a_norm = 100 * (a - a.min()) / (a.max() - a.min())
        else:
            a_norm = pd.Series([0] * len(a), index=a.index)
        out += 0.30 * a_norm

    if ts_col and ts_col in df.columns:
        t = pd.to_datetime(df[ts_col], errors="coerce")
        if t.notna().any():
            age_hours = (t.max() - t).dt.total_seconds() / 3600
            age_hours = age_hours.fillna(age_hours.max() if age_hours.notna().any() else 0)
            if age_hours.max() > age_hours.min():
                recency = 100 * (1 - (age_hours - age_hours.min()) / (age_hours.max() - age_hours.min()))
            else:
                recency = pd.Series([50] * len(t), index=t.index)
            out += 0.10 * recency

    return out.round(2)


def human_format_amount(x: float) -> str:
    if pd.isna(x):
        return "N/A"
    x = float(x)
    if abs(x) >= 1_000_000:
        return f"{x/1_000_000:.2f} M"
    if abs(x) >= 1_000:
        return f"{x/1_000:.1f} K"
    return f"{x:,.0f}"


def safe_group_top(
    df: pd.DataFrame,
    group_col: Optional[str],
    score_col: Optional[str],
    amount_col: Optional[str],
    top_n: int = 10
) -> pd.DataFrame:
    if group_col is None or group_col not in df.columns:
        return pd.DataFrame()

    tmp = df.copy()
    tmp[group_col] = tmp[group_col].fillna("Desconocido").astype(str)

    agg_spec = {"alertas": (group_col, "count")}
    if score_col and score_col in tmp.columns:
        agg_spec["score_medio"] = (score_col, "mean")
    if amount_col and amount_col in tmp.columns:
        agg_spec["importe_total"] = (amount_col, "sum")

    out = tmp.groupby(group_col, dropna=False).agg(**agg_spec).reset_index()

    sort_col = "importe_total" if "importe_total" in out.columns else "alertas"
    out = out.sort_values(sort_col, ascending=False).head(top_n)
    return out


def build_reason_text(row: pd.Series, schema: Dict[str, Optional[str]]) -> str:
    parts = []

    score_col = schema["score"]
    amount_col = schema["amount"]
    bucket_col = "sf_risk_bucket"
    channel_col = schema["channel"]
    country_col = schema["country"]
    reason_col = schema["reason"]

    if bucket_col in row and pd.notna(row[bucket_col]):
        parts.append(f"bucket {row[bucket_col]}")
    if score_col and score_col in row.index and pd.notna(row[score_col]):
        parts.append(f"score {float(row[score_col]):.2f}")
    if amount_col and amount_col in row.index and pd.notna(row[amount_col]):
        parts.append(f"importe {human_format_amount(row[amount_col])}")
    if channel_col and channel_col in row.index and pd.notna(row[channel_col]):
        parts.append(f"canal {row[channel_col]}")
    if country_col and country_col in row.index and pd.notna(row[country_col]):
        parts.append(f"país {row[country_col]}")
    if reason_col and reason_col in row.index and pd.notna(row[reason_col]):
        parts.append(str(row[reason_col]))

    return " · ".join(parts[:5]) if parts else "Sin explicación disponible"


# =========================================================
# LOAD + PREPARE DATA
# =========================================================
datasets = load_dashboard_exports()
raw_df, dataset_name = pick_alert_dataset(datasets)

st.title("🚨 Alert Queue")
st.caption("Cola operativa de alertas para priorización manual, lectura ejecutiva y enfoque antifraude.")

if raw_df.empty:
    st.warning("No se encontró un CSV utilizable en artifacts/dashboard_exports/ para construir la cola de alertas.")
    st.stop()

df = standardize_columns(raw_df)
schema = infer_schema(df)

df = coerce_numeric(df, [schema["score"], schema["amount"], schema["rank"]])
df = coerce_datetime(df, [schema["timestamp"]])

if schema["risk_bucket"] and schema["risk_bucket"] in df.columns:
    df["sf_risk_bucket"] = df[schema["risk_bucket"]].astype(str).fillna("Desconocido")
else:
    df["sf_risk_bucket"] = build_risk_bucket(df, schema["score"]).astype(str)

if schema["score"] and schema["score"] in df.columns:
    df["sf_score_100"] = normalize_score_to_100(df[schema["score"]])
else:
    df["sf_score_100"] = np.nan

df["sf_priority_score"] = build_priority_score(df, schema)
df["sf_priority_rank"] = df["sf_priority_score"].rank(method="dense", ascending=False).astype(int)

for col_key in ["country", "channel", "payment_type", "merchant_category", "status"]:
    c = schema.get(col_key)
    if c and c in df.columns:
        df[c] = df[c].fillna("Desconocido").astype(str)

amount_col = schema["amount"]
score_col = schema["score"]
country_col = schema["country"]
channel_col = schema["channel"]
payment_type_col = schema["payment_type"]
merchant_category_col = schema["merchant_category"]
status_col = schema["status"]

df = df.sort_values(["sf_priority_score"], ascending=False).reset_index(drop=True)
df["sf_priority_reason"] = df.apply(lambda row: build_reason_text(row, schema), axis=1)

# =========================================================
# CONTEXTO
# =========================================================
st.markdown(
    """
    **Cómo leer esta página**  
    Esta vista organiza las alertas como una **cola de trabajo priorizada**.  
    Sirve para decidir **qué revisar primero**, **dónde se concentra la presión operativa**
    y **qué importe económico está expuesto** en la parte alta de la cola.
    """
)

with st.expander("Ver fuente y lógica de priorización", expanded=False):
    st.write(f"**Dataset utilizado:** `{dataset_name}`")
    st.write(
        """
        La prioridad operativa combina, cuando están disponibles:
        - score del modelo
        - importe económico
        - recencia temporal

        El objetivo es convertir la salida del modelo en una cola de revisión manual
        creíble y útil para una demo profesional de fraude transaccional.
        """
    )

# =========================================================
# FILTROS
# =========================================================
st.sidebar.header("Filtros operativos")

bucket_options = sorted(df["sf_risk_bucket"].dropna().astype(str).unique().tolist())
selected_buckets = st.sidebar.multiselect(
    "Risk bucket",
    options=bucket_options,
    default=bucket_options
)

def unique_options(base_df: pd.DataFrame, col: Optional[str]) -> List[str]:
    if col and col in base_df.columns:
        return sorted(base_df[col].dropna().astype(str).unique().tolist())
    return []

selected_countries = st.sidebar.multiselect(
    "País",
    options=unique_options(df, country_col),
    default=unique_options(df, country_col)
)

selected_channels = st.sidebar.multiselect(
    "Canal",
    options=unique_options(df, channel_col),
    default=unique_options(df, channel_col)
)

selected_payment_types = st.sidebar.multiselect(
    "Tipo de pago",
    options=unique_options(df, payment_type_col),
    default=unique_options(df, payment_type_col)
)

selected_merchant_categories = st.sidebar.multiselect(
    "Categoría comercio",
    options=unique_options(df, merchant_category_col),
    default=unique_options(df, merchant_category_col)
)

score_slider_min = 0.0
score_slider_max = 100.0
if df["sf_score_100"].notna().any():
    score_slider_min = float(np.floor(df["sf_score_100"].min()))
    score_slider_max = float(np.ceil(df["sf_score_100"].max()))

selected_score_range = st.sidebar.slider(
    "Rango de score",
    min_value=float(score_slider_min),
    max_value=float(score_slider_max if score_slider_max > score_slider_min else score_slider_min + 1.0),
    value=(
        float(score_slider_min),
        float(score_slider_max if score_slider_max > score_slider_min else score_slider_min + 1.0),
    ),
)

amount_slider_min = 0.0
amount_slider_max = 1.0
if amount_col and amount_col in df.columns and df[amount_col].notna().any():
    amount_slider_min = float(df[amount_col].min())
    amount_slider_max = float(df[amount_col].max())

selected_amount_range = st.sidebar.slider(
    "Rango de importe",
    min_value=float(amount_slider_min),
    max_value=float(amount_slider_max if amount_slider_max > amount_slider_min else amount_slider_min + 1.0),
    value=(
        float(amount_slider_min),
        float(amount_slider_max if amount_slider_max > amount_slider_min else amount_slider_min + 1.0),
    ),
)

filtered = df.copy()

filtered = filtered[filtered["sf_risk_bucket"].astype(str).isin(selected_buckets)]

if country_col and country_col in filtered.columns and selected_countries:
    filtered = filtered[filtered[country_col].astype(str).isin(selected_countries)]

if channel_col and channel_col in filtered.columns and selected_channels:
    filtered = filtered[filtered[channel_col].astype(str).isin(selected_channels)]

if payment_type_col and payment_type_col in filtered.columns and selected_payment_types:
    filtered = filtered[filtered[payment_type_col].astype(str).isin(selected_payment_types)]

if merchant_category_col and merchant_category_col in filtered.columns and selected_merchant_categories:
    filtered = filtered[filtered[merchant_category_col].astype(str).isin(selected_merchant_categories)]

filtered = filtered[
    filtered["sf_score_100"].fillna(-np.inf).between(selected_score_range[0], selected_score_range[1])
]

if amount_col and amount_col in filtered.columns:
    filtered = filtered[
        filtered[amount_col].fillna(-np.inf).between(selected_amount_range[0], selected_amount_range[1])
    ]

filtered = filtered.sort_values("sf_priority_score", ascending=False).reset_index(drop=True)

if filtered.empty:
    st.warning("No hay alertas que cumplan los filtros actuales.")
    st.stop()

# =========================================================
# KPIS
# =========================================================
total_alerts = len(filtered)
total_amount = filtered[amount_col].sum() if amount_col and amount_col in filtered.columns else np.nan
avg_score = filtered["sf_score_100"].mean() if filtered["sf_score_100"].notna().any() else np.nan
max_score = filtered["sf_score_100"].max() if filtered["sf_score_100"].notna().any() else np.nan

top_10pct_n = max(1, int(np.ceil(len(filtered) * 0.10)))
top_10pct = filtered.head(top_10pct_n)
top_10pct_amount = top_10pct[amount_col].sum() if amount_col and amount_col in top_10pct.columns else np.nan

top_10pct_amount_share = (
    100 * top_10pct_amount / total_amount
    if amount_col and amount_col in filtered.columns and pd.notna(total_amount) and total_amount != 0
    else np.nan
)

critical_share = 100 * (
    filtered["sf_risk_bucket"].astype(str).isin(["Crítico", "Alto"]).sum() / len(filtered)
)

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Alertas en cola", f"{total_alerts:,.0f}")
c2.metric("Importe total alertado", human_format_amount(total_amount) if pd.notna(total_amount) else "N/A")
c3.metric("Score medio", f"{avg_score:.1f}" if pd.notna(avg_score) else "N/A")
c4.metric("Score máximo", f"{max_score:.1f}" if pd.notna(max_score) else "N/A")
c5.metric("% alto + crítico", f"{critical_share:.1f}%")

c6, c7, c8 = st.columns(3)
c6.metric("Top 10% de la cola", f"{top_10pct_n:,.0f} alertas")
c7.metric("Importe expuesto Top 10%", human_format_amount(top_10pct_amount) if pd.notna(top_10pct_amount) else "N/A")
c8.metric("Concentración económica Top 10%", f"{top_10pct_amount_share:.1f}%" if pd.notna(top_10pct_amount_share) else "N/A")

# =========================================================
# LECTURA EJECUTIVA
# =========================================================
st.markdown("### Lectura ejecutiva / operativa")

top_country = None
if country_col and country_col in filtered.columns:
    x = filtered[country_col].value_counts()
    if not x.empty:
        top_country = x.index[0]

top_channel = None
if channel_col and channel_col in filtered.columns:
    x = filtered[channel_col].value_counts()
    if not x.empty:
        top_channel = x.index[0]

texto = f"""
- La cola actual contiene **{total_alerts:,.0f} alertas**.
- El **Top 10%** concentra **{top_10pct_amount_share:.1f}% del importe alertado**.
- Los buckets **alto + crítico** representan **{critical_share:.1f}%** de la cola filtrada.
"""
if top_country is not None:
    texto += f"\n- El principal foco geográfico en esta vista es **{top_country}**."
if top_channel is not None:
    texto += f"\n- El canal con mayor presión operativa es **{top_channel}**."
texto += "\n- La lógica recomendada es revisar primero la parte alta de la cola, donde coinciden mayor score, mayor importe y mayor urgencia relativa."
st.markdown(texto)

# =========================================================
# VISUALES
# =========================================================
st.markdown("### Priorización y composición de la cola")
st.caption("Estos gráficos separan alertas críticas de alertas rutinarias y muestran la concentración operativa.")

left, right = st.columns(2)

with left:
    bucket_order = ["Crítico", "Alto", "Medio", "Bajo", "Sin score", "Desconocido"]
    bucket_counts = (
        filtered["sf_risk_bucket"]
        .astype(str)
        .value_counts(dropna=False)
        .rename_axis("bucket")
        .reset_index(name="alertas")
    )
    bucket_counts["bucket"] = pd.Categorical(bucket_counts["bucket"], categories=bucket_order, ordered=True)
    bucket_counts = bucket_counts.sort_values("bucket")

    fig_bucket = px.bar(
        bucket_counts,
        x="bucket",
        y="alertas",
        title="Distribución por risk bucket",
        text="alertas",
    )
    fig_bucket.update_layout(height=360, xaxis_title="", yaxis_title="Alertas")
    st.plotly_chart(fig_bucket, use_container_width=True)

with right:
    if amount_col and amount_col in filtered.columns and filtered["sf_score_100"].notna().any():
        scatter_df = filtered.head(2000).copy()
        hover_cols = [c for c in [
            schema["id"], country_col, channel_col, payment_type_col, merchant_category_col, "sf_risk_bucket"
        ] if c and c in scatter_df.columns]

        scatter_df["sf_size"] = normalize_size_index(scatter_df[amount_col])

        fig_scatter = px.scatter(
            scatter_df,
            x="sf_score_100",
            y=amount_col,
            color="sf_risk_bucket",
            size="sf_size",
            hover_data=hover_cols,
            title="Score vs importe alertado",
        )
        fig_scatter.update_layout(height=360, xaxis_title="Score (0-100)", yaxis_title="Importe")
        st.plotly_chart(fig_scatter, use_container_width=True)
    else:
        st.info("No hay columnas suficientes para el gráfico de score vs importe.")

left2, right2 = st.columns(2)

with left2:
    if channel_col and channel_col in filtered.columns:
        channel_agg = (
            filtered.groupby(channel_col)
            .agg(
                alertas=(channel_col, "count"),
                score_medio=("sf_score_100", "mean"),
            )
            .reset_index()
            .sort_values("alertas", ascending=False)
            .head(10)
        )

        fig_channel = px.bar(
            channel_agg,
            x=channel_col,
            y="alertas",
            color="score_medio",
            title="Alertas por canal",
            text="alertas",
        )
        fig_channel.update_layout(height=360, xaxis_title="", yaxis_title="Alertas")
        st.plotly_chart(fig_channel, use_container_width=True)
    else:
        st.info("No se encontró una columna de canal.")

with right2:
    if country_col and country_col in filtered.columns:
        agg_dict = {"alertas": (country_col, "count")}
        if amount_col and amount_col in filtered.columns:
            agg_dict["importe_total"] = (amount_col, "sum")

        country_agg = (
            filtered.groupby(country_col)
            .agg(**agg_dict)
            .reset_index()
            .sort_values("alertas", ascending=False)
            .head(12)
        )

        color_col = "importe_total" if "importe_total" in country_agg.columns else "alertas"

        fig_country = px.bar(
            country_agg,
            x=country_col,
            y="alertas",
            color=color_col,
            title="Alertas por país",
            text="alertas",
        )
        fig_country.update_layout(height=360, xaxis_title="", yaxis_title="Alertas")
        st.plotly_chart(fig_country, use_container_width=True)
    else:
        st.info("No se encontró una columna de país.")

# =========================================================
# FOCOS PRIORITARIOS
# =========================================================
st.markdown("### Focos prioritarios")
st.caption("Este bloque resume dónde conviene enfocar primero al equipo antifraude.")

f1, f2 = st.columns(2)

with f1:
    group_choice_1 = None
    title_1 = None

    if payment_type_col and payment_type_col in filtered.columns:
        group_choice_1 = payment_type_col
        title_1 = "tipo de pago"
    elif merchant_category_col and merchant_category_col in filtered.columns:
        group_choice_1 = merchant_category_col
        title_1 = "categoría de comercio"

    if group_choice_1:
        focus_agg = (
            filtered.groupby(group_choice_1)
            .agg(
                alertas=(group_choice_1, "count"),
                score_medio=("sf_score_100", "mean"),
            )
            .reset_index()
            .sort_values(["score_medio", "alertas"], ascending=False)
            .head(10)
        )

        fig_focus = px.bar(
            focus_agg,
            x=group_choice_1,
            y="score_medio",
            color="alertas",
            title=f"Severidad media por {title_1}",
            text="alertas",
        )
        fig_focus.update_layout(height=360, xaxis_title="", yaxis_title="Score medio")
        st.plotly_chart(fig_focus, use_container_width=True)
    else:
        st.info("No se encontró payment_type ni merchant_category.")

with f2:
    if country_col and country_col in filtered.columns and channel_col and channel_col in filtered.columns:
        heat_df = (
            filtered.groupby([country_col, channel_col])
            .size()
            .reset_index(name="alertas")
        )

        if not heat_df.empty:
            fig_heat = px.density_heatmap(
                heat_df,
                x=channel_col,
                y=country_col,
                z="alertas",
                title="Concentración país × canal",
            )
            fig_heat.update_layout(height=360, xaxis_title="", yaxis_title="")
            st.plotly_chart(fig_heat, use_container_width=True)
        else:
            st.info("No hay datos suficientes para país × canal.")
    else:
        st.info("No hay columnas suficientes para país × canal.")

# =========================================================
# TABLAS RESUMEN
# =========================================================
st.markdown("### Casos más críticos / focos prioritarios")

tabs = st.tabs(["Países", "Canales", "Tipologías"])

with tabs[0]:
    out = safe_group_top(filtered, country_col, "sf_score_100", amount_col, top_n=12)
    if not out.empty:
        if "score_medio" in out.columns:
            out["score_medio"] = out["score_medio"].round(1)
        if "importe_total" in out.columns:
            out["importe_total"] = out["importe_total"].round(2)
        st.dataframe(out, use_container_width=True, hide_index=True)
    else:
        st.info("No hay columna de país disponible.")

with tabs[1]:
    out = safe_group_top(filtered, channel_col, "sf_score_100", amount_col, top_n=12)
    if not out.empty:
        if "score_medio" in out.columns:
            out["score_medio"] = out["score_medio"].round(1)
        if "importe_total" in out.columns:
            out["importe_total"] = out["importe_total"].round(2)
        st.dataframe(out, use_container_width=True, hide_index=True)
    else:
        st.info("No hay columna de canal disponible.")

with tabs[2]:
    tipology_col = payment_type_col if payment_type_col and payment_type_col in filtered.columns else merchant_category_col
    out = safe_group_top(filtered, tipology_col, "sf_score_100", amount_col, top_n=12)
    if not out.empty:
        if "score_medio" in out.columns:
            out["score_medio"] = out["score_medio"].round(1)
        if "importe_total" in out.columns:
            out["importe_total"] = out["importe_total"].round(2)
        st.dataframe(out, use_container_width=True, hide_index=True)
    else:
        st.info("No hay payment_type ni merchant_category disponible.")

# =========================================================
# TABLA OPERATIVA PRINCIPAL
# =========================================================
st.markdown("### Cola operativa de revisión")
st.caption("Tabla principal para priorizar trabajo manual. La recomendación es ordenar por prioridad operativa o por score.")

sort_options = {
    "Prioridad operativa": "sf_priority_score",
    "Rank operativo": "sf_priority_rank",
    "Score": "sf_score_100",
}

if amount_col and amount_col in filtered.columns:
    sort_options["Importe"] = amount_col
if schema["timestamp"] and schema["timestamp"] in filtered.columns:
    sort_options["Fecha"] = schema["timestamp"]
if country_col and country_col in filtered.columns:
    sort_options["País"] = country_col
if channel_col and channel_col in filtered.columns:
    sort_options["Canal"] = channel_col

selected_sort_label = st.selectbox("Ordenar tabla por", options=list(sort_options.keys()), index=0)
selected_sort_col = sort_options[selected_sort_label]
sort_ascending = st.checkbox("Orden ascendente", value=False)
max_rows = st.slider("Filas visibles", min_value=20, max_value=500, value=100, step=20)

display_cols = [
    "sf_priority_rank",
    "sf_priority_score",
    "sf_risk_bucket",
    "sf_priority_reason",
]

for c in [
    schema["id"],
    schema["timestamp"],
    schema["score"],
    "sf_score_100",
    schema["amount"],
    schema["country"],
    schema["channel"],
    schema["payment_type"],
    schema["merchant_category"],
    schema["merchant_id"],
    schema["customer_id"],
    schema["status"],
]:
    if c and c in filtered.columns and c not in display_cols:
        display_cols.append(c)

table_df = filtered.sort_values(selected_sort_col, ascending=sort_ascending).copy()

rename_map = {
    "sf_priority_rank": "rank_operativo",
    "sf_priority_score": "prioridad_operativa",
    "sf_risk_bucket": "risk_bucket",
    "sf_priority_reason": "explicacion_prioridad",
    "sf_score_100": "score_100",
}

table_df = table_df.rename(columns=rename_map)

visible_cols = []
for c in display_cols:
    if c in rename_map:
        visible_cols.append(rename_map[c])
    elif c in table_df.columns:
        visible_cols.append(c)

visible_cols = [c for c in visible_cols if c in table_df.columns]

st.dataframe(
    table_df[visible_cols].head(max_rows),
    use_container_width=True,
    hide_index=True,
)

csv_export = table_df[visible_cols].to_csv(index=False).encode("utf-8")
st.download_button(
    label="Descargar cola filtrada en CSV",
    data=csv_export,
    file_name="sentinelflow_alert_queue_filtered.csv",
    mime="text/csv",
)

# =========================================================
# TOP CASOS CRÍTICOS
# =========================================================
st.markdown("### Top casos críticos")
st.caption("Selección corta de alertas que combinan prioridad operativa, severidad y exposición económica.")

top_n_cases = st.slider("Número de casos críticos visibles", min_value=10, max_value=100, value=25, step=5)

sort_cols = ["sf_priority_score"]
ascending_flags = [False]

if amount_col and amount_col in filtered.columns:
    sort_cols.append(amount_col)
    ascending_flags.append(False)

top_cases = filtered.sort_values(sort_cols, ascending=ascending_flags).head(top_n_cases).copy()
top_cases = top_cases.rename(columns=rename_map)

top_cols = []
for c in [
    "sf_priority_rank",
    "sf_priority_score",
    "sf_risk_bucket",
    "sf_priority_reason",
    schema["id"],
    schema["timestamp"],
    "sf_score_100",
    schema["amount"],
    schema["country"],
    schema["channel"],
    schema["payment_type"],
    schema["merchant_category"],
    schema["merchant_id"],
    schema["customer_id"],
]:
    if c in rename_map:
        top_cols.append(rename_map[c])
    elif c and c in top_cases.columns:
        top_cols.append(c)

top_cols = [c for c in top_cols if c in top_cases.columns]

st.dataframe(
    top_cases[top_cols],
    use_container_width=True,
    hide_index=True,
)
