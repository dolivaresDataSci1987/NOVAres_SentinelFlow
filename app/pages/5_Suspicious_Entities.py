from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# =========================================================
# PATHS / CONFIG
# =========================================================
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

st.set_page_config(
    page_title="Suspicious Entities | SentinelFlow",
    page_icon="🕵️",
    layout="wide",
)

try:
    from src.data.load_dashboard_data import load_selected_datasets  # type: ignore
except Exception:
    load_selected_datasets = None


# =========================================================
# ESTILO
# =========================================================
st.markdown(
    """
    <style>
    .se-info-box {
        background: linear-gradient(135deg, rgba(17,24,39,0.98), rgba(31,41,55,0.96));
        border: 1px solid rgba(148,163,184,0.20);
        border-radius: 16px;
        padding: 18px 20px;
        margin-bottom: 14px;
    }
    .se-kpi-note {
        font-size: 0.88rem;
        color: #94a3b8;
        margin-top: -6px;
    }
    .se-block-help {
        font-size: 0.93rem;
        color: #cbd5e1;
        margin-bottom: 8px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# HELPERS: FORMATO
# =========================================================
def fmt_int(x: float | int | None) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "N/A"
    return f"{int(round(float(x))):,}".replace(",", ".")


def fmt_num(x: float | int | None, digits: int = 2) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "N/A"
    return f"{float(x):,.{digits}f}".replace(",", "X").replace(".", ",").replace("X", ".")


def fmt_pct(x: float | int | None, digits: int = 1) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "N/A"
    return f"{float(x):.{digits}f}%"


def fmt_money(x: float | int | None, currency: str = "€") -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "N/A"
    return f"{currency} {float(x):,.0f}".replace(",", ".")


# =========================================================
# HELPERS: SCHEMA
# =========================================================
def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(c).strip() for c in out.columns]
    return out


def find_first_existing(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    cols_lower = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in cols_lower:
            return cols_lower[cand.lower()]
    return None


def safe_to_datetime(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce")


def safe_to_numeric(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def find_date_col(df: pd.DataFrame) -> Optional[str]:
    candidates = [
        "alert_date",
        "alert_ts",
        "alert_timestamp",
        "queue_date",
        "review_date",
        "transaction_date",
        "transaction_ts",
        "event_date",
        "event_ts",
        "created_at",
        "created_ts",
        "timestamp",
        "date",
    ]
    return find_first_existing(df, candidates)


def find_alert_id_col(df: pd.DataFrame) -> Optional[str]:
    candidates = [
        "alert_id",
        "queue_id",
        "case_id",
        "event_id",
        "transaction_id",
        "id",
    ]
    return find_first_existing(df, candidates)


def find_risk_score_col(df: pd.DataFrame) -> Optional[str]:
    candidates = [
        "alert_score",
        "priority_score",
        "risk_score",
        "model_score",
        "fraud_risk_score",
        "fraud_probability",
        "predicted_fraud_probability",
        "predicted_risk_probability",
        "score",
        "probability",
    ]
    return find_first_existing(df, candidates)


def find_risk_bucket_col(df: pd.DataFrame) -> Optional[str]:
    candidates = [
        "risk_bucket",
        "alert_bucket",
        "priority_bucket",
        "score_bucket",
        "risk_band",
        "risk_segment",
        "priority_segment",
        "predicted_risk_segment",
    ]
    return find_first_existing(df, candidates)


def find_priority_col(df: pd.DataFrame) -> Optional[str]:
    candidates = [
        "priority_tier",
        "priority_level",
        "priority",
        "queue_priority",
        "alert_priority",
        "review_priority",
    ]
    return find_first_existing(df, candidates)


def find_amount_col(df: pd.DataFrame) -> Optional[str]:
    candidates = [
        "amount",
        "transaction_amount",
        "usd_amount",
        "eur_amount",
        "gbp_amount",
        "payment_amount",
        "authorized_amount",
        "captured_amount",
        "declined_amount",
        "exposure_amount",
        "economic_exposure",
        "expected_loss",
        "alert_amount",
        "case_amount",
    ]
    return find_first_existing(df, candidates)


def find_channel_col(df: pd.DataFrame) -> Optional[str]:
    candidates = [
        "channel",
        "payment_channel",
        "transaction_channel",
        "entry_channel",
        "device_channel",
    ]
    return find_first_existing(df, candidates)


def find_country_col(df: pd.DataFrame) -> Optional[str]:
    candidates = [
        "country",
        "country_code",
        "merchant_country",
        "issuer_country",
        "card_country",
        "customer_country",
        "transaction_country",
    ]
    return find_first_existing(df, candidates)


def find_entity_columns(df: pd.DataFrame) -> Dict[str, Optional[str]]:
    return {
        "Cliente": find_first_existing(
            df,
            ["customer_id", "client_id", "user_id", "account_id", "member_id"]
        ),
        "Comercio": find_first_existing(
            df,
            ["merchant_id", "merchant_name", "seller_id", "vendor_id", "store_id"]
        ),
        "Dispositivo": find_first_existing(
            df,
            ["device_id", "device_fingerprint", "fingerprint", "browser_fingerprint"]
        ),
        "Método de pago": find_first_existing(
            df,
            ["payment_method_id", "payment_instrument_id", "card_id", "payment_token", "payment_method"]
        ),
        "Email": find_first_existing(
            df,
            ["email", "customer_email", "user_email"]
        ),
        "IP": find_first_existing(
            df,
            ["ip_address", "ip", "customer_ip", "device_ip"]
        ),
        "Teléfono": find_first_existing(
            df,
            ["phone", "phone_number", "customer_phone", "mobile_phone"]
        ),
    }


# =========================================================
# CARGA DE DATOS
# =========================================================
@st.cache_data(show_spinner=False)
def load_dashboard_data() -> pd.DataFrame:
    candidate_frames: List[pd.DataFrame] = []

    if load_selected_datasets is not None:
        try:
            loaded = load_selected_datasets()

            if isinstance(loaded, pd.DataFrame):
                candidate_frames.append(normalize_columns(loaded))

            elif isinstance(loaded, dict):
                for _, obj in loaded.items():
                    if isinstance(obj, pd.DataFrame) and not obj.empty:
                        candidate_frames.append(normalize_columns(obj))

            elif isinstance(loaded, (list, tuple)):
                for obj in loaded:
                    if isinstance(obj, pd.DataFrame) and not obj.empty:
                        candidate_frames.append(normalize_columns(obj))
        except Exception:
            pass

    export_dir = ROOT / "artifacts" / "dashboard_exports"
    preferred_files = [
        "dashboard_alert_queue.csv",
        "dashboard_scored_transactions.csv",
        "alert_queue.csv",
        "scored_transactions.csv",
        "transactions_scored.csv",
        "transactions_dashboard.csv",
        "dashboard_transactions.csv",
    ]

    if export_dir.exists():
        for fname in preferred_files:
            p = export_dir / fname
            if p.exists():
                try:
                    df = pd.read_csv(p)
                    if not df.empty:
                        candidate_frames.append(normalize_columns(df))
                except Exception:
                    pass

        if not candidate_frames:
            for p in export_dir.glob("*.csv"):
                try:
                    df = pd.read_csv(p)
                    if not df.empty:
                        candidate_frames.append(normalize_columns(df))
                except Exception:
                    pass

    if not candidate_frames:
        raise FileNotFoundError(
            "No se pudo cargar ningún CSV válido desde artifacts/dashboard_exports "
            "ni mediante load_selected_datasets()."
        )

    scored: List[Tuple[int, pd.DataFrame]] = []
    for df in candidate_frames:
        score = 0
        if find_date_col(df):
            score += 3
        if find_risk_score_col(df):
            score += 4
        if find_risk_bucket_col(df):
            score += 3
        if find_priority_col(df):
            score += 2
        if find_amount_col(df):
            score += 2
        if find_alert_id_col(df):
            score += 1

        entity_cols = find_entity_columns(df)
        score += sum(1 for _, v in entity_cols.items() if v is not None) * 2

        score += min(len(df.columns), 200) / 200.0
        scored.append((score, df))

    scored.sort(key=lambda x: x[0], reverse=True)
    base = scored[0][1].copy()

    return base


# =========================================================
# PREPARACIÓN
# =========================================================
def build_operational_base(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    id_col = find_alert_id_col(out)
    dt_col = find_date_col(out)
    score_col = find_risk_score_col(out)
    bucket_col = find_risk_bucket_col(out)
    priority_col = find_priority_col(out)
    amount_col = find_amount_col(out)
    channel_col = find_channel_col(out)
    country_col = find_country_col(out)
    entity_cols = find_entity_columns(out)

    if id_col is None:
        out["__alert_id__"] = np.arange(1, len(out) + 1)
        id_col = "__alert_id__"

    out["__alert_id_text__"] = out[id_col].astype(str)

    if dt_col is not None:
        out["__event_dt__"] = safe_to_datetime(out[dt_col])
    else:
        out["__event_dt__"] = pd.NaT

    if out["__event_dt__"].notna().sum() > 0:
        out["__event_day__"] = out["__event_dt__"].dt.floor("D")
    else:
        pseudo_start = pd.Timestamp.today().normalize() - pd.Timedelta(days=max(6, min(len(out) // 200, 29)))
        pseudo_dates = pd.date_range(start=pseudo_start, periods=len(out), freq="min")
        out["__event_dt__"] = pseudo_dates
        out["__event_day__"] = out["__event_dt__"].dt.floor("D")

    if score_col is not None:
        out["__risk_score__"] = safe_to_numeric(out[score_col])
    else:
        out["__risk_score__"] = np.nan

    if amount_col is not None:
        out["__amount__"] = safe_to_numeric(out[amount_col]).fillna(0)
    else:
        out["__amount__"] = 0.0

    if channel_col is not None:
        out["__channel__"] = out[channel_col].astype(str).fillna("N/A")
    else:
        out["__channel__"] = "N/A"

    if country_col is not None:
        out["__country__"] = out[country_col].astype(str).fillna("N/A")
    else:
        out["__country__"] = "N/A"

    if bucket_col is not None:
        out["__risk_bucket__"] = out[bucket_col].astype(str).fillna("N/A")
    else:
        if out["__risk_score__"].notna().sum() > 0:
            s = out["__risk_score__"]
            s_max = np.nanmax(s.values) if s.notna().any() else np.nan
            if pd.notna(s_max) and s_max <= 1.5:
                bins = [-0.001, 0.20, 0.40, 0.60, 0.80, 1.00]
                labels = ["Muy bajo", "Bajo", "Medio", "Alto", "Muy alto"]
            else:
                bins = [-0.001, 20, 40, 60, 80, 1000000]
                labels = ["Muy bajo", "Bajo", "Medio", "Alto", "Muy alto"]

            out["__risk_bucket__"] = pd.cut(
                s,
                bins=bins,
                labels=labels,
                include_lowest=True,
            ).astype(str)
        else:
            out["__risk_bucket__"] = "N/A"

    if priority_col is not None:
        pr = out[priority_col].astype(str).str.strip().str.lower()
        out["__priority__"] = np.select(
            [
                pr.str.contains("critical|crítica|critica", na=False),
                pr.str.contains("high|alta", na=False),
                pr.str.contains("medium|media", na=False),
                pr.str.contains("low|baja", na=False),
            ],
            ["Crítica", "Alta", "Media", "Baja"],
            default=out["__risk_bucket__"].astype(str),
        )
    else:
        bucket_text = out["__risk_bucket__"].astype(str).str.lower()
        out["__priority__"] = np.select(
            [
                bucket_text.str.contains("muy alto", na=False),
                bucket_text.str.contains("alto", na=False),
                bucket_text.str.contains("medio", na=False),
                bucket_text.str.contains("bajo|muy bajo", na=False),
            ],
            ["Crítica", "Alta", "Media", "Baja"],
            default="Media",
        )

    # columnas canónicas de entidades
    for label, col in entity_cols.items():
        canon = f"__entity_{label.lower().replace(' ', '_').replace('é','e').replace('ó','o')}__"
        if col is not None:
            out[canon] = out[col].astype(str).fillna("N/A")
        else:
            out[canon] = np.nan

    return out


# =========================================================
# FILTROS
# =========================================================
def build_filters(df: pd.DataFrame) -> pd.DataFrame:
    st.sidebar.header("Filtros")

    date_min = pd.to_datetime(df["__event_day__"], errors="coerce").min()
    date_max = pd.to_datetime(df["__event_day__"], errors="coerce").max()

    selected_range = st.sidebar.date_input(
        "Rango temporal",
        value=(date_min.date(), date_max.date()) if pd.notna(date_min) and pd.notna(date_max) else None,
    )

    filtered = df.copy()

    if isinstance(selected_range, tuple) and len(selected_range) == 2:
        start_date = pd.Timestamp(selected_range[0])
        end_date = pd.Timestamp(selected_range[1])
        filtered = filtered[
            (filtered["__event_day__"] >= start_date) &
            (filtered["__event_day__"] <= end_date)
        ]

    priority_order = ["Crítica", "Alta", "Media", "Baja"]
    priorities_available = [p for p in priority_order if p in filtered["__priority__"].astype(str).unique().tolist()]
    priorities_selected = st.sidebar.multiselect(
        "Prioridad",
        options=priorities_available if priorities_available else sorted(filtered["__priority__"].astype(str).unique().tolist()),
        default=priorities_available if priorities_available else sorted(filtered["__priority__"].astype(str).unique().tolist()),
    )
    if priorities_selected:
        filtered = filtered[filtered["__priority__"].isin(priorities_selected)]

    buckets_available = sorted(filtered["__risk_bucket__"].astype(str).dropna().unique().tolist())
    buckets_selected = st.sidebar.multiselect(
        "Risk bucket",
        options=buckets_available,
        default=buckets_available,
    )
    if buckets_selected:
        filtered = filtered[filtered["__risk_bucket__"].isin(buckets_selected)]

    channel_available = sorted(filtered["__channel__"].astype(str).dropna().unique().tolist())
    if len(channel_available) > 1:
        channels_selected = st.sidebar.multiselect(
            "Canal",
            options=channel_available,
            default=channel_available,
        )
        if channels_selected:
            filtered = filtered[filtered["__channel__"].isin(channels_selected)]

    country_available = sorted(filtered["__country__"].astype(str).dropna().unique().tolist())
    if len(country_available) > 1 and "N/A" not in country_available[:1]:
        countries_selected = st.sidebar.multiselect(
            "País",
            options=country_available,
            default=country_available,
        )
        if countries_selected:
            filtered = filtered[filtered["__country__"].isin(countries_selected)]

    return filtered


# =========================================================
# MÉTRICAS DE ENTIDADES
# =========================================================
def available_entity_options(df: pd.DataFrame) -> Dict[str, str]:
    options = {}
    possible = {
        "Cliente": "__entity_cliente__",
        "Comercio": "__entity_comercio__",
        "Dispositivo": "__entity_dispositivo__",
        "Método de pago": "__entity_metodo_de_pago__",
        "Email": "__entity_email__",
        "IP": "__entity_ip__",
        "Teléfono": "__entity_telefono__",
    }
    for label, col in possible.items():
        if col in df.columns and df[col].notna().sum() > 0:
            non_na = df[col].dropna().astype(str)
            valid = non_na[~non_na.isin(["N/A", "nan", "None", ""])]
            if len(valid) > 0:
                options[label] = col
    return options


def build_entity_summary(
    df: pd.DataFrame,
    entity_col: str,
    min_alerts: int = 2,
) -> pd.DataFrame:
    base_n = max(len(df), 1)
    base_amount = float(df["__amount__"].sum())
    base_avg_score = float(df["__risk_score__"].mean()) if df["__risk_score__"].notna().sum() > 0 else np.nan

    tmp = df.copy()
    tmp = tmp[tmp[entity_col].notna()]
    tmp = tmp[~tmp[entity_col].astype(str).isin(["N/A", "nan", "None", ""])]

    if tmp.empty:
        return pd.DataFrame()

    def top_mode(series: pd.Series) -> str:
        s = series.dropna().astype(str)
        if s.empty:
            return "N/A"
        mode = s.mode()
        return str(mode.iloc[0]) if not mode.empty else "N/A"

    agg = (
        tmp.groupby(entity_col, dropna=False)
        .agg(
            alertas=("__alert_id_text__", "count"),
            importe_total=("__amount__", "sum"),
            score_medio=("__risk_score__", "mean"),
            score_max=("__risk_score__", "max"),
            prioridad_dominante=("__priority__", top_mode),
            bucket_dominante=("__risk_bucket__", top_mode),
            canal_dominante=("__channel__", top_mode),
            pais_dominante=("__country__", top_mode),
            fecha_primera=("__event_day__", "min"),
            fecha_ultima=("__event_day__", "max"),
            dias_activos=("__event_day__", lambda s: s.nunique()),
        )
        .reset_index()
        .rename(columns={entity_col: "entidad"})
    )

    agg = agg[agg["alertas"] >= min_alerts].copy()
    if agg.empty:
        return agg

    agg["importe_total"] = agg["importe_total"].fillna(0)
    agg["score_medio"] = agg["score_medio"].fillna(0)
    agg["score_max"] = agg["score_max"].fillna(0)
    agg["dias_activos"] = agg["dias_activos"].fillna(0)

    agg["share_alertas_pct"] = 100 * agg["alertas"] / base_n
    agg["share_importe_pct"] = 100 * agg["importe_total"] / max(base_amount, 1)

    # score relativo
    if pd.notna(base_avg_score) and base_avg_score > 0:
        agg["score_relativo"] = agg["score_medio"] / base_avg_score
    else:
        agg["score_relativo"] = 1.0

    # intensidad
    agg["alertas_por_dia_activo"] = agg["alertas"] / np.maximum(agg["dias_activos"], 1)

    # score de sospecha compuesto y estable
    def robust_z(s: pd.Series) -> pd.Series:
        if s.nunique(dropna=True) <= 1:
            return pd.Series(np.zeros(len(s)), index=s.index)
        std = s.std(ddof=0)
        if pd.isna(std) or std == 0:
            return pd.Series(np.zeros(len(s)), index=s.index)
        return (s - s.mean()) / std

    z_alertas = robust_z(agg["alertas"].astype(float))
    z_importe = robust_z(agg["importe_total"].astype(float))
    z_score = robust_z(agg["score_medio"].astype(float))
    z_intensidad = robust_z(agg["alertas_por_dia_activo"].astype(float))

    agg["suspicion_score"] = (
        0.35 * z_alertas +
        0.25 * z_importe +
        0.25 * z_score +
        0.15 * z_intensidad
    )

    if agg["suspicion_score"].nunique(dropna=True) <= 1:
        agg["suspicion_score_norm"] = 50.0
    else:
        smin = agg["suspicion_score"].min()
        smax = agg["suspicion_score"].max()
        agg["suspicion_score_norm"] = 100 * (agg["suspicion_score"] - smin) / max(smax - smin, 1e-9)

    agg["flag_concentracion_alertas"] = agg["share_alertas_pct"] >= 5
    agg["flag_concentracion_importe"] = agg["share_importe_pct"] >= 5
    agg["flag_score_alto"] = agg["score_relativo"] >= 1.25
    agg["flag_intensidad"] = agg["alertas_por_dia_activo"] >= agg["alertas_por_dia_activo"].quantile(0.75)

    agg["motivo_operativo"] = (
        np.where(agg["flag_concentracion_alertas"], "Alta concentración de alertas; ", "") +
        np.where(agg["flag_concentracion_importe"], "Alta concentración de importe; ", "") +
        np.where(agg["flag_score_alto"], "Score medio superior al baseline; ", "") +
        np.where(agg["flag_intensidad"], "Alta recurrencia por día activo; ", "")
    ).str.strip()

    agg["motivo_operativo"] = agg["motivo_operativo"].replace("", "Patrón persistente a monitorizar")

    agg = agg.sort_values(
        by=["suspicion_score_norm", "alertas", "importe_total", "score_medio"],
        ascending=[False, False, False, False],
    ).reset_index(drop=True)

    return agg


def build_entity_timeseries(df: pd.DataFrame, entity_col: str, entity_value: str) -> pd.DataFrame:
    tmp = df[df[entity_col].astype(str) == str(entity_value)].copy()
    if tmp.empty:
        return pd.DataFrame(columns=["__event_day__", "alertas", "importe_total", "score_medio"])

    out = (
        tmp.groupby("__event_day__", dropna=False)
        .agg(
            alertas=("__alert_id_text__", "count"),
            importe_total=("__amount__", "sum"),
            score_medio=("__risk_score__", "mean"),
        )
        .reset_index()
        .sort_values("__event_day__")
    )
    return out


def build_cooccurrence_table(
    df: pd.DataFrame,
    selected_entity_label: str,
    selected_entity_col: str,
    selected_entity_value: str,
) -> pd.DataFrame:
    tmp = df[df[selected_entity_col].astype(str) == str(selected_entity_value)].copy()
    if tmp.empty:
        return pd.DataFrame()

    mapping = {
        "Cliente": "__entity_cliente__",
        "Comercio": "__entity_comercio__",
        "Dispositivo": "__entity_dispositivo__",
        "Método de pago": "__entity_metodo_de_pago__",
        "Email": "__entity_email__",
        "IP": "__entity_ip__",
        "Teléfono": "__entity_telefono__",
    }

    rows = []
    for label, col in mapping.items():
        if label == selected_entity_label:
            continue
        if col not in tmp.columns:
            continue

        s = tmp[col].dropna().astype(str)
        s = s[~s.isin(["N/A", "nan", "None", ""])]
        if s.empty:
            continue

        vc = s.value_counts().head(5)
        for val, cnt in vc.items():
            rows.append(
                {
                    "Tipo relacionado": label,
                    "Valor relacionado": val,
                    "Alertas compartidas": int(cnt),
                }
            )

    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values("Alertas compartidas", ascending=False).reset_index(drop=True)
    return out


def build_entity_readout(summary_df: pd.DataFrame) -> List[str]:
    if summary_df.empty:
        return ["No hay suficiente volumen para identificar entidades con patrón sospechoso bajo los filtros actuales."]

    top = summary_df.iloc[0]

    bullets = [
        f"La entidad líder concentra {fmt_int(top['alertas'])} alertas y {fmt_money(top['importe_total'])}, con un suspicion score de {fmt_num(top['suspicion_score_norm'], 1)} sobre 100.",
    ]

    high_concentration = summary_df[summary_df["share_alertas_pct"] >= 5]
    if not high_concentration.empty:
        bullets.append(
            f"Se detectan {fmt_int(len(high_concentration))} entidades con concentración material de alertas, señal de recurrencia operativa."
        )

    high_amount = summary_df[summary_df["share_importe_pct"] >= 5]
    if not high_amount.empty:
        bullets.append(
            f"{fmt_int(len(high_amount))} entidades concentran una parte relevante del importe expuesto, por lo que el riesgo no es solo de frecuencia sino también económico."
        )

    return bullets[:3]


# =========================================================
# PLOTS
# =========================================================
def plot_top_entities_bubble(df: pd.DataFrame, top_n: int) -> go.Figure:
    tmp = df.head(top_n).copy()
    if tmp.empty:
        return go.Figure()

    fig = px.scatter(
        tmp,
        x="alertas",
        y="importe_total",
        size="suspicion_score_norm",
        color="score_medio",
        hover_name="entidad",
        hover_data={
            "share_alertas_pct": ":.1f",
            "share_importe_pct": ":.1f",
            "alertas_por_dia_activo": ":.2f",
        },
    )
    fig.update_layout(
        height=420,
        margin=dict(l=20, r=20, t=20, b=20),
        xaxis_title="Número de alertas",
        yaxis_title="Importe total",
    )
    return fig


def plot_top_entities_bar(df: pd.DataFrame, top_n: int) -> go.Figure:
    tmp = df.head(top_n).copy()
    if tmp.empty:
        return go.Figure()

    tmp = tmp.sort_values("suspicion_score_norm", ascending=True)

    fig = px.bar(
        tmp,
        x="suspicion_score_norm",
        y="entidad",
        orientation="h",
        text="suspicion_score_norm",
        hover_data=["alertas", "importe_total", "score_medio", "motivo_operativo"],
    )
    fig.update_traces(texttemplate="%{text:.1f}", textposition="outside")
    fig.update_layout(
        height=max(380, top_n * 26),
        margin=dict(l=20, r=20, t=20, b=20),
        xaxis_title="Suspicion score (0-100)",
        yaxis_title="Entidad",
        showlegend=False,
    )
    return fig


def plot_priority_mix_for_top_entities(df: pd.DataFrame, top_n: int) -> go.Figure:
    tmp = df.head(top_n).copy()
    if tmp.empty:
        return go.Figure()

    order = ["Crítica", "Alta", "Media", "Baja"]
    melted = []
    for _, row in tmp.iterrows():
        for p in order:
            cnt = row.get(f"prio_{p}", 0)
            melted.append(
                {
                    "entidad": row["entidad"],
                    "Prioridad": p,
                    "Alertas": cnt,
                }
            )
    mix = pd.DataFrame(melted)

    fig = px.bar(
        mix,
        x="entidad",
        y="Alertas",
        color="Prioridad",
        barmode="stack",
    )
    fig.update_layout(
        height=400,
        margin=dict(l=20, r=20, t=20, b=20),
        xaxis_title="Entidad",
        yaxis_title="Alertas",
    )
    return fig


def plot_entity_timeline(ts_df: pd.DataFrame) -> go.Figure:
    if ts_df.empty:
        return go.Figure()

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=ts_df["__event_day__"],
            y=ts_df["alertas"],
            mode="lines+markers",
            name="Alertas",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=ts_df["__event_day__"],
            y=ts_df["score_medio"],
            mode="lines+markers",
            name="Score medio",
            yaxis="y2",
        )
    )

    fig.update_layout(
        height=400,
        margin=dict(l=20, r=20, t=20, b=20),
        xaxis_title="Fecha",
        yaxis=dict(title="Alertas"),
        yaxis2=dict(
            title="Score medio",
            overlaying="y",
            side="right",
            showgrid=False,
        ),
        legend=dict(orientation="h"),
    )
    return fig


def plot_bucket_distribution_for_entity(df: pd.DataFrame, entity_col: str, entity_value: str) -> go.Figure:
    tmp = df[df[entity_col].astype(str) == str(entity_value)].copy()
    if tmp.empty:
        return go.Figure()

    order = ["Muy alto", "Alto", "Medio", "Bajo", "Muy bajo", "N/A"]
    dist = (
        tmp.groupby("__risk_bucket__", dropna=False)
        .size()
        .reset_index(name="alertas")
    )
    dist["__risk_bucket__"] = pd.Categorical(dist["__risk_bucket__"], categories=order, ordered=True)
    dist = dist.sort_values("__risk_bucket__")

    fig = px.bar(
        dist,
        x="__risk_bucket__",
        y="alertas",
        text="alertas",
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(
        height=360,
        margin=dict(l=20, r=20, t=20, b=20),
        xaxis_title="Risk bucket",
        yaxis_title="Alertas",
    )
    return fig


# =========================================================
# APP
# =========================================================
st.title("Suspicious Entities")
st.caption("Detección de entidades con concentración anómala de alertas, importe o score dentro de la operación antifraude.")

st.markdown(
    """
    <div class="se-info-box">
        <div class="se-block-help">
            Esta página identifica <b>entidades recurrentes o concentradas</b> detrás de la cola de alertas:
            clientes, comercios, dispositivos, métodos de pago u otros identificadores operativos.
            La idea no es decidir qué alerta revisar primero, sino detectar <b>qué entidades están generando presión o riesgo de forma persistente</b>.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# =========================================================
# DATA
# =========================================================
try:
    raw_df = load_dashboard_data()
    df = build_operational_base(raw_df)
except Exception as e:
    st.error("No se pudo cargar la base operativa para Suspicious Entities.")
    st.exception(e)
    st.stop()

filtered_df = build_filters(df)
if filtered_df.empty:
    st.warning("No hay datos con los filtros seleccionados.")
    st.stop()

entity_options = available_entity_options(filtered_df)
if not entity_options:
    st.warning("El dataset actual no incluye dimensiones de entidad utilizables para esta vista.")
    st.stop()

# =========================================================
# CONFIG DE ENTIDAD
# =========================================================
st.subheader("Configuración de análisis")
st.caption("Selecciona la dimensión operativa sobre la que quieres detectar concentración de riesgo.")

c1, c2, c3 = st.columns(3)
with c1:
    entity_label = st.selectbox("Tipo de entidad", list(entity_options.keys()), index=0)
with c2:
    min_alerts = st.slider("Mínimo de alertas por entidad", min_value=1, max_value=20, value=2, step=1)
with c3:
    top_n = st.slider("Top entidades a mostrar", min_value=5, max_value=50, value=15, step=1)

entity_col = entity_options[entity_label]
summary_df = build_entity_summary(filtered_df, entity_col=entity_col, min_alerts=min_alerts)

if summary_df.empty:
    st.warning("No hay entidades con suficiente volumen bajo los filtros actuales.")
    st.stop()

# Añadir mix de prioridad para visuales
priority_mix = (
    filtered_df.groupby([entity_col, "__priority__"], dropna=False)
    .size()
    .reset_index(name="n")
)
priority_pivot = priority_mix.pivot(index=entity_col, columns="__priority__", values="n").fillna(0).reset_index()
rename_map = {c: f"prio_{c}" for c in priority_pivot.columns if c != entity_col}
priority_pivot = priority_pivot.rename(columns=rename_map)

summary_df = summary_df.merge(
    priority_pivot.rename(columns={entity_col: "entidad"}),
    on="entidad",
    how="left",
)
for col in ["prio_Crítica", "prio_Alta", "prio_Media", "prio_Baja"]:
    if col not in summary_df.columns:
        summary_df[col] = 0

# =========================================================
# KPIS
# =========================================================
total_entities = summary_df["entidad"].nunique()
top1 = summary_df.iloc[0]
share_top1_alertas = float(top1["share_alertas_pct"])
share_top1_importe = float(top1["share_importe_pct"])
high_score_entities = int((summary_df["score_relativo"] >= 1.25).sum())
material_entities = int((summary_df["share_alertas_pct"] >= 5).sum())

st.subheader("Resumen de concentración")
st.caption("Mide si el riesgo está distribuido o si se concentra en un grupo reducido de entidades sospechosas.")

k1, k2, k3, k4, k5 = st.columns(5)
with k1:
    st.metric("Entidades analizadas", fmt_int(total_entities))
with k2:
    st.metric("Top entidad por alertas", str(top1["entidad"])[:28] + ("..." if len(str(top1["entidad"])) > 28 else ""))
with k3:
    st.metric("Concentración top 1 (alertas)", fmt_pct(share_top1_alertas))
with k4:
    st.metric("Concentración top 1 (importe)", fmt_pct(share_top1_importe))
with k5:
    st.metric("Entidades con score elevado", fmt_int(high_score_entities))

st.markdown(
    """
    <div class="se-kpi-note">
        Lectura rápida: si pocas entidades concentran alertas, importe o score, la operación puede estar ante focos claros de riesgo recurrente.
    </div>
    """,
    unsafe_allow_html=True,
)

# =========================================================
# LECTURA EJECUTIVA
# =========================================================
st.subheader("Lectura ejecutiva / operativa")
st.caption("Resumen breve para identificar si existe concentración de riesgo en un conjunto reducido de entidades.")

for bullet in build_entity_readout(summary_df):
    st.markdown(f"- {bullet}")

# =========================================================
# VISUALES PRINCIPALES
# =========================================================
st.subheader("Mapa de entidades sospechosas")
st.caption("Combina frecuencia, importe y score para localizar entidades que merecen investigación específica.")

c1, c2 = st.columns(2)

with c1:
    st.markdown("**Frecuencia vs importe**")
    st.markdown(
        "<div class='se-block-help'>Las burbujas más grandes combinan alta recurrencia, mayor exposición económica y mayor nivel de sospecha compuesta.</div>",
        unsafe_allow_html=True,
    )
    st.plotly_chart(plot_top_entities_bubble(summary_df, top_n), use_container_width=True)

with c2:
    st.markdown("**Ranking de suspicion score**")
    st.markdown(
        "<div class='se-block-help'>Prioriza las entidades que concentran un patrón más anómalo respecto al resto de la población filtrada.</div>",
        unsafe_allow_html=True,
    )
    st.plotly_chart(plot_top_entities_bar(summary_df, top_n), use_container_width=True)

# =========================================================
# CONCENTRACIÓN Y PRIORIDAD
# =========================================================
st.subheader("Concentración operativa")
st.caption("Permite ver si las entidades más sospechosas también están generando la parte alta de la cola.")

st.markdown(
    "<div class='se-block-help'>Si las entidades top concentran prioridad crítica o alta, conviene tratarlas como focos operativos persistentes y no solo como casos aislados.</div>",
    unsafe_allow_html=True,
)
st.plotly_chart(plot_priority_mix_for_top_entities(summary_df, min(top_n, 10)), use_container_width=True)

# =========================================================
# TABLA PRINCIPAL
# =========================================================
st.subheader("Tabla de entidades")
st.caption("Detalle analítico para inspección, priorización y posible apertura de investigación.")

display_df = summary_df.copy()
display_df = display_df.rename(
    columns={
        "entidad": "Entidad",
        "alertas": "Alertas",
        "importe_total": "Importe total",
        "score_medio": "Score medio",
        "score_max": "Score máximo",
        "share_alertas_pct": "% alertas",
        "share_importe_pct": "% importe",
        "alertas_por_dia_activo": "Alertas / día activo",
        "prioridad_dominante": "Prioridad dominante",
        "bucket_dominante": "Bucket dominante",
        "canal_dominante": "Canal dominante",
        "pais_dominante": "País dominante",
        "dias_activos": "Días activos",
        "fecha_primera": "Primera fecha",
        "fecha_ultima": "Última fecha",
        "suspicion_score_norm": "Suspicion score",
        "motivo_operativo": "Motivo operativo",
    }
)

for col in ["Importe total"]:
    display_df[col] = display_df[col].apply(fmt_money)

for col in ["Score medio", "Score máximo", "Alertas / día activo", "Suspicion score"]:
    display_df[col] = display_df[col].apply(lambda x: fmt_num(x, 2))

for col in ["% alertas", "% importe"]:
    display_df[col] = display_df[col].apply(lambda x: fmt_pct(x, 1))

for col in ["Primera fecha", "Última fecha"]:
    display_df[col] = pd.to_datetime(display_df[col], errors="coerce").dt.strftime("%Y-%m-%d")

cols_to_show = [
    "Entidad",
    "Suspicion score",
    "Alertas",
    "% alertas",
    "Importe total",
    "% importe",
    "Score medio",
    "Score máximo",
    "Alertas / día activo",
    "Prioridad dominante",
    "Bucket dominante",
    "Canal dominante",
    "País dominante",
    "Días activos",
    "Primera fecha",
    "Última fecha",
    "Motivo operativo",
]
cols_to_show = [c for c in cols_to_show if c in display_df.columns]

st.dataframe(
    display_df[cols_to_show].head(500),
    use_container_width=True,
    hide_index=True,
)

# =========================================================
# EXPLORADOR DE ENTIDAD
# =========================================================
st.subheader("Explorador de entidad")
st.caption("Permite abrir una entidad concreta para ver su evolución temporal, su distribución de riesgo y sus coocurrencias operativas.")

entity_values = summary_df["entidad"].astype(str).tolist()
default_idx = 0
selected_entity_value = st.selectbox(
    f"Selecciona una {entity_label.lower()}",
    options=entity_values,
    index=default_idx,
)

selected_row = summary_df[summary_df["entidad"].astype(str) == str(selected_entity_value)].iloc[0]
entity_ts = build_entity_timeseries(filtered_df, entity_col, selected_entity_value)
entity_links = build_cooccurrence_table(filtered_df, entity_label, entity_col, selected_entity_value)

c1, c2, c3, c4 = st.columns(4)
with c1:
    st.metric("Alertas", fmt_int(selected_row["alertas"]))
with c2:
    st.metric("Importe total", fmt_money(selected_row["importe_total"]))
with c3:
    st.metric("Score medio", fmt_num(selected_row["score_medio"], 2))
with c4:
    st.metric("Suspicion score", fmt_num(selected_row["suspicion_score_norm"], 1))

st.markdown(
    f"""
    **Motivo operativo:** {selected_row["motivo_operativo"]}  
    **Prioridad dominante:** {selected_row["prioridad_dominante"]} ·
    **Bucket dominante:** {selected_row["bucket_dominante"]} ·
    **Canal dominante:** {selected_row["canal_dominante"]} ·
    **País dominante:** {selected_row["pais_dominante"]}
    """
)

c1, c2 = st.columns(2)

with c1:
    st.markdown("**Evolución temporal de la entidad**")
    st.markdown(
        "<div class='se-block-help'>Sirve para ver si la entidad tiene un patrón puntual o una recurrencia sostenida en el tiempo.</div>",
        unsafe_allow_html=True,
    )
    st.plotly_chart(plot_entity_timeline(entity_ts), use_container_width=True)

with c2:
    st.markdown("**Distribución por risk bucket**")
    st.markdown(
        "<div class='se-block-help'>Ayuda a entender si la entidad aparece sobre todo en buckets altos o si su señal es más difusa.</div>",
        unsafe_allow_html=True,
    )
    st.plotly_chart(
        plot_bucket_distribution_for_entity(filtered_df, entity_col, selected_entity_value),
        use_container_width=True,
    )

# =========================================================
# COOCURRENCIAS
# =========================================================
st.subheader("Coocurrencias operativas")
st.caption("Muestra otros identificadores que aparecen repetidamente junto a la entidad seleccionada.")

if entity_links.empty:
    st.info("No se detectaron coocurrencias relevantes para la entidad seleccionada.")
else:
    st.dataframe(
        entity_links.head(50),
        use_container_width=True,
        hide_index=True,
    )

# =========================================================
# CUELLOS / HALLAZGOS
# =========================================================
st.subheader("Hallazgos operativos")
st.caption("Conclusiones rápidas sobre concentración, persistencia y posible foco de investigación.")

top5_alert_share = summary_df.head(5)["share_alertas_pct"].sum()
top5_amount_share = summary_df.head(5)["share_importe_pct"].sum()

bullets = [
    f"Las 5 entidades más sospechosas concentran aproximadamente el {fmt_pct(top5_alert_share, 1)} de las alertas filtradas.",
    f"Esas mismas 5 entidades acumulan alrededor del {fmt_pct(top5_amount_share, 1)} del importe total observado.",
    f"Se identifican {fmt_int(material_entities)} entidades con concentración material de alertas, útiles para priorizar investigación o reglas de contención.",
]

for b in bullets:
    st.markdown(f"- {b}")

# =========================================================
# NOTA FINAL
# =========================================================
st.markdown("---")
st.markdown(
    """
    **Cómo interpretar esta página**
    
    - **Alta concentración en pocas entidades**: posible foco recurrente de fraude o abuso operativo.  
    - **Alta frecuencia + alto importe + score elevado**: entidad candidata a investigación prioritaria.  
    - **Recurrencia sostenida en el tiempo**: patrón más estructural que incidente puntual.  
    - **Coocurrencias repetidas**: posible red operativa o reutilización de infraestructura fraudulenta.  
    """
)
