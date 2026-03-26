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
    page_title="Review Capacity | SentinelFlow",
    page_icon="🧠",
    layout="wide",
)

# Intentar usar loader común del proyecto si existe
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
    .rc-info-box {
        background: linear-gradient(135deg, rgba(17,24,39,0.98), rgba(31,41,55,0.96));
        border: 1px solid rgba(148,163,184,0.20);
        border-radius: 16px;
        padding: 18px 20px;
        margin-bottom: 14px;
    }
    .rc-kpi-note {
        font-size: 0.88rem;
        color: #94a3b8;
        margin-top: -6px;
    }
    .rc-section-caption {
        color: #94a3b8;
        font-size: 0.93rem;
        margin-top: -8px;
        margin-bottom: 12px;
    }
    .rc-block-help {
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


def fmt_num(x: float | int | None, digits: int = 1) -> str:
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


def find_date_col(df: pd.DataFrame) -> Optional[str]:
    candidates = [
        "alert_date",
        "alert_ts",
        "alert_timestamp",
        "review_date",
        "queue_date",
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


def safe_to_datetime(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce")


def safe_to_numeric(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


# =========================================================
# CARGA DE DATOS
# =========================================================
@st.cache_data(show_spinner=False)
def load_dashboard_data() -> pd.DataFrame:
    """
    Carga robusta de la base operativa principal del dashboard.
    Prioriza loader común del proyecto y, si falla, busca CSVs en artifacts/dashboard_exports.
    """
    candidate_frames: List[pd.DataFrame] = []

    # 1) Intento mediante loader del proyecto
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

    # 2) Fallback: buscar CSVs directamente
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

    # Elegir la base más útil: priorizar la que tenga score/bucket/fecha
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
        score += min(len(df.columns), 200) / 200.0
        scored.append((score, df))

    scored.sort(key=lambda x: x[0], reverse=True)
    base = scored[0][1].copy()

    return base


# =========================================================
# PREPARACIÓN DE BASE OPERATIVA
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

    if id_col is None:
        out["__alert_id__"] = np.arange(1, len(out) + 1)
        id_col = "__alert_id__"

    if dt_col is not None:
        out["__event_dt__"] = safe_to_datetime(out[dt_col])
    else:
        out["__event_dt__"] = pd.NaT

    if score_col is not None:
        out["__risk_score__"] = safe_to_numeric(out[score_col])
    else:
        out["__risk_score__"] = np.nan

    if amount_col is not None:
        out["__amount__"] = safe_to_numeric(out[amount_col]).fillna(0)
    else:
        out["__amount__"] = 0.0

    if channel_col is not None:
        out["__channel__"] = out[channel_col].astype(str)
    else:
        out["__channel__"] = "N/A"

    # Bucket de riesgo
    if bucket_col is not None:
        out["__risk_bucket__"] = out[bucket_col].astype(str).fillna("N/A")
    else:
        # Crear bucket a partir de score si existe
        if out["__risk_score__"].notna().sum() > 0:
            s = out["__risk_score__"]
            # robusto ante escalas 0-1 o 0-100
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

    # Prioridad operativa
    if priority_col is not None:
        pr = out[priority_col].astype(str).str.strip().str.lower()
        mapped = np.select(
            [
                pr.str.contains("critical|crítica|critica", na=False),
                pr.str.contains("high|alta", na=False),
                pr.str.contains("medium|media", na=False),
                pr.str.contains("low|baja", na=False),
            ],
            [
                "Crítica",
                "Alta",
                "Media",
                "Baja",
            ],
            default=out["__risk_bucket__"].astype(str),
        )
        out["__priority__"] = mapped
    else:
        # Derivar prioridad desde bucket
        bucket_text = out["__risk_bucket__"].astype(str).str.lower()
        out["__priority__"] = np.select(
            [
                bucket_text.str.contains("muy alto", na=False),
                bucket_text.str.contains("alto", na=False),
                bucket_text.str.contains("medio", na=False),
                bucket_text.str.contains("bajo|muy bajo", na=False),
            ],
            [
                "Crítica",
                "Alta",
                "Media",
                "Baja",
            ],
            default="Media",
        )

    # Fecha/día
    if out["__event_dt__"].notna().sum() > 0:
        out["__event_day__"] = out["__event_dt__"].dt.floor("D")
        out["__weekday__"] = out["__event_dt__"].dt.day_name()
    else:
        # Si no hay fecha, crear una pseudo-fecha para que la página no rompa
        pseudo_start = pd.Timestamp.today().normalize() - pd.Timedelta(days=max(6, min(len(out) // 200, 29)))
        pseudo_dates = pd.date_range(start=pseudo_start, periods=len(out), freq="min")
        out["__event_dt__"] = pseudo_dates
        out["__event_day__"] = out["__event_dt__"].dt.floor("D")
        out["__weekday__"] = out["__event_dt__"].dt.day_name()

    out["__alert_id_text__"] = out[id_col].astype(str)

    return out


# =========================================================
# FILTROS
# =========================================================
def build_filters(df: pd.DataFrame) -> pd.DataFrame:
    st.sidebar.header("Filtros operativos")

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

    return filtered


# =========================================================
# CÁLCULO DE DEMANDA / CAPACIDAD
# =========================================================
def get_volume_metrics(df: pd.DataFrame) -> Dict[str, float]:
    if df.empty:
        return {
            "n_alerts": 0,
            "n_days": 0,
            "alerts_per_day": 0.0,
            "alerts_per_week": 0.0,
            "total_amount": 0.0,
            "amount_per_day": 0.0,
        }

    n_alerts = len(df)
    total_amount = float(df["__amount__"].fillna(0).sum())

    days = pd.Series(df["__event_day__"].dropna().unique())
    n_days = int(max(len(days), 1))

    alerts_per_day = n_alerts / n_days
    alerts_per_week = alerts_per_day * 5
    amount_per_day = total_amount / n_days

    return {
        "n_alerts": float(n_alerts),
        "n_days": float(n_days),
        "alerts_per_day": float(alerts_per_day),
        "alerts_per_week": float(alerts_per_week),
        "total_amount": float(total_amount),
        "amount_per_day": float(amount_per_day),
    }


def simulate_capacity(
    df: pd.DataFrame,
    n_analysts: int,
    alerts_per_analyst_day: int,
    workdays_per_week: int,
    initial_backlog: int,
    horizon_mode: str,
) -> Dict[str, float]:
    volume = get_volume_metrics(df)

    incoming_per_day = volume["alerts_per_day"]
    incoming_per_week = incoming_per_day * workdays_per_week

    capacity_per_day = n_analysts * alerts_per_analyst_day
    capacity_per_week = capacity_per_day * workdays_per_week

    if horizon_mode == "Diario":
        incoming = incoming_per_day
        capacity = capacity_per_day
    else:
        incoming = incoming_per_week
        capacity = capacity_per_week

    reviewable = min(incoming + initial_backlog, capacity)
    uncovered = max((incoming + initial_backlog) - capacity, 0)
    backlog_increment = max(incoming - capacity, 0)
    coverage_rate = 100 * reviewable / max(incoming + initial_backlog, 1)

    amount_total = volume["total_amount"]
    avg_amount = amount_total / max(volume["n_alerts"], 1)
    uncovered_amount = uncovered * avg_amount

    # Drenaje solo tiene sentido si la capacidad supera la entrada estructural
    net_drain_per_period = capacity - incoming
    if initial_backlog <= 0:
        drain_periods = 0.0
    elif net_drain_per_period > 0:
        drain_periods = initial_backlog / net_drain_per_period
    else:
        drain_periods = math.inf

    return {
        "incoming": float(incoming),
        "capacity": float(capacity),
        "reviewable": float(reviewable),
        "uncovered": float(uncovered),
        "coverage_rate": float(coverage_rate),
        "backlog_increment": float(backlog_increment),
        "uncovered_amount": float(uncovered_amount),
        "drain_periods": float(drain_periods) if math.isfinite(drain_periods) else math.inf,
        "incoming_per_day": float(incoming_per_day),
        "incoming_per_week": float(incoming_per_week),
        "capacity_per_day": float(capacity_per_day),
        "capacity_per_week": float(capacity_per_week),
        "avg_amount": float(avg_amount),
    }


def allocate_reviewable_vs_uncovered(
    df: pd.DataFrame,
    reviewable_count: int,
) -> pd.DataFrame:
    """
    Separa la cola en parte revisable y parte no revisable.
    Orden de prioridad:
    1) prioridad operativa
    2) risk score descendente
    3) importe descendente
    """
    if df.empty:
        return pd.DataFrame(columns=list(df.columns) + ["__review_status__"])

    priority_rank = {
        "Crítica": 4,
        "Alta": 3,
        "Media": 2,
        "Baja": 1,
    }

    tmp = df.copy()
    tmp["__priority_rank__"] = tmp["__priority__"].map(priority_rank).fillna(0)
    tmp["__score_sort__"] = tmp["__risk_score__"].fillna(-1)
    tmp["__amount_sort__"] = tmp["__amount__"].fillna(0)

    tmp = tmp.sort_values(
        by=["__priority_rank__", "__score_sort__", "__amount_sort__"],
        ascending=[False, False, False],
    ).reset_index(drop=True)

    reviewable_count = int(max(0, min(reviewable_count, len(tmp))))
    tmp["__review_status__"] = "Fuera de capacidad"
    if reviewable_count > 0:
        tmp.loc[: reviewable_count - 1, "__review_status__"] = "Revisable"

    return tmp


def build_scenario_table(
    df: pd.DataFrame,
    base_analysts: int,
    alerts_per_analyst_day: int,
    workdays_per_week: int,
    initial_backlog: int,
    horizon_mode: str,
) -> pd.DataFrame:
    scenarios = [
        ("Infra-capacidad", max(1, base_analysts - 2), alerts_per_analyst_day),
        ("Equipo actual", base_analysts, alerts_per_analyst_day),
        ("Refuerzo moderado", base_analysts + 2, alerts_per_analyst_day),
        ("Refuerzo alto", base_analysts + 4, alerts_per_analyst_day),
        ("Mejora productividad", base_analysts, alerts_per_analyst_day + 5),
    ]

    rows = []
    for name, analysts, throughput in scenarios:
        sim = simulate_capacity(
            df=df,
            n_analysts=analysts,
            alerts_per_analyst_day=throughput,
            workdays_per_week=workdays_per_week,
            initial_backlog=initial_backlog,
            horizon_mode=horizon_mode,
        )
        rows.append(
            {
                "Escenario": name,
                "Analistas": analysts,
                "Alertas por analista/día": throughput,
                "Capacidad": sim["capacity"],
                "Entrantes": sim["incoming"],
                "Cobertura %": sim["coverage_rate"],
                "Backlog no absorbido": sim["uncovered"],
                "Importe no revisado": sim["uncovered_amount"],
                "Drenaje (periodos)": sim["drain_periods"] if math.isfinite(sim["drain_periods"]) else np.nan,
            }
        )

    return pd.DataFrame(rows)


def build_sensitivity_table(
    df: pd.DataFrame,
    analyst_range: List[int],
    throughput_range: List[int],
    workdays_per_week: int,
    initial_backlog: int,
    horizon_mode: str,
) -> pd.DataFrame:
    rows = []
    for analysts in analyst_range:
        for throughput in throughput_range:
            sim = simulate_capacity(
                df=df,
                n_analysts=analysts,
                alerts_per_analyst_day=throughput,
                workdays_per_week=workdays_per_week,
                initial_backlog=initial_backlog,
                horizon_mode=horizon_mode,
            )
            rows.append(
                {
                    "Analistas": analysts,
                    "Throughput": throughput,
                    "Cobertura %": sim["coverage_rate"],
                    "Backlog": sim["uncovered"],
                    "Importe no revisado": sim["uncovered_amount"],
                }
            )
    return pd.DataFrame(rows)


# =========================================================
# LECTURAS EJECUTIVAS
# =========================================================
def build_executive_readout(sim: Dict[str, float], horizon_mode: str) -> List[str]:
    period_label = "día" if horizon_mode == "Diario" else "semana"
    bullets = []

    if sim["coverage_rate"] >= 95:
        bullets.append(
            f"La operación está cerca del equilibrio: la capacidad cubre aproximadamente el {fmt_pct(sim['coverage_rate'])} de la demanda del periodo."
        )
    elif sim["coverage_rate"] >= 75:
        bullets.append(
            f"Existe presión operativa moderada: una parte relevante de la cola entra en revisión, pero quedan aproximadamente {fmt_int(sim['uncovered'])} alertas fuera de capacidad por {period_label}."
        )
    else:
        bullets.append(
            f"La operación está claramente tensionada: solo se cubriría en torno al {fmt_pct(sim['coverage_rate'])} y quedarían {fmt_int(sim['uncovered'])} alertas fuera de capacidad por {period_label}."
        )

    if sim["uncovered_amount"] > 0:
        bullets.append(
            f"El importe económico potencialmente no revisado asciende a {fmt_money(sim['uncovered_amount'])} por {period_label}, lo que indica riesgo residual operativo."
        )

    if math.isfinite(sim["drain_periods"]):
        if sim["drain_periods"] == 0:
            bullets.append("No hay backlog inicial pendiente de drenaje.")
        else:
            bullets.append(
                f"Con la configuración actual, el backlog inicial tardaría aproximadamente {fmt_num(sim['drain_periods'], 1)} periodos en drenarse."
            )
    else:
        bullets.append(
            "Con la configuración actual, la capacidad no supera a la entrada estructural; la cola no se drenaría y tendería a seguir creciendo."
        )

    return bullets[:3]


def build_bottleneck_readout(alloc_df: pd.DataFrame) -> List[str]:
    if alloc_df.empty:
        return ["No hay datos suficientes para identificar cuellos de botella."]

    out = []

    pressure_by_priority = (
        alloc_df.groupby(["__priority__", "__review_status__"], dropna=False)
        .size()
        .reset_index(name="n")
    )

    critical_out = pressure_by_priority[
        (pressure_by_priority["__priority__"] == "Crítica") &
        (pressure_by_priority["__review_status__"] == "Fuera de capacidad")
    ]["n"].sum()

    high_out = pressure_by_priority[
        (pressure_by_priority["__priority__"] == "Alta") &
        (pressure_by_priority["__review_status__"] == "Fuera de capacidad")
    ]["n"].sum()

    if critical_out > 0:
        out.append(
            f"Parte de la prioridad crítica queda fuera de capacidad ({fmt_int(critical_out)} alertas), señal de infradimensionamiento severo."
        )
    elif high_out > 0:
        out.append(
            f"La prioridad crítica entra en revisión, pero ya se observan {fmt_int(high_out)} alertas de prioridad alta fuera de capacidad."
        )
    else:
        out.append(
            "La parte alta de la cola queda absorbida; la presión residual se desplaza hacia prioridades medias o bajas."
        )

    uncovered = alloc_df[alloc_df["__review_status__"] == "Fuera de capacidad"]
    if not uncovered.empty:
        top_bucket = (
            uncovered.groupby("__risk_bucket__", dropna=False)
            .size()
            .sort_values(ascending=False)
            .index[0]
        )
        out.append(
            f"El bucket con mayor volumen fuera de capacidad es '{top_bucket}', útil para rediseñar reglas, umbrales o staffing."
        )

        out_amount = uncovered["__amount__"].fillna(0).sum()
        out.append(
            f"El volumen económico fuera de capacidad asciende a {fmt_money(out_amount)}, por lo que la limitación no es solo de volumen sino también de exposición."
        )
    else:
        out.append("No se detecta volumen fuera de capacidad con la configuración actual.")

    return out[:3]


# =========================================================
# PLOTS
# =========================================================
def plot_demand_vs_capacity(sim: Dict[str, float], horizon_mode: str) -> go.Figure:
    period_label = "día" if horizon_mode == "Diario" else "semana"

    fig = go.Figure()
    fig.add_bar(
        x=[f"Entrantes por {period_label}", f"Capacidad por {period_label}"],
        y=[sim["incoming"], sim["capacity"]],
        text=[fmt_int(sim["incoming"]), fmt_int(sim["capacity"])],
        textposition="outside",
        name="Volumen",
    )
    fig.update_layout(
        height=380,
        margin=dict(l=20, r=20, t=20, b=20),
        showlegend=False,
        yaxis_title="Número de alertas",
    )
    return fig


def plot_daily_pressure(df: pd.DataFrame, capacity_per_day: float) -> go.Figure:
    day_df = (
        df.groupby("__event_day__", dropna=False)
        .agg(alertas=("__alert_id_text__", "count"), importe=("__amount__", "sum"))
        .reset_index()
        .sort_values("__event_day__")
    )

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=day_df["__event_day__"],
            y=day_df["alertas"],
            mode="lines+markers",
            name="Alertas entrantes",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=day_df["__event_day__"],
            y=[capacity_per_day] * len(day_df),
            mode="lines",
            name="Capacidad diaria",
            line=dict(dash="dash"),
        )
    )
    fig.update_layout(
        height=380,
        margin=dict(l=20, r=20, t=20, b=20),
        yaxis_title="Alertas",
        xaxis_title="Fecha",
    )
    return fig


def plot_reviewable_split(alloc_df: pd.DataFrame) -> go.Figure:
    tmp = (
        alloc_df.groupby("__review_status__", dropna=False)
        .agg(alertas=("__alert_id_text__", "count"), importe=("__amount__", "sum"))
        .reset_index()
    )

    fig = px.bar(
        tmp,
        x="__review_status__",
        y="alertas",
        text="alertas",
        hover_data={"importe": ":,.0f"},
    )
    fig.update_traces(texttemplate="%{text}", textposition="outside")
    fig.update_layout(
        height=360,
        margin=dict(l=20, r=20, t=20, b=20),
        xaxis_title="Estado operativo",
        yaxis_title="Número de alertas",
    )
    return fig


def plot_priority_pressure(alloc_df: pd.DataFrame) -> go.Figure:
    order = ["Crítica", "Alta", "Media", "Baja"]
    tmp = (
        alloc_df.groupby(["__priority__", "__review_status__"], dropna=False)
        .size()
        .reset_index(name="alertas")
    )
    tmp["__priority__"] = pd.Categorical(tmp["__priority__"], categories=order, ordered=True)
    tmp = tmp.sort_values("__priority__")

    fig = px.bar(
        tmp,
        x="__priority__",
        y="alertas",
        color="__review_status__",
        barmode="stack",
        text="alertas",
    )
    fig.update_traces(textposition="inside")
    fig.update_layout(
        height=380,
        margin=dict(l=20, r=20, t=20, b=20),
        xaxis_title="Prioridad",
        yaxis_title="Alertas",
        legend_title="Estado",
    )
    return fig


def plot_bucket_pressure(alloc_df: pd.DataFrame) -> go.Figure:
    bucket_order = ["Muy alto", "Alto", "Medio", "Bajo", "Muy bajo", "N/A"]
    tmp = (
        alloc_df.groupby(["__risk_bucket__", "__review_status__"], dropna=False)
        .size()
        .reset_index(name="alertas")
    )
    tmp["__risk_bucket__"] = pd.Categorical(tmp["__risk_bucket__"], categories=bucket_order, ordered=True)
    tmp = tmp.sort_values("__risk_bucket__")

    fig = px.bar(
        tmp,
        x="__risk_bucket__",
        y="alertas",
        color="__review_status__",
        barmode="stack",
        text="alertas",
    )
    fig.update_layout(
        height=380,
        margin=dict(l=20, r=20, t=20, b=20),
        xaxis_title="Risk bucket",
        yaxis_title="Alertas",
        legend_title="Estado",
    )
    return fig


def plot_scenario_coverage(scenario_df: pd.DataFrame) -> go.Figure:
    fig = px.bar(
        scenario_df,
        x="Escenario",
        y="Cobertura %",
        text="Cobertura %",
        hover_data=["Analistas", "Alertas por analista/día", "Backlog no absorbido", "Importe no revisado"],
    )
    fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
    fig.update_layout(
        height=380,
        margin=dict(l=20, r=20, t=20, b=20),
        yaxis_title="Cobertura %",
    )
    return fig


def plot_scenario_backlog(scenario_df: pd.DataFrame) -> go.Figure:
    fig = px.bar(
        scenario_df,
        x="Escenario",
        y="Backlog no absorbido",
        text="Backlog no absorbido",
        hover_data=["Analistas", "Alertas por analista/día", "Cobertura %", "Importe no revisado"],
    )
    fig.update_traces(texttemplate="%{text:.0f}", textposition="outside")
    fig.update_layout(
        height=380,
        margin=dict(l=20, r=20, t=20, b=20),
        yaxis_title="Alertas fuera de capacidad",
    )
    return fig


def plot_sensitivity_heatmap(sensitivity_df: pd.DataFrame) -> go.Figure:
    pivot = sensitivity_df.pivot(index="Analistas", columns="Throughput", values="Cobertura %")
    pivot = pivot.sort_index().sort_index(axis=1)

    fig = go.Figure(
        data=go.Heatmap(
            z=pivot.values,
            x=[str(c) for c in pivot.columns],
            y=[str(i) for i in pivot.index],
            text=np.round(pivot.values, 1),
            texttemplate="%{text}%",
            hovertemplate="Analistas: %{y}<br>Throughput: %{x}<br>Cobertura: %{z:.1f}%<extra></extra>",
        )
    )
    fig.update_layout(
        height=420,
        margin=dict(l=20, r=20, t=20, b=20),
        xaxis_title="Alertas por analista / día",
        yaxis_title="Número de analistas",
    )
    return fig


# =========================================================
# APP
# =========================================================
st.title("Review Capacity")
st.caption("Capacidad operativa del equipo antifraude para absorber la cola de alertas y cuantificar el riesgo que queda fuera de revisión.")

st.markdown(
    """
    <div class="rc-info-box">
        <div class="rc-block-help">
            Esta página responde una pregunta distinta a <b>Alert Queue</b>: no decide qué alerta revisar primero, sino si el equipo antifraude
            <b>tiene capacidad real para revisar el volumen generado</b>. La lectura clave es comparar <b>demanda</b> (alertas entrantes)
            frente a <b>capacidad</b> (alertas revisables), y medir la brecha en forma de <b>backlog</b>, <b>cobertura</b> e
            <b>importe económico potencialmente no revisado</b>.
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
    st.error("No se pudo cargar la base operativa para Review Capacity.")
    st.exception(e)
    st.stop()

filtered_df = build_filters(df)

if filtered_df.empty:
    st.warning("No hay datos con los filtros seleccionados.")
    st.stop()

# =========================================================
# CONFIG OPERATIVA
# =========================================================
st.subheader("Configuración operativa")
st.caption("Define el tamaño del equipo y su productividad para comparar la demanda real de alertas con la capacidad de revisión.")

c1, c2, c3, c4, c5 = st.columns(5)
with c1:
    horizon_mode = st.selectbox("Horizonte de análisis", ["Diario", "Semanal"], index=1)
with c2:
    n_analysts = st.slider("Número de analistas", min_value=1, max_value=50, value=8, step=1)
with c3:
    alerts_per_analyst_day = st.slider("Alertas por analista / día", min_value=5, max_value=120, value=25, step=1)
with c4:
    workdays_per_week = st.slider("Días laborables / semana", min_value=1, max_value=7, value=5, step=1)
with c5:
    initial_backlog = st.number_input("Backlog inicial pendiente", min_value=0, max_value=100000, value=0, step=10)

sim = simulate_capacity(
    df=filtered_df,
    n_analysts=n_analysts,
    alerts_per_analyst_day=alerts_per_analyst_day,
    workdays_per_week=workdays_per_week,
    initial_backlog=initial_backlog,
    horizon_mode=horizon_mode,
)

alloc_df = allocate_reviewable_vs_uncovered(
    filtered_df,
    reviewable_count=int(round(sim["reviewable"])),
)

# =========================================================
# KPIs
# =========================================================
st.subheader("Resumen de capacidad")
st.caption("Mide si el equipo puede absorber la cola actual, cuánto backlog se generaría y qué importe económico quedaría sin revisar.")

k1, k2, k3, k4, k5, k6 = st.columns(6)

with k1:
    st.metric(
        "Alertas entrantes",
        fmt_int(sim["incoming"]),
        help="Volumen estimado de alertas generadas en el horizonte seleccionado.",
    )

with k2:
    st.metric(
        "Capacidad teórica",
        fmt_int(sim["capacity"]),
        help="Capacidad de revisión estimada según analistas y throughput definidos.",
    )

with k3:
    st.metric(
        "Cobertura de revisión",
        fmt_pct(sim["coverage_rate"]),
        help="Porcentaje de la cola total (entrantes + backlog inicial) que podría revisarse.",
    )

with k4:
    st.metric(
        "Backlog no absorbido",
        fmt_int(sim["uncovered"]),
        help="Alertas que quedarían fuera de capacidad en el horizonte seleccionado.",
    )

with k5:
    drain_label = "∞" if not math.isfinite(sim["drain_periods"]) else fmt_num(sim["drain_periods"], 1)
    st.metric(
        "Drenaje estimado de cola",
        drain_label,
        help="Número estimado de periodos necesarios para vaciar el backlog inicial, si la capacidad excede la entrada.",
    )

with k6:
    st.metric(
        "Importe no revisado",
        fmt_money(sim["uncovered_amount"]),
        help="Importe económico aproximado asociado a las alertas que quedarían fuera de capacidad.",
    )

st.markdown(
    """
    <div class="rc-kpi-note">
        Lectura rápida: una cobertura alta indica equilibrio operativo; un backlog persistente y un importe no revisado elevado indican presión y riesgo residual.
    </div>
    """,
    unsafe_allow_html=True,
)

# =========================================================
# LECTURA EJECUTIVA
# =========================================================
st.subheader("Lectura ejecutiva / operativa")
st.caption("Resumen breve para dirección y para el responsable del equipo antifraude.")

for bullet in build_executive_readout(sim, horizon_mode):
    st.markdown(f"- {bullet}")

# =========================================================
# DEMANDA VS CAPACIDAD
# =========================================================
st.subheader("Demanda vs capacidad")
st.caption("Compara el volumen de alertas generado frente a la capacidad estimada de revisión del equipo.")

c1, c2 = st.columns(2)

with c1:
    st.markdown("**Volumen entrante frente a capacidad**")
    st.markdown(
        "<div class='rc-block-help'>Si la barra de entrada supera a la capacidad, la cola tenderá a crecer y aparecerá backlog operativo.</div>",
        unsafe_allow_html=True,
    )
    st.plotly_chart(plot_demand_vs_capacity(sim, horizon_mode), use_container_width=True)

with c2:
    st.markdown("**Parte revisable vs parte fuera de capacidad**")
    st.markdown(
        "<div class='rc-block-help'>Separa la cola entre alertas que sí entran en revisión y alertas que previsiblemente quedarían fuera con la capacidad actual.</div>",
        unsafe_allow_html=True,
    )
    st.plotly_chart(plot_reviewable_split(alloc_df), use_container_width=True)

# =========================================================
# PRESIÓN TEMPORAL
# =========================================================
st.subheader("Presión operativa en el tiempo")
st.caption("Permite ver si la capacidad diaria soporta los picos de entrada o si hay jornadas donde la cola presiona al equipo.")

st.markdown(
    "<div class='rc-block-help'>Los picos por encima de la línea de capacidad indican días donde la operación se tensiona y puede empezar a acumularse backlog.</div>",
    unsafe_allow_html=True,
)
st.plotly_chart(plot_daily_pressure(filtered_df, sim["capacity_per_day"]), use_container_width=True)

# =========================================================
# PRESIÓN POR PRIORIDAD Y RIESGO
# =========================================================
st.subheader("Presión sobre la cola priorizada")
st.caption("Muestra qué parte de la cola alta entra realmente en revisión y qué segmentos quedarían fuera.")

c1, c2 = st.columns(2)

with c1:
    st.markdown("**Cobertura por prioridad operativa**")
    st.markdown(
        "<div class='rc-block-help'>Idealmente, la prioridad crítica y alta deberían quedar absorbidas. Si quedan fuera, la operación está claramente infradimensionada.</div>",
        unsafe_allow_html=True,
    )
    st.plotly_chart(plot_priority_pressure(alloc_df), use_container_width=True)

with c2:
    st.markdown("**Cobertura por risk bucket**")
    st.markdown(
        "<div class='rc-block-help'>Ayuda a ver si la falta de capacidad expulsa primero buckets bajos o si incluso parte del riesgo alto queda sin revisar.</div>",
        unsafe_allow_html=True,
    )
    st.plotly_chart(plot_bucket_pressure(alloc_df), use_container_width=True)

# =========================================================
# ESCENARIOS
# =========================================================
st.subheader("Escenarios de capacidad")
st.caption("Simula cómo cambiaría la cobertura si el equipo se reduce, se refuerza o mejora su productividad.")

scenario_df = build_scenario_table(
    df=filtered_df,
    base_analysts=n_analysts,
    alerts_per_analyst_day=alerts_per_analyst_day,
    workdays_per_week=workdays_per_week,
    initial_backlog=initial_backlog,
    horizon_mode=horizon_mode,
)

c1, c2 = st.columns(2)

with c1:
    st.markdown("**Cobertura por escenario**")
    st.markdown(
        "<div class='rc-block-help'>Sirve para valorar si un refuerzo moderado ya estabiliza la operación o si hace falta una ampliación más agresiva.</div>",
        unsafe_allow_html=True,
    )
    st.plotly_chart(plot_scenario_coverage(scenario_df), use_container_width=True)

with c2:
    st.markdown("**Backlog por escenario**")
    st.markdown(
        "<div class='rc-block-help'>Cuanto menor sea este backlog, más sostenible será la operación y menor será el riesgo residual fuera de revisión.</div>",
        unsafe_allow_html=True,
    )
    st.plotly_chart(plot_scenario_backlog(scenario_df), use_container_width=True)

st.markdown("**Tabla de escenarios**")
scenario_display = scenario_df.copy()
for col in ["Capacidad", "Entrantes", "Backlog no absorbido"]:
    scenario_display[col] = scenario_display[col].apply(fmt_int)
scenario_display["Cobertura %"] = scenario_display["Cobertura %"].apply(lambda x: fmt_pct(x, 1))
scenario_display["Importe no revisado"] = scenario_display["Importe no revisado"].apply(fmt_money)
scenario_display["Drenaje (periodos)"] = scenario_display["Drenaje (periodos)"].apply(
    lambda x: "∞" if pd.isna(x) else fmt_num(x, 1)
)
st.dataframe(scenario_display, use_container_width=True, hide_index=True)

# =========================================================
# SENSIBILIDAD
# =========================================================
st.subheader("Sensibilidad operativa")
st.caption("Analiza cómo cambia la cobertura al mover simultáneamente número de analistas y throughput por analista.")

analyst_min = max(1, n_analysts - 4)
analyst_max = n_analysts + 4
throughput_min = max(5, alerts_per_analyst_day - 10)
throughput_max = alerts_per_analyst_day + 10

sensitivity_df = build_sensitivity_table(
    df=filtered_df,
    analyst_range=list(range(analyst_min, analyst_max + 1)),
    throughput_range=list(range(throughput_min, throughput_max + 1, 5)),
    workdays_per_week=workdays_per_week,
    initial_backlog=initial_backlog,
    horizon_mode=horizon_mode,
)

st.markdown(
    "<div class='rc-block-help'>La matriz permite detectar combinaciones mínimas de staffing y productividad para mantener la cola bajo control.</div>",
    unsafe_allow_html=True,
)
st.plotly_chart(plot_sensitivity_heatmap(sensitivity_df), use_container_width=True)

# =========================================================
# CUELLOS DE BOTELLA
# =========================================================
st.subheader("Cuellos de botella / presión operativa")
st.caption("Identifica dónde se concentra la tensión de la operación y qué parte del riesgo se queda fuera.")

for bullet in build_bottleneck_readout(alloc_df):
    st.markdown(f"- {bullet}")

# =========================================================
# DETALLE OPERATIVO
# =========================================================
st.subheader("Detalle operativo de la cola")
st.caption("Vista de apoyo para distinguir la parte de la cola que entra en revisión de la que probablemente quedaría fuera con la capacidad actual.")

detail_cols = [
    "__event_day__",
    "__priority__",
    "__risk_bucket__",
    "__risk_score__",
    "__amount__",
    "__channel__",
    "__review_status__",
    "__alert_id_text__",
]

detail_df = alloc_df[detail_cols].copy()
detail_df = detail_df.rename(
    columns={
        "__event_day__": "Fecha",
        "__priority__": "Prioridad",
        "__risk_bucket__": "Risk bucket",
        "__risk_score__": "Risk score",
        "__amount__": "Importe",
        "__channel__": "Canal",
        "__review_status__": "Estado de capacidad",
        "__alert_id_text__": "Alert ID",
    }
)

detail_df["Fecha"] = pd.to_datetime(detail_df["Fecha"], errors="coerce").dt.strftime("%Y-%m-%d")
detail_df["Importe"] = detail_df["Importe"].apply(fmt_money)
detail_df["Risk score"] = detail_df["Risk score"].apply(lambda x: fmt_num(x, 3) if pd.notna(x) else "N/A")

st.dataframe(
    detail_df.head(500),
    use_container_width=True,
    hide_index=True,
)

# =========================================================
# NOTA FINAL
# =========================================================
st.markdown("---")
st.markdown(
    """
    **Cómo interpretar esta página**
    
    - **Cobertura alta + backlog bajo**: operación razonablemente dimensionada.  
    - **Cobertura media + backlog creciente**: la cola entra en tensión y conviene reforzar capacidad o ajustar reglas.  
    - **Prioridad alta fuera de capacidad**: señal clara de infradimensionamiento operativo.  
    - **Importe no revisado elevado**: no solo falta capacidad, también queda exposición económica sin tratar.  
    """
)
