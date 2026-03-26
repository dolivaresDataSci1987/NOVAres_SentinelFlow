from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.data.load_dashboard_data import load_selected_datasets


st.set_page_config(
    page_title="Transaction Monitoring | SentinelFlow",
    page_icon="📈",
    layout="wide",
)


# =========================================================
# DATA LOADING
# =========================================================

@st.cache_data
def get_data():
    return load_selected_datasets(
        [
            "daily_monitoring",
            "daily_channel_monitoring",
            "dashboard_scored_transactions",
            "alert_queue",
            "transaction_country_summary",
            "channel_summary",
        ]
    )


data = get_data()

daily_monitoring_df = data["daily_monitoring"].copy()
daily_channel_monitoring_df = data["daily_channel_monitoring"].copy()
scored_df = data["dashboard_scored_transactions"].copy()
alert_queue_df = data["alert_queue"].copy()
country_summary_df = data["transaction_country_summary"].copy()
channel_summary_df = data["channel_summary"].copy()


# =========================================================
# HELPERS
# =========================================================

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(c).strip() for c in out.columns]
    return out


def first_existing_column(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    for col in candidates:
        if col in df.columns:
            return col
    return None


def first_existing_column_case_insensitive(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    lower_map = {c.lower(): c for c in df.columns}
    for candidate in candidates:
        if candidate.lower() in lower_map:
            return lower_map[candidate.lower()]
    return None


def parse_date_column(df: pd.DataFrame, candidates: List[str]) -> Tuple[pd.DataFrame, Optional[str]]:
    out = df.copy()
    col = first_existing_column_case_insensitive(out, candidates)
    if col is None:
        return out, None

    out[col] = pd.to_datetime(out[col], errors="coerce")
    return out, col


def choose_numeric_column(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    col = first_existing_column_case_insensitive(df, candidates)
    if col is not None:
        return col
    return None


def to_numeric_safe(df: pd.DataFrame, col: Optional[str]) -> pd.DataFrame:
    out = df.copy()
    if col is not None and col in out.columns:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def infer_daily_columns(df: pd.DataFrame) -> dict:
    df = normalize_columns(df)

    date_col = first_existing_column_case_insensitive(
        df,
        ["transaction_date", "date", "day", "business_date", "event_date"],
    )

    tx_col = first_existing_column_case_insensitive(
        df,
        ["transactions", "transaction_count", "total_transactions", "tx_count", "n_transactions"],
    )

    alerts_col = first_existing_column_case_insensitive(
        df,
        ["alerts", "alert_count", "total_alerts", "n_alerts", "flagged_transactions"],
    )

    alert_rate_col = first_existing_column_case_insensitive(
        df,
        ["alert_rate", "alerts_rate", "fraud_alert_rate", "flag_rate"],
    )

    avg_score_col = first_existing_column_case_insensitive(
        df,
        ["avg_score", "average_score", "mean_score", "avg_risk_score", "mean_risk_score"],
    )

    return {
        "date_col": date_col,
        "tx_col": tx_col,
        "alerts_col": alerts_col,
        "alert_rate_col": alert_rate_col,
        "avg_score_col": avg_score_col,
    }


def infer_channel_columns(df: pd.DataFrame) -> dict:
    df = normalize_columns(df)

    date_col = first_existing_column_case_insensitive(
        df,
        ["transaction_date", "date", "day", "business_date", "event_date"],
    )

    channel_col = first_existing_column_case_insensitive(
        df,
        ["channel", "transaction_channel", "payment_channel"],
    )

    tx_col = first_existing_column_case_insensitive(
        df,
        ["transactions", "transaction_count", "total_transactions", "tx_count", "n_transactions"],
    )

    alerts_col = first_existing_column_case_insensitive(
        df,
        ["alerts", "alert_count", "total_alerts", "n_alerts", "flagged_transactions"],
    )

    alert_rate_col = first_existing_column_case_insensitive(
        df,
        ["alert_rate", "alerts_rate", "fraud_alert_rate", "flag_rate"],
    )

    return {
        "date_col": date_col,
        "channel_col": channel_col,
        "tx_col": tx_col,
        "alerts_col": alerts_col,
        "alert_rate_col": alert_rate_col,
    }


def infer_scored_columns(df: pd.DataFrame) -> dict:
    df = normalize_columns(df)

    date_col = first_existing_column_case_insensitive(
        df,
        ["transaction_date", "transaction_ts", "date", "timestamp", "event_date"],
    )

    channel_col = first_existing_column_case_insensitive(
        df,
        ["channel", "transaction_channel", "payment_channel"],
    )

    country_col = first_existing_column_case_insensitive(
        df,
        ["transaction_country", "country", "merchant_country", "issuer_country"],
    )

    cross_border_col = first_existing_column_case_insensitive(
        df,
        ["is_cross_border", "cross_border_flag", "cross_border", "international_flag"],
    )

    score_col = first_existing_column_case_insensitive(
        df,
        ["fraud_score", "score", "prediction_score", "risk_score", "model_score"],
    )

    fraud_label_col = first_existing_column_case_insensitive(
        df,
        ["fraud_label", "label", "is_fraud", "target"],
    )

    amount_col = first_existing_column_case_insensitive(
        df,
        ["amount", "transaction_amount", "payment_amount", "amount_usd"],
    )

    return {
        "date_col": date_col,
        "channel_col": channel_col,
        "country_col": country_col,
        "cross_border_col": cross_border_col,
        "score_col": score_col,
        "fraud_label_col": fraud_label_col,
        "amount_col": amount_col,
    }


def infer_country_summary_columns(df: pd.DataFrame) -> dict:
    df = normalize_columns(df)

    country_col = first_existing_column_case_insensitive(
        df,
        ["transaction_country", "country", "merchant_country"],
    )

    tx_col = first_existing_column_case_insensitive(
        df,
        ["transactions", "transaction_count", "total_transactions", "tx_count", "n_transactions"],
    )

    alerts_col = first_existing_column_case_insensitive(
        df,
        ["alerts", "alert_count", "total_alerts", "n_alerts", "flagged_transactions"],
    )

    alert_rate_col = first_existing_column_case_insensitive(
        df,
        ["alert_rate", "alerts_rate", "fraud_alert_rate", "flag_rate"],
    )

    return {
        "country_col": country_col,
        "tx_col": tx_col,
        "alerts_col": alerts_col,
        "alert_rate_col": alert_rate_col,
    }


def infer_channel_summary_columns(df: pd.DataFrame) -> dict:
    df = normalize_columns(df)

    channel_col = first_existing_column_case_insensitive(
        df,
        ["channel", "transaction_channel", "payment_channel"],
    )

    tx_col = first_existing_column_case_insensitive(
        df,
        ["transactions", "transaction_count", "total_transactions", "tx_count", "n_transactions"],
    )

    alerts_col = first_existing_column_case_insensitive(
        df,
        ["alerts", "alert_count", "total_alerts", "n_alerts", "flagged_transactions"],
    )

    alert_rate_col = first_existing_column_case_insensitive(
        df,
        ["alert_rate", "alerts_rate", "fraud_alert_rate", "flag_rate"],
    )

    return {
        "channel_col": channel_col,
        "tx_col": tx_col,
        "alerts_col": alerts_col,
        "alert_rate_col": alert_rate_col,
    }


def format_int(x) -> str:
    if x is None or pd.isna(x):
        return "N/A"
    return f"{int(round(float(x))):,}"


def format_pct(x, decimals: int = 2) -> str:
    if x is None or pd.isna(x):
        return "N/A"
    return f"{float(x):.{decimals}%}"


def format_float(x, decimals: int = 4) -> str:
    if x is None or pd.isna(x):
        return "N/A"
    return f"{float(x):.{decimals}f}"


def safe_divide(a, b):
    if b in [0, None] or pd.isna(b):
        return None
    return a / b


def build_daily_from_scored(
    scored_data: pd.DataFrame,
    alert_data: pd.DataFrame,
    scored_meta: dict,
) -> pd.DataFrame:
    if scored_data.empty:
        return pd.DataFrame()

    df = normalize_columns(scored_data)
    alert_df = normalize_columns(alert_data)

    date_col = scored_meta["date_col"]
    score_col = scored_meta["score_col"]

    if date_col is None:
        return pd.DataFrame()

    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df[df[date_col].notna()].copy()
    df["monitoring_date"] = df[date_col].dt.date

    agg_dict = {"transactions": ("monitoring_date", "size")}
    if score_col is not None and score_col in df.columns:
        df[score_col] = pd.to_numeric(df[score_col], errors="coerce")
        agg_dict["avg_score"] = (score_col, "mean")

    daily = df.groupby("monitoring_date", as_index=False).agg(**agg_dict)

    if not alert_df.empty:
        alert_date_col = scored_meta["date_col"]
        if alert_date_col in alert_df.columns:
            alert_df[alert_date_col] = pd.to_datetime(alert_df[alert_date_col], errors="coerce")
            alert_df = alert_df[alert_df[alert_date_col].notna()].copy()
            alert_df["monitoring_date"] = alert_df[alert_date_col].dt.date
            daily_alerts = (
                alert_df.groupby("monitoring_date", as_index=False)
                .size()
                .rename(columns={"size": "alerts"})
            )
            daily = daily.merge(daily_alerts, on="monitoring_date", how="left")
        else:
            daily["alerts"] = None
    else:
        daily["alerts"] = None

    daily["alerts"] = daily["alerts"].fillna(0)
    daily["alert_rate"] = daily["alerts"] / daily["transactions"]

    daily["monitoring_date"] = pd.to_datetime(daily["monitoring_date"], errors="coerce")
    daily = daily.sort_values("monitoring_date").reset_index(drop=True)
    return daily


def build_channel_summary_from_daily_channel(df: pd.DataFrame, meta: dict) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    out = normalize_columns(df)
    channel_col = meta["channel_col"]
    tx_col = meta["tx_col"]
    alerts_col = meta["alerts_col"]
    alert_rate_col = meta["alert_rate_col"]

    if channel_col is None:
        return pd.DataFrame()

    for col in [tx_col, alerts_col, alert_rate_col]:
        if col is not None and col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    agg_parts = {}
    if tx_col is not None:
        agg_parts["transactions"] = (tx_col, "sum")
    if alerts_col is not None:
        agg_parts["alerts"] = (alerts_col, "sum")
    if alert_rate_col is not None:
        agg_parts["avg_alert_rate"] = (alert_rate_col, "mean")

    if not agg_parts:
        return pd.DataFrame()

    summary = out.groupby(channel_col, as_index=False).agg(**agg_parts)

    if "transactions" in summary.columns and "alerts" in summary.columns:
        summary["alert_rate"] = summary["alerts"] / summary["transactions"]
    elif "avg_alert_rate" in summary.columns:
        summary["alert_rate"] = summary["avg_alert_rate"]

    summary = summary.sort_values(
        by="alerts" if "alerts" in summary.columns else "transactions",
        ascending=False,
    ).reset_index(drop=True)

    summary = summary.rename(columns={channel_col: "channel"})
    return summary


def build_country_summary_from_scored(
    scored_data: pd.DataFrame,
    alert_data: pd.DataFrame,
    scored_meta: dict,
) -> pd.DataFrame:
    if scored_data.empty:
        return pd.DataFrame()

    country_col = scored_meta["country_col"]
    if country_col is None or country_col not in scored_data.columns:
        return pd.DataFrame()

    df = normalize_columns(scored_data).copy()
    df[country_col] = df[country_col].fillna("Unknown").astype(str)

    tx_summary = (
        df.groupby(country_col, as_index=False)
        .size()
        .rename(columns={"size": "transactions", country_col: "country"})
    )

    if alert_data.empty or country_col not in alert_data.columns:
        tx_summary["alerts"] = 0
        tx_summary["alert_rate"] = 0.0
        return tx_summary.sort_values("transactions", ascending=False).reset_index(drop=True)

    alert_df = normalize_columns(alert_data).copy()
    alert_df[country_col] = alert_df[country_col].fillna("Unknown").astype(str)

    alert_summary = (
        alert_df.groupby(country_col, as_index=False)
        .size()
        .rename(columns={"size": "alerts", country_col: "country"})
    )

    out = tx_summary.merge(alert_summary, on="country", how="left")
    out["alerts"] = out["alerts"].fillna(0)
    out["alert_rate"] = out["alerts"] / out["transactions"]
    out = out.sort_values("alerts", ascending=False).reset_index(drop=True)
    return out


def compute_cross_border_summary(scored_data: pd.DataFrame, alert_data: pd.DataFrame, meta: dict) -> pd.DataFrame:
    cross_col = meta["cross_border_col"]
    if scored_data.empty or cross_col is None or cross_col not in scored_data.columns:
        return pd.DataFrame()

    df = normalize_columns(scored_data).copy()
    alert_df = normalize_columns(alert_data).copy()

    def normalize_flag(x):
        if pd.isna(x):
            return "Unknown"
        s = str(x).strip().lower()
        if s in {"1", "true", "yes", "y"}:
            return "Cross-border"
        if s in {"0", "false", "no", "n"}:
            return "Domestic"
        return str(x)

    df["cross_border_group"] = df[cross_col].apply(normalize_flag)

    tx = (
        df.groupby("cross_border_group", as_index=False)
        .size()
        .rename(columns={"size": "transactions"})
    )

    if not alert_df.empty and cross_col in alert_df.columns:
        alert_df["cross_border_group"] = alert_df[cross_col].apply(normalize_flag)
        al = (
            alert_df.groupby("cross_border_group", as_index=False)
            .size()
            .rename(columns={"size": "alerts"})
        )
        out = tx.merge(al, on="cross_border_group", how="left")
    else:
        out = tx.copy()
        out["alerts"] = 0

    out["alerts"] = out["alerts"].fillna(0)
    out["alert_rate"] = out["alerts"] / out["transactions"]
    out = out.sort_values("alerts", ascending=False).reset_index(drop=True)
    return out


def make_exec_reading(
    daily_df: pd.DataFrame,
    channel_df: pd.DataFrame,
    country_df: pd.DataFrame,
    cross_df: pd.DataFrame,
) -> List[str]:
    insights = []

    if not daily_df.empty and {"transactions", "alerts", "alert_rate"}.issubset(daily_df.columns):
        last_7 = daily_df.tail(7).copy()
        prev_7 = daily_df.iloc[-14:-7].copy()

        if len(last_7) > 0:
            last_alert_rate = last_7["alert_rate"].mean()

            if len(prev_7) > 0:
                prev_alert_rate = prev_7["alert_rate"].mean()
                if pd.notna(last_alert_rate) and pd.notna(prev_alert_rate):
                    delta = last_alert_rate - prev_alert_rate
                    if delta > 0.002:
                        insights.append(
                            f"El alert rate medio de los últimos 7 días sube frente a la ventana anterior "
                            f"({last_alert_rate:.2%} vs {prev_alert_rate:.2%})."
                        )
                    elif delta < -0.002:
                        insights.append(
                            f"El alert rate medio de los últimos 7 días baja frente a la ventana anterior "
                            f"({last_alert_rate:.2%} vs {prev_alert_rate:.2%})."
                        )
                    else:
                        insights.append(
                            f"El alert rate se mantiene relativamente estable en la última semana "
                            f"({last_alert_rate:.2%})."
                        )

        peak_row = daily_df.sort_values("alerts", ascending=False).head(1)
        if not peak_row.empty:
            peak_date = peak_row.iloc[0]["monitoring_date"]
            peak_alerts = peak_row.iloc[0]["alerts"]
            if pd.notna(peak_date):
                insights.append(
                    f"El pico diario de alertas se produjo el {pd.to_datetime(peak_date).date()} "
                    f"con {int(peak_alerts):,} alertas."
                )

    if not channel_df.empty and {"channel", "alerts", "alert_rate"}.issubset(channel_df.columns):
        top_alert_channel = channel_df.sort_values("alerts", ascending=False).head(1)
        top_rate_channel = channel_df.sort_values("alert_rate", ascending=False).head(1)

        if not top_alert_channel.empty:
            r = top_alert_channel.iloc[0]
            insights.append(
                f"El canal con mayor volumen de alertas es **{r['channel']}** "
                f"con {int(r['alerts']):,} casos."
            )

        if not top_rate_channel.empty:
            r = top_rate_channel.iloc[0]
            insights.append(
                f"El canal con mayor alert rate es **{r['channel']}** "
                f"con una tasa del {r['alert_rate']:.2%}."
            )

    if not country_df.empty and {"country", "alerts"}.issubset(country_df.columns):
        top_country = country_df.sort_values("alerts", ascending=False).head(1)
        if not top_country.empty:
            r = top_country.iloc[0]
            insights.append(
                f"El país con mayor concentración de alertas es **{r['country']}** "
                f"con {int(r['alerts']):,} alertas."
            )

    if not cross_df.empty and {"cross_border_group", "alerts", "alert_rate"}.issubset(cross_df.columns):
        top_cross = cross_df.sort_values("alert_rate", ascending=False).head(1)
        if not top_cross.empty:
            r = top_cross.iloc[0]
            insights.append(
                f"En el corte doméstico vs internacional, **{r['cross_border_group']}** presenta "
                f"el mayor alert rate ({r['alert_rate']:.2%})."
            )

    return insights[:5]


# =========================================================
# NORMALIZATION / PREP
# =========================================================

daily_monitoring_df = normalize_columns(daily_monitoring_df)
daily_channel_monitoring_df = normalize_columns(daily_channel_monitoring_df)
scored_df = normalize_columns(scored_df)
alert_queue_df = normalize_columns(alert_queue_df)
country_summary_df = normalize_columns(country_summary_df)
channel_summary_df = normalize_columns(channel_summary_df)

daily_meta = infer_daily_columns(daily_monitoring_df)
daily_channel_meta = infer_channel_columns(daily_channel_monitoring_df)
scored_meta = infer_scored_columns(scored_df)
country_meta = infer_country_summary_columns(country_summary_df)
channel_meta = infer_channel_summary_columns(channel_summary_df)

# Daily monitoring
if not daily_monitoring_df.empty and daily_meta["date_col"] is not None:
    daily_monitoring_df[daily_meta["date_col"]] = pd.to_datetime(
        daily_monitoring_df[daily_meta["date_col"]],
        errors="coerce",
    )

    for c in [daily_meta["tx_col"], daily_meta["alerts_col"], daily_meta["alert_rate_col"], daily_meta["avg_score_col"]]:
        if c is not None and c in daily_monitoring_df.columns:
            daily_monitoring_df[c] = pd.to_numeric(daily_monitoring_df[c], errors="coerce")

    if daily_meta["alert_rate_col"] is None and daily_meta["tx_col"] and daily_meta["alerts_col"]:
        daily_monitoring_df["alert_rate"] = (
            daily_monitoring_df[daily_meta["alerts_col"]] / daily_monitoring_df[daily_meta["tx_col"]]
        )
        daily_meta["alert_rate_col"] = "alert_rate"

    daily_monitoring_df = daily_monitoring_df.rename(
        columns={
            daily_meta["date_col"]: "monitoring_date",
            daily_meta["tx_col"]: "transactions" if daily_meta["tx_col"] else None,
            daily_meta["alerts_col"]: "alerts" if daily_meta["alerts_col"] else None,
            daily_meta["alert_rate_col"]: "alert_rate" if daily_meta["alert_rate_col"] else None,
            daily_meta["avg_score_col"]: "avg_score" if daily_meta["avg_score_col"] else None,
        }
    )

    daily_monitoring_df = daily_monitoring_df.sort_values("monitoring_date").reset_index(drop=True)
else:
    daily_monitoring_df = build_daily_from_scored(scored_df, alert_queue_df, scored_meta)

# Daily channel monitoring
if not daily_channel_monitoring_df.empty and daily_channel_meta["date_col"] is not None:
    daily_channel_monitoring_df[daily_channel_meta["date_col"]] = pd.to_datetime(
        daily_channel_monitoring_df[daily_channel_meta["date_col"]],
        errors="coerce",
    )
    for c in [daily_channel_meta["tx_col"], daily_channel_meta["alerts_col"], daily_channel_meta["alert_rate_col"]]:
        if c is not None and c in daily_channel_monitoring_df.columns:
            daily_channel_monitoring_df[c] = pd.to_numeric(daily_channel_monitoring_df[c], errors="coerce")

    if daily_channel_meta["alert_rate_col"] is None and daily_channel_meta["tx_col"] and daily_channel_meta["alerts_col"]:
        daily_channel_monitoring_df["alert_rate"] = (
            daily_channel_monitoring_df[daily_channel_meta["alerts_col"]] / daily_channel_monitoring_df[daily_channel_meta["tx_col"]]
        )
        daily_channel_meta["alert_rate_col"] = "alert_rate"

    rename_map = {}
    if daily_channel_meta["date_col"]:
        rename_map[daily_channel_meta["date_col"]] = "monitoring_date"
    if daily_channel_meta["channel_col"]:
        rename_map[daily_channel_meta["channel_col"]] = "channel"
    if daily_channel_meta["tx_col"]:
        rename_map[daily_channel_meta["tx_col"]] = "transactions"
    if daily_channel_meta["alerts_col"]:
        rename_map[daily_channel_meta["alerts_col"]] = "alerts"
    if daily_channel_meta["alert_rate_col"]:
        rename_map[daily_channel_meta["alert_rate_col"]] = "alert_rate"

    daily_channel_monitoring_df = daily_channel_monitoring_df.rename(columns=rename_map)
    daily_channel_monitoring_df = daily_channel_monitoring_df.sort_values(["monitoring_date", "channel"]).reset_index(drop=True)
else:
    daily_channel_monitoring_df = pd.DataFrame()

# Channel summary
if not channel_summary_df.empty and channel_meta["channel_col"] is not None:
    for c in [channel_meta["tx_col"], channel_meta["alerts_col"], channel_meta["alert_rate_col"]]:
        if c is not None and c in channel_summary_df.columns:
            channel_summary_df[c] = pd.to_numeric(channel_summary_df[c], errors="coerce")

    rename_map = {}
    if channel_meta["channel_col"]:
        rename_map[channel_meta["channel_col"]] = "channel"
    if channel_meta["tx_col"]:
        rename_map[channel_meta["tx_col"]] = "transactions"
    if channel_meta["alerts_col"]:
        rename_map[channel_meta["alerts_col"]] = "alerts"
    if channel_meta["alert_rate_col"]:
        rename_map[channel_meta["alert_rate_col"]] = "alert_rate"

    channel_summary_df = channel_summary_df.rename(columns=rename_map)

    if "alert_rate" not in channel_summary_df.columns and {"transactions", "alerts"}.issubset(channel_summary_df.columns):
        channel_summary_df["alert_rate"] = channel_summary_df["alerts"] / channel_summary_df["transactions"]

    channel_summary_df = channel_summary_df.sort_values(
        by="alerts" if "alerts" in channel_summary_df.columns else "transactions",
        ascending=False,
    ).reset_index(drop=True)
else:
    channel_summary_df = build_channel_summary_from_daily_channel(daily_channel_monitoring_df, {"channel_col": "channel", "tx_col": "transactions", "alerts_col": "alerts", "alert_rate_col": "alert_rate"})

# Country summary
if not country_summary_df.empty and country_meta["country_col"] is not None:
    for c in [country_meta["tx_col"], country_meta["alerts_col"], country_meta["alert_rate_col"]]:
        if c is not None and c in country_summary_df.columns:
            country_summary_df[c] = pd.to_numeric(country_summary_df[c], errors="coerce")

    rename_map = {}
    if country_meta["country_col"]:
        rename_map[country_meta["country_col"]] = "country"
    if country_meta["tx_col"]:
        rename_map[country_meta["tx_col"]] = "transactions"
    if country_meta["alerts_col"]:
        rename_map[country_meta["alerts_col"]] = "alerts"
    if country_meta["alert_rate_col"]:
        rename_map[country_meta["alert_rate_col"]] = "alert_rate"

    country_summary_df = country_summary_df.rename(columns=rename_map)

    if "alert_rate" not in country_summary_df.columns and {"transactions", "alerts"}.issubset(country_summary_df.columns):
        country_summary_df["alert_rate"] = country_summary_df["alerts"] / country_summary_df["transactions"]

    country_summary_df = country_summary_df.sort_values(
        by="alerts" if "alerts" in country_summary_df.columns else "transactions",
        ascending=False,
    ).reset_index(drop=True)
else:
    country_summary_df = build_country_summary_from_scored(scored_df, alert_queue_df, scored_meta)

# Cross-border summary
cross_border_summary_df = compute_cross_border_summary(scored_df, alert_queue_df, scored_meta)

# Dates for filters
date_min = None
date_max = None
if not daily_monitoring_df.empty and "monitoring_date" in daily_monitoring_df.columns:
    date_min = pd.to_datetime(daily_monitoring_df["monitoring_date"], errors="coerce").min()
    date_max = pd.to_datetime(daily_monitoring_df["monitoring_date"], errors="coerce").max()

# =========================================================
# SIDEBAR FILTERS
# =========================================================

st.title("📈 Transaction Monitoring")
st.caption("Seguimiento operativo de volumen, alertas y concentración de riesgo transaccional")

with st.sidebar:
    st.markdown("## Filtros")

    if date_min is not None and date_max is not None and pd.notna(date_min) and pd.notna(date_max):
        selected_dates = st.date_input(
            "Ventana temporal",
            value=(date_min.date(), date_max.date()),
            min_value=date_min.date(),
            max_value=date_max.date(),
        )
    else:
        selected_dates = None

    available_channels = []
    if not channel_summary_df.empty and "channel" in channel_summary_df.columns:
        available_channels = sorted(channel_summary_df["channel"].dropna().astype(str).unique().tolist())

    selected_channels = st.multiselect(
        "Canales",
        options=available_channels,
        default=available_channels,
    )

    available_countries = []
    if not country_summary_df.empty and "country" in country_summary_df.columns:
        available_countries = sorted(country_summary_df["country"].dropna().astype(str).unique().tolist())[:50]

    selected_countries = st.multiselect(
        "Países (top disponibles)",
        options=available_countries,
        default=available_countries[:10] if len(available_countries) > 10 else available_countries,
    )

    st.markdown("---")
    st.markdown(
        """
        **Objetivo de esta página**

        - vigilar volumen diario
        - detectar cambios en alertas
        - localizar concentración por canal
        - identificar focos geográficos
        """
    )


# =========================================================
# APPLY FILTERS
# =========================================================

daily_filtered = daily_monitoring_df.copy()
daily_channel_filtered = daily_channel_monitoring_df.copy()
channel_summary_filtered = channel_summary_df.copy()
country_summary_filtered = country_summary_df.copy()

if selected_dates is not None and isinstance(selected_dates, tuple) and len(selected_dates) == 2:
    start_date = pd.to_datetime(selected_dates[0])
    end_date = pd.to_datetime(selected_dates[1])

    if not daily_filtered.empty and "monitoring_date" in daily_filtered.columns:
        daily_filtered = daily_filtered[
            (pd.to_datetime(daily_filtered["monitoring_date"]) >= start_date) &
            (pd.to_datetime(daily_filtered["monitoring_date"]) <= end_date)
        ].copy()

    if not daily_channel_filtered.empty and "monitoring_date" in daily_channel_filtered.columns:
        daily_channel_filtered = daily_channel_filtered[
            (pd.to_datetime(daily_channel_filtered["monitoring_date"]) >= start_date) &
            (pd.to_datetime(daily_channel_filtered["monitoring_date"]) <= end_date)
        ].copy()

if selected_channels:
    if not daily_channel_filtered.empty and "channel" in daily_channel_filtered.columns:
        daily_channel_filtered = daily_channel_filtered[
            daily_channel_filtered["channel"].astype(str).isin(selected_channels)
        ].copy()

    if not channel_summary_filtered.empty and "channel" in channel_summary_filtered.columns:
        channel_summary_filtered = channel_summary_filtered[
            channel_summary_filtered["channel"].astype(str).isin(selected_channels)
        ].copy()

if selected_countries and not country_summary_filtered.empty and "country" in country_summary_filtered.columns:
    country_summary_filtered = country_summary_filtered[
        country_summary_filtered["country"].astype(str).isin(selected_countries)
    ].copy()


# =========================================================
# KPI BLOCK
# =========================================================

total_transactions = None
total_alerts = None
alert_rate = None
avg_score = None
peak_alert_day = None
peak_alerts = None

if not daily_filtered.empty:
    if "transactions" in daily_filtered.columns:
        total_transactions = pd.to_numeric(daily_filtered["transactions"], errors="coerce").sum()

    if "alerts" in daily_filtered.columns:
        total_alerts = pd.to_numeric(daily_filtered["alerts"], errors="coerce").sum()

    if total_transactions not in [None, 0] and pd.notna(total_transactions) and total_alerts is not None:
        alert_rate = total_alerts / total_transactions

    if "avg_score" in daily_filtered.columns:
        avg_score = pd.to_numeric(daily_filtered["avg_score"], errors="coerce").mean()

    if "alerts" in daily_filtered.columns and "monitoring_date" in daily_filtered.columns:
        peak_row = daily_filtered.sort_values("alerts", ascending=False).head(1)
        if not peak_row.empty:
            peak_alert_day = pd.to_datetime(peak_row.iloc[0]["monitoring_date"]).date()
            peak_alerts = peak_row.iloc[0]["alerts"]

# fallback 1: si daily_monitoring no trae volumen, usar scored_df directo
if total_transactions in [None, 0] or pd.isna(total_transactions):
    if scored_df is not None and not scored_df.empty:
        total_transactions = len(scored_df)

# fallback 2: si sigue sin haber alert rate, calcularlo con alert queue
if (alert_rate is None or pd.isna(alert_rate)) and total_transactions not in [None, 0] and total_alerts is not None:
    alert_rate = total_alerts / total_transactions

# fallback 3: score medio desde scored_df
if avg_score is None or pd.isna(avg_score):
    possible_score_cols = ["fraud_score", "score", "prediction_score", "risk_score", "model_score"]
    lower_map = {str(c).lower(): c for c in scored_df.columns}
    score_col = None
    for c in possible_score_cols:
        if c.lower() in lower_map:
            score_col = lower_map[c.lower()]
            break

    if score_col is not None:
        avg_score = pd.to_numeric(scored_df[score_col], errors="coerce").mean()

    if "alerts" in daily_filtered.columns:
        total_alerts = daily_filtered["alerts"].sum()

    if total_transactions not in [None, 0] and total_alerts is not None:
        alert_rate = total_alerts / total_transactions

    if "avg_score" in daily_filtered.columns:
        avg_score = daily_filtered["avg_score"].mean()

    if "alerts" in daily_filtered.columns and "monitoring_date" in daily_filtered.columns:
        peak_row = daily_filtered.sort_values("alerts", ascending=False).head(1)
        if not peak_row.empty:
            peak_alert_day = pd.to_datetime(peak_row.iloc[0]["monitoring_date"]).date()
            peak_alerts = peak_row.iloc[0]["alerts"]

st.markdown("## Vista ejecutiva")

k1, k2, k3, k4, k5 = st.columns(5)

with k1:
    st.metric("Transacciones", format_int(total_transactions))

with k2:
    st.metric("Alertas", format_int(total_alerts))

with k3:
    st.metric("Alert rate", format_pct(alert_rate))

with k4:
    st.metric("Score medio", format_float(avg_score, 4))

with k5:
    st.metric("Pico de alertas", format_int(peak_alerts))

if peak_alert_day is not None:
    st.caption(f"Día con mayor volumen de alertas en la ventana seleccionada: **{peak_alert_day}**")

st.markdown("---")


# =========================================================
# EXECUTIVE READING
# =========================================================

st.markdown("## Lectura ejecutiva")

exec_insights = make_exec_reading(
    daily_df=daily_filtered,
    channel_df=channel_summary_filtered,
    country_df=country_summary_filtered,
    cross_df=cross_border_summary_df,
)

if exec_insights:
    for insight in exec_insights:
        st.markdown(f"- {insight}")
else:
    st.info("Todavía no se pudieron generar insights automáticos con los datos disponibles en esta página.")

st.markdown("---")


# =========================================================
# TIME SERIES
# =========================================================

st.markdown("## Evolución temporal")

if daily_filtered.empty:
    st.warning("No hay datos de monitoring diario disponibles para la ventana seleccionada.")
else:
    ts1, ts2 = st.columns(2)

    with ts1:
        if {"monitoring_date", "transactions"}.issubset(daily_filtered.columns):
            fig_tx = px.line(
                daily_filtered,
                x="monitoring_date",
                y="transactions",
                markers=True,
                title="Transacciones por día",
            )
            fig_tx.update_layout(
                xaxis_title="Fecha",
                yaxis_title="Nº transacciones",
                height=420,
            )
            st.plotly_chart(fig_tx, use_container_width=True)

    with ts2:
        if {"monitoring_date", "alerts"}.issubset(daily_filtered.columns):
            fig_alerts = px.line(
                daily_filtered,
                x="monitoring_date",
                y="alerts",
                markers=True,
                title="Alertas por día",
            )
            fig_alerts.update_layout(
                xaxis_title="Fecha",
                yaxis_title="Nº alertas",
                height=420,
            )
            st.plotly_chart(fig_alerts, use_container_width=True)

    if {"monitoring_date", "alert_rate"}.issubset(daily_filtered.columns):
        fig_alert_rate = px.line(
            daily_filtered,
            x="monitoring_date",
            y="alert_rate",
            markers=True,
            title="Alert rate diario",
        )
        fig_alert_rate.update_layout(
            xaxis_title="Fecha",
            yaxis_title="Alert rate",
            yaxis_tickformat=".2%",
            height=420,
        )
        st.plotly_chart(fig_alert_rate, use_container_width=True)

st.markdown("---")


# =========================================================
# CHANNEL ANALYSIS
# =========================================================

st.markdown("## Análisis por canal")

if channel_summary_filtered.empty:
    st.info("No hay resumen por canal disponible.")
else:
    c1, c2 = st.columns(2)

    with c1:
        if {"channel", "alerts"}.issubset(channel_summary_filtered.columns):
            fig_channel_alerts = px.bar(
                channel_summary_filtered.sort_values("alerts", ascending=False),
                x="channel",
                y="alerts",
                title="Alertas por canal",
            )
            fig_channel_alerts.update_layout(
                xaxis_title="Canal",
                yaxis_title="Nº alertas",
                height=420,
            )
            st.plotly_chart(fig_channel_alerts, use_container_width=True)

    with c2:
        if {"channel", "alert_rate"}.issubset(channel_summary_filtered.columns):
            fig_channel_rate = px.bar(
                channel_summary_filtered.sort_values("alert_rate", ascending=False),
                x="channel",
                y="alert_rate",
                title="Alert rate por canal",
            )
            fig_channel_rate.update_layout(
                xaxis_title="Canal",
                yaxis_title="Alert rate",
                yaxis_tickformat=".2%",
                height=420,
            )
            st.plotly_chart(fig_channel_rate, use_container_width=True)

    if {"channel", "transactions", "alerts", "alert_rate"}.issubset(channel_summary_filtered.columns):
        show_channel_df = channel_summary_filtered.copy()
        show_channel_df["alert_rate"] = show_channel_df["alert_rate"].map(lambda x: f"{x:.2%}" if pd.notna(x) else "N/A")
        st.dataframe(show_channel_df, use_container_width=True, hide_index=True)

st.markdown("---")


# =========================================================
# DAILY CHANNEL TREND
# =========================================================

st.markdown("## Tendencia diaria por canal")

if daily_channel_filtered.empty:
    st.info("No hay datos diarios por canal disponibles.")
else:
    metric_option = st.radio(
        "Métrica a visualizar",
        options=["transactions", "alerts", "alert_rate"],
        horizontal=True,
        format_func=lambda x: {
            "transactions": "Transacciones",
            "alerts": "Alertas",
            "alert_rate": "Alert rate",
        }[x],
    )

    if {"monitoring_date", "channel", metric_option}.issubset(daily_channel_filtered.columns):
        fig_daily_channel = px.line(
            daily_channel_filtered,
            x="monitoring_date",
            y=metric_option,
            color="channel",
            markers=False,
            title=f"Tendencia diaria por canal · {metric_option}",
        )
        fig_daily_channel.update_layout(
            xaxis_title="Fecha",
            yaxis_title=metric_option,
            height=460,
        )
        if metric_option == "alert_rate":
            fig_daily_channel.update_yaxes(tickformat=".2%")
        st.plotly_chart(fig_daily_channel, use_container_width=True)

st.markdown("---")


# =========================================================
# COUNTRY ANALYSIS
# =========================================================

st.markdown("## Análisis geográfico")

if country_summary_filtered.empty:
    st.info("No hay resumen geográfico disponible.")
else:
    geo1, geo2 = st.columns(2)

    top_country_alerts = country_summary_filtered.sort_values(
        "alerts" if "alerts" in country_summary_filtered.columns else "transactions",
        ascending=False,
    ).head(15)

    with geo1:
        if {"country", "alerts"}.issubset(top_country_alerts.columns):
            fig_country_alerts = px.bar(
                top_country_alerts,
                x="country",
                y="alerts",
                title="Top países por alertas",
            )
            fig_country_alerts.update_layout(
                xaxis_title="País",
                yaxis_title="Nº alertas",
                height=420,
            )
            st.plotly_chart(fig_country_alerts, use_container_width=True)

    with geo2:
        if {"country", "alert_rate"}.issubset(top_country_alerts.columns):
            fig_country_rate = px.bar(
                top_country_alerts.sort_values("alert_rate", ascending=False),
                x="country",
                y="alert_rate",
                title="Top países por alert rate",
            )
            fig_country_rate.update_layout(
                xaxis_title="País",
                yaxis_title="Alert rate",
                yaxis_tickformat=".2%",
                height=420,
            )
            st.plotly_chart(fig_country_rate, use_container_width=True)

    show_country_df = country_summary_filtered.copy()
    if "alert_rate" in show_country_df.columns:
        show_country_df["alert_rate"] = show_country_df["alert_rate"].map(lambda x: f"{x:.2%}" if pd.notna(x) else "N/A")
    st.dataframe(show_country_df.head(25), use_container_width=True, hide_index=True)

st.markdown("---")


# =========================================================
# CROSS-BORDER
# =========================================================

st.markdown("## Corte doméstico vs cross-border")

if cross_border_summary_df.empty:
    st.info("No se encontró una columna de cross-border en el dataset scored.")
else:
    cb1, cb2 = st.columns(2)

    with cb1:
        if {"cross_border_group", "alerts"}.issubset(cross_border_summary_df.columns):
            fig_cb_alerts = px.bar(
                cross_border_summary_df,
                x="cross_border_group",
                y="alerts",
                title="Alertas por grupo",
            )
            fig_cb_alerts.update_layout(
                xaxis_title="Grupo",
                yaxis_title="Nº alertas",
                height=380,
            )
            st.plotly_chart(fig_cb_alerts, use_container_width=True)

    with cb2:
        if {"cross_border_group", "alert_rate"}.issubset(cross_border_summary_df.columns):
            fig_cb_rate = px.bar(
                cross_border_summary_df,
                x="cross_border_group",
                y="alert_rate",
                title="Alert rate por grupo",
            )
            fig_cb_rate.update_layout(
                xaxis_title="Grupo",
                yaxis_title="Alert rate",
                yaxis_tickformat=".2%",
                height=380,
            )
            st.plotly_chart(fig_cb_rate, use_container_width=True)

    show_cross_df = cross_border_summary_df.copy()
    if "alert_rate" in show_cross_df.columns:
        show_cross_df["alert_rate"] = show_cross_df["alert_rate"].map(lambda x: f"{x:.2%}" if pd.notna(x) else "N/A")
    st.dataframe(show_cross_df, use_container_width=True, hide_index=True)

st.markdown("---")


# =========================================================
# DATA PREVIEW
# =========================================================

with st.expander("Ver muestras de datos usados en esta página"):
    st.markdown("### daily_monitoring")
    st.dataframe(daily_monitoring_df.head(20), use_container_width=True, hide_index=True)

    st.markdown("### daily_channel_monitoring")
    st.dataframe(daily_channel_monitoring_df.head(20), use_container_width=True, hide_index=True)

    st.markdown("### channel_summary")
    st.dataframe(channel_summary_df.head(20), use_container_width=True, hide_index=True)

    st.markdown("### transaction_country_summary")
    st.dataframe(country_summary_df.head(20), use_container_width=True, hide_index=True)
