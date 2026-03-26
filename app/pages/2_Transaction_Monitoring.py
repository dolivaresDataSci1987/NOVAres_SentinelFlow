from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Optional

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.data.load_dashboard_data import load_selected_datasets


st.set_page_config(
    page_title="Monitorización transaccional | SentinelFlow",
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
            "executive_kpis",
        ]
    )


data = get_data()

daily_monitoring_df = data.get("daily_monitoring", pd.DataFrame()).copy()
daily_channel_monitoring_df = data.get("daily_channel_monitoring", pd.DataFrame()).copy()
scored_df = data.get("dashboard_scored_transactions", pd.DataFrame()).copy()
alert_queue_df = data.get("alert_queue", pd.DataFrame()).copy()
country_summary_df = data.get("transaction_country_summary", pd.DataFrame()).copy()
channel_summary_df = data.get("channel_summary", pd.DataFrame()).copy()
executive_kpis_df = data.get("executive_kpis", pd.DataFrame()).copy()


# =========================================================
# HELPERS
# =========================================================

ISO2_TO_ISO3 = {
    "AL": "ALB", "AD": "AND", "AT": "AUT", "BY": "BLR", "BE": "BEL", "BA": "BIH",
    "BG": "BGR", "HR": "HRV", "CY": "CYP", "CZ": "CZE", "DK": "DNK", "EE": "EST",
    "FI": "FIN", "FR": "FRA", "DE": "DEU", "GR": "GRC", "HU": "HUN", "IS": "ISL",
    "IE": "IRL", "IT": "ITA", "XK": "XKX", "LV": "LVA", "LI": "LIE", "LT": "LTU",
    "LU": "LUX", "MT": "MLT", "MD": "MDA", "MC": "MCO", "ME": "MNE", "NL": "NLD",
    "MK": "MKD", "NO": "NOR", "PL": "POL", "PT": "PRT", "RO": "ROU", "RU": "RUS",
    "SM": "SMR", "RS": "SRB", "SK": "SVK", "SI": "SVN", "ES": "ESP", "SE": "SWE",
    "CH": "CHE", "TR": "TUR", "UA": "UKR", "GB": "GBR", "UK": "GBR",
}

RISK_ORDER = {
    "low": 1,
    "medium": 2,
    "high": 3,
    "critical": 4,
}


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(c).strip() for c in out.columns]
    return out


def first_existing_column_case_insensitive(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    lower_map = {str(c).lower(): c for c in df.columns}
    for candidate in candidates:
        if candidate.lower() in lower_map:
            return lower_map[candidate.lower()]
    return None


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


def info_box(text: str):
    st.caption(text)


def extract_kpi_value(df: pd.DataFrame, candidate_keys: list[str]) -> float | None:
    if df is None or df.empty:
        return None

    normalized_keys = {str(k).strip().lower() for k in candidate_keys}
    wide_col_map = {str(c).strip().lower(): c for c in df.columns}

    for key in candidate_keys:
        col = wide_col_map.get(str(key).strip().lower())
        if col is not None:
            series = pd.to_numeric(df[col], errors="coerce").dropna()
            if not series.empty:
                return float(series.iloc[0])

    possible_key_cols = ["metric", "kpi", "key", "name", "metric_name", "kpi_name"]
    possible_value_cols = ["value", "metric_value", "kpi_value", "metric_result", "score"]

    key_col = first_existing_column_case_insensitive(df, possible_key_cols)
    value_col = first_existing_column_case_insensitive(df, possible_value_cols)

    if key_col is not None and value_col is not None:
        tmp = df.copy()
        tmp[key_col] = tmp[key_col].astype(str).str.strip().str.lower()
        match = tmp[tmp[key_col].isin(normalized_keys)]
        if not match.empty:
            series = pd.to_numeric(match[value_col], errors="coerce").dropna()
            if not series.empty:
                return float(series.iloc[0])

    return None


def infer_daily_columns(df: pd.DataFrame) -> dict:
    df = normalize_columns(df)

    return {
        "date_col": first_existing_column_case_insensitive(df, ["transaction_date", "date", "day", "business_date", "event_date"]),
        "tx_col": first_existing_column_case_insensitive(df, ["transactions", "transaction_count", "total_transactions", "tx_count", "n_transactions"]),
        "alerts_col": first_existing_column_case_insensitive(df, ["alerts", "alert_count", "total_alerts", "n_alerts", "flagged_transactions"]),
        "alert_rate_col": first_existing_column_case_insensitive(df, ["alert_rate", "alerts_rate", "fraud_alert_rate", "flag_rate"]),
        "avg_score_col": first_existing_column_case_insensitive(df, ["avg_score", "average_score", "mean_score", "avg_risk_score", "mean_risk_score"]),
    }


def infer_channel_columns(df: pd.DataFrame) -> dict:
    df = normalize_columns(df)

    return {
        "date_col": first_existing_column_case_insensitive(df, ["transaction_date", "date", "day", "business_date", "event_date"]),
        "channel_col": first_existing_column_case_insensitive(df, ["channel", "transaction_channel", "payment_channel"]),
        "tx_col": first_existing_column_case_insensitive(df, ["transactions", "transaction_count", "total_transactions", "tx_count", "n_transactions"]),
        "alerts_col": first_existing_column_case_insensitive(df, ["alerts", "alert_count", "total_alerts", "n_alerts", "flagged_transactions"]),
        "alert_rate_col": first_existing_column_case_insensitive(df, ["alert_rate", "alerts_rate", "fraud_alert_rate", "flag_rate"]),
    }


def infer_scored_columns(df: pd.DataFrame) -> dict:
    df = normalize_columns(df)

    return {
        "date_col": first_existing_column_case_insensitive(df, ["transaction_ts", "transaction_date", "date", "timestamp", "event_date"]),
        "channel_col": first_existing_column_case_insensitive(df, ["channel", "transaction_channel", "payment_channel"]),
        "country_col": first_existing_column_case_insensitive(df, ["transaction_country", "country", "merchant_country", "issuer_country"]),
        "cross_border_col": first_existing_column_case_insensitive(df, ["is_cross_border", "cross_border_flag", "cross_border", "international_flag"]),
        "score_col": first_existing_column_case_insensitive(df, ["fraud_score", "score", "prediction_score", "risk_score", "model_score"]),
        "fraud_label_col": first_existing_column_case_insensitive(df, ["fraud_label", "label", "is_fraud", "target"]),
        "amount_col": first_existing_column_case_insensitive(df, ["amount", "transaction_amount", "payment_amount", "amount_usd", "amount_eur"]),
        "payment_type_col": first_existing_column_case_insensitive(df, ["payment_type", "payment_method_type", "payment_method", "card_type"]),
        "merchant_category_col": first_existing_column_case_insensitive(df, ["merchant_category", "merchant_mcc_group", "merchant_segment", "merchant_type"]),
        "risk_bucket_col": first_existing_column_case_insensitive(df, ["risk_bucket", "risk_segment", "score_bucket", "bucket"]),
    }


def infer_country_summary_columns(df: pd.DataFrame) -> dict:
    df = normalize_columns(df)

    return {
        "country_col": first_existing_column_case_insensitive(df, ["transaction_country", "country", "merchant_country"]),
        "tx_col": first_existing_column_case_insensitive(df, ["transactions", "transaction_count", "total_transactions", "tx_count", "n_transactions"]),
        "alerts_col": first_existing_column_case_insensitive(df, ["alerts", "alert_count", "total_alerts", "n_alerts", "flagged_transactions"]),
        "alert_rate_col": first_existing_column_case_insensitive(df, ["alert_rate", "alerts_rate", "fraud_alert_rate", "flag_rate"]),
    }


def infer_channel_summary_columns(df: pd.DataFrame) -> dict:
    df = normalize_columns(df)

    return {
        "channel_col": first_existing_column_case_insensitive(df, ["channel", "transaction_channel", "payment_channel"]),
        "tx_col": first_existing_column_case_insensitive(df, ["transactions", "transaction_count", "total_transactions", "tx_count", "n_transactions"]),
        "alerts_col": first_existing_column_case_insensitive(df, ["alerts", "alert_count", "total_alerts", "n_alerts", "flagged_transactions"]),
        "alert_rate_col": first_existing_column_case_insensitive(df, ["alert_rate", "alerts_rate", "fraud_alert_rate", "flag_rate"]),
    }


def build_daily_from_scored(scored_data: pd.DataFrame, alert_data: pd.DataFrame, scored_meta: dict) -> pd.DataFrame:
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

    if not alert_df.empty and date_col in alert_df.columns:
        alert_df[date_col] = pd.to_datetime(alert_df[date_col], errors="coerce")
        alert_df = alert_df[alert_df[date_col].notna()].copy()
        alert_df["monitoring_date"] = alert_df[date_col].dt.date
        daily_alerts = (
            alert_df.groupby("monitoring_date", as_index=False)
            .size()
            .rename(columns={"size": "alerts"})
        )
        daily = daily.merge(daily_alerts, on="monitoring_date", how="left")
    else:
        daily["alerts"] = None

    daily["alerts"] = daily["alerts"].fillna(0)
    daily["alert_rate"] = daily["alerts"] / daily["transactions"]
    daily["monitoring_date"] = pd.to_datetime(daily["monitoring_date"], errors="coerce")
    return daily.sort_values("monitoring_date").reset_index(drop=True)


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

    return summary.rename(columns={channel_col: "channel"})


def build_country_summary_from_scored(scored_data: pd.DataFrame, alert_data: pd.DataFrame, scored_meta: dict) -> pd.DataFrame:
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
    return out.sort_values("alerts", ascending=False).reset_index(drop=True)


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
            return "Doméstico"
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
    return out.sort_values("alerts", ascending=False).reset_index(drop=True)


def apply_scored_filters(df: pd.DataFrame, meta: dict, start_date=None, end_date=None, selected_channels=None, selected_countries=None) -> pd.DataFrame:
    out = df.copy()
    if out.empty:
        return out

    date_col = meta["date_col"]
    channel_col = meta["channel_col"]
    country_col = meta["country_col"]

    if start_date is not None and end_date is not None and date_col is not None and date_col in out.columns:
        out[date_col] = pd.to_datetime(out[date_col], errors="coerce")
        out = out[out[date_col].notna()].copy()
        out = out[
            (out[date_col] >= pd.to_datetime(start_date)) &
            (out[date_col] <= pd.to_datetime(end_date))
        ].copy()

    if selected_channels and channel_col is not None and channel_col in out.columns:
        out = out[out[channel_col].astype(str).isin(selected_channels)].copy()

    if selected_countries and country_col is not None and country_col in out.columns:
        out = out[out[country_col].astype(str).isin(selected_countries)].copy()

    return out


def build_score_distribution(df: pd.DataFrame, score_col: Optional[str], bins: int = 25) -> pd.DataFrame:
    if df.empty or score_col is None or score_col not in df.columns:
        return pd.DataFrame()

    out = df.copy()
    out[score_col] = pd.to_numeric(out[score_col], errors="coerce")
    out = out[out[score_col].notna()].copy()
    if out.empty:
        return pd.DataFrame()

    out["score_bin"] = pd.cut(out[score_col], bins=bins, include_lowest=True)
    hist = out.groupby("score_bin", observed=False).size().reset_index(name="transactions")
    hist["score_bin"] = hist["score_bin"].astype(str)
    return hist


def build_amount_by_bucket(df: pd.DataFrame, bucket_col: Optional[str], amount_col: Optional[str]) -> pd.DataFrame:
    if df.empty or bucket_col is None or amount_col is None:
        return pd.DataFrame()
    if bucket_col not in df.columns or amount_col not in df.columns:
        return pd.DataFrame()

    out = df.copy()
    out[amount_col] = pd.to_numeric(out[amount_col], errors="coerce")
    out = out[out[amount_col].notna()].copy()
    if out.empty:
        return pd.DataFrame()

    summary = (
        out.groupby(bucket_col, as_index=False)[amount_col]
        .sum()
        .rename(columns={bucket_col: "bucket_riesgo", amount_col: "importe_total"})
        .sort_values("importe_total", ascending=False)
    )
    return summary


def build_payment_type_alerts(df: pd.DataFrame, payment_col: Optional[str]) -> pd.DataFrame:
    if df.empty or payment_col is None or payment_col not in df.columns:
        return pd.DataFrame()

    return (
        df.groupby(payment_col, as_index=False)
        .size()
        .rename(columns={payment_col: "payment_type", "size": "alerts"})
        .sort_values("alerts", ascending=False)
    )


def build_merchant_category_alerts(df: pd.DataFrame, category_col: Optional[str], top_n: int = 15) -> pd.DataFrame:
    if df.empty or category_col is None or category_col not in df.columns:
        return pd.DataFrame()

    return (
        df.groupby(category_col, as_index=False)
        .size()
        .rename(columns={category_col: "merchant_category", "size": "alerts"})
        .sort_values("alerts", ascending=False)
        .head(top_n)
    )


def build_weekday_alerts(df: pd.DataFrame, date_col: Optional[str]) -> pd.DataFrame:
    if df.empty or date_col is None or date_col not in df.columns:
        return pd.DataFrame()

    out = df.copy()
    out[date_col] = pd.to_datetime(out[date_col], errors="coerce")
    out = out[out[date_col].notna()].copy()
    if out.empty:
        return pd.DataFrame()

    weekday_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    weekday_map_es = {
        "Monday": "Lunes", "Tuesday": "Martes", "Wednesday": "Miércoles",
        "Thursday": "Jueves", "Friday": "Viernes", "Saturday": "Sábado", "Sunday": "Domingo",
    }

    out["weekday"] = out[date_col].dt.day_name()
    summary = out.groupby("weekday", as_index=False).size().rename(columns={"size": "alerts"})
    summary["weekday_order"] = summary["weekday"].map({d: i for i, d in enumerate(weekday_order)})
    summary["día_semana"] = summary["weekday"].map(weekday_map_es)
    return summary.sort_values("weekday_order")[["día_semana", "alerts"]]


def build_hourly_alerts(df: pd.DataFrame, date_col: Optional[str]) -> pd.DataFrame:
    if df.empty or date_col is None or date_col not in df.columns:
        return pd.DataFrame()

    out = df.copy()
    out[date_col] = pd.to_datetime(out[date_col], errors="coerce")
    out = out[out[date_col].notna()].copy()
    if out.empty:
        return pd.DataFrame()

    out["hour"] = out[date_col].dt.hour
    return out.groupby("hour", as_index=False).size().rename(columns={"size": "alerts"}).sort_values("hour")


def build_channel_scatter_df(df: pd.DataFrame) -> pd.DataFrame:
    required = {"channel", "transactions", "alerts", "alert_rate"}
    if df.empty or not required.issubset(df.columns):
        return pd.DataFrame()

    out = df.copy()
    for col in ["transactions", "alerts", "alert_rate"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    return out.dropna(subset=["transactions", "alerts", "alert_rate"]).copy()


def build_channel_heatmap(df: pd.DataFrame, metric_col: str) -> pd.DataFrame:
    required = {"monitoring_date", "channel", metric_col}
    if df.empty or not required.issubset(df.columns):
        return pd.DataFrame()

    out = df.copy()
    out["monitoring_date"] = pd.to_datetime(out["monitoring_date"], errors="coerce")
    out = out[out["monitoring_date"].notna()].copy()
    out["monitoring_date"] = out["monitoring_date"].dt.strftime("%Y-%m-%d")
    pivot = out.pivot_table(index="channel", columns="monitoring_date", values=metric_col, aggfunc="sum")
    return pivot.fillna(0)


def build_channel_share_over_time(df: pd.DataFrame, metric_col: str) -> pd.DataFrame:
    required = {"monitoring_date", "channel", metric_col}
    if df.empty or not required.issubset(df.columns):
        return pd.DataFrame()

    out = df.copy()
    out["monitoring_date"] = pd.to_datetime(out["monitoring_date"], errors="coerce")
    out = out[out["monitoring_date"].notna()].copy()
    out[metric_col] = pd.to_numeric(out[metric_col], errors="coerce")
    out = out[out[metric_col].notna()].copy()
    if out.empty:
        return pd.DataFrame()

    total_per_day = out.groupby("monitoring_date")[metric_col].sum().rename("daily_total").reset_index()
    out = out.merge(total_per_day, on="monitoring_date", how="left")
    out["share"] = out[metric_col] / out["daily_total"]
    return out


def build_geo_metric_df(
    scored_filtered: pd.DataFrame,
    alert_filtered: pd.DataFrame,
    meta: dict,
) -> pd.DataFrame:
    country_col = meta["country_col"]
    amount_col = meta["amount_col"]

    if country_col is None or country_col not in scored_filtered.columns:
        return pd.DataFrame()

    tx = (
        scored_filtered.groupby(country_col, as_index=False)
        .size()
        .rename(columns={country_col: "country", "size": "transactions"})
    )

    if amount_col is not None and amount_col in scored_filtered.columns:
        tmp = scored_filtered.copy()
        tmp[amount_col] = pd.to_numeric(tmp[amount_col], errors="coerce")
        amt = tmp.groupby(country_col, as_index=False)[amount_col].sum().rename(
            columns={country_col: "country", amount_col: "importe_total"}
        )
        tx = tx.merge(amt, on="country", how="left")
    else:
        tx["importe_total"] = None

    if not alert_filtered.empty and country_col in alert_filtered.columns:
        al = (
            alert_filtered.groupby(country_col, as_index=False)
            .size()
            .rename(columns={country_col: "country", "size": "alerts"})
        )
        tx = tx.merge(al, on="country", how="left")
    else:
        tx["alerts"] = 0

    tx["alerts"] = tx["alerts"].fillna(0)
    tx["alert_rate"] = tx["alerts"] / tx["transactions"]
    tx["country"] = tx["country"].astype(str).str.upper()
    tx["iso_alpha"] = tx["country"].map(ISO2_TO_ISO3)
    tx = tx[tx["iso_alpha"].notna()].copy()
    return tx.sort_values("alerts", ascending=False)


def build_risk_bucket_summary(scored_filtered: pd.DataFrame, alert_filtered: pd.DataFrame, meta: dict) -> pd.DataFrame:
    bucket_col = meta["risk_bucket_col"]
    amount_col = meta["amount_col"]
    score_col = meta["score_col"]

    if bucket_col is None or bucket_col not in scored_filtered.columns:
        return pd.DataFrame()

    tmp = scored_filtered.copy()
    group_cols = [bucket_col]

    agg = {bucket_col: "size"}
    rename_map = {bucket_col: "bucket_riesgo"}

    # transactions
    tx = (
        tmp.groupby(bucket_col, as_index=False)
        .size()
        .rename(columns={bucket_col: "bucket_riesgo", "size": "transactions"})
    )

    if amount_col is not None and amount_col in tmp.columns:
        tmp[amount_col] = pd.to_numeric(tmp[amount_col], errors="coerce")
        amt = (
            tmp.groupby(bucket_col, as_index=False)[amount_col]
            .sum()
            .rename(columns={bucket_col: "bucket_riesgo", amount_col: "importe_total"})
        )
        tx = tx.merge(amt, on="bucket_riesgo", how="left")
    else:
        tx["importe_total"] = None

    if score_col is not None and score_col in tmp.columns:
        tmp[score_col] = pd.to_numeric(tmp[score_col], errors="coerce")
        sc = (
            tmp.groupby(bucket_col, as_index=False)[score_col]
            .mean()
            .rename(columns={bucket_col: "bucket_riesgo", score_col: "score_medio"})
        )
        tx = tx.merge(sc, on="bucket_riesgo", how="left")
    else:
        tx["score_medio"] = None

    if not alert_filtered.empty and bucket_col in alert_filtered.columns:
        al = (
            alert_filtered.groupby(bucket_col, as_index=False)
            .size()
            .rename(columns={bucket_col: "bucket_riesgo", "size": "alerts"})
        )
        tx = tx.merge(al, on="bucket_riesgo", how="left")
    else:
        tx["alerts"] = 0

    tx["alerts"] = tx["alerts"].fillna(0)
    tx["alert_rate"] = tx["alerts"] / tx["transactions"]
    tx["bucket_order"] = tx["bucket_riesgo"].astype(str).str.lower().map(RISK_ORDER).fillna(999)
    tx = tx.sort_values(["bucket_order", "alerts"], ascending=[True, False]).reset_index(drop=True)
    return tx


def make_exec_reading(
    daily_df: pd.DataFrame,
    channel_df: pd.DataFrame,
    country_df: pd.DataFrame,
    cross_df: pd.DataFrame,
    scored_filtered_df: pd.DataFrame,
    scored_meta: dict,
) -> List[str]:
    insights = []

    if not daily_df.empty and {"transactions", "alerts", "alert_rate"}.issubset(daily_df.columns):
        last_7 = daily_df.tail(7).copy()
        prev_7 = daily_df.iloc[-14:-7].copy()

        if len(last_7) > 0:
            last_alert_rate = pd.to_numeric(last_7["alert_rate"], errors="coerce").mean()
            if len(prev_7) > 0:
                prev_alert_rate = pd.to_numeric(prev_7["alert_rate"], errors="coerce").mean()
                if pd.notna(last_alert_rate) and pd.notna(prev_alert_rate):
                    delta = last_alert_rate - prev_alert_rate
                    if delta > 0.002:
                        insights.append(
                            f"El alert rate medio de los últimos 7 días sube frente a la ventana anterior ({last_alert_rate:.2%} vs {prev_alert_rate:.2%})."
                        )
                    elif delta < -0.002:
                        insights.append(
                            f"El alert rate medio de los últimos 7 días baja frente a la ventana anterior ({last_alert_rate:.2%} vs {prev_alert_rate:.2%})."
                        )
                    else:
                        insights.append(
                            f"El alert rate se mantiene estable en la última semana, alrededor de **{last_alert_rate:.2%}**."
                        )

        peak_row = daily_df.sort_values("alerts", ascending=False).head(1)
        if not peak_row.empty:
            peak_date = peak_row.iloc[0]["monitoring_date"]
            peak_alerts = peak_row.iloc[0]["alerts"]
            if pd.notna(peak_date):
                insights.append(
                    f"El pico diario de alertas se produjo el **{pd.to_datetime(peak_date).date()}** con **{int(peak_alerts):,} alertas**."
                )

    if not channel_df.empty and {"channel", "alerts", "alert_rate"}.issubset(channel_df.columns):
        top_alert_channel = channel_df.sort_values("alerts", ascending=False).head(1)
        top_rate_channel = channel_df.sort_values("alert_rate", ascending=False).head(1)

        if not top_alert_channel.empty:
            r = top_alert_channel.iloc[0]
            insights.append(
                f"El canal con mayor volumen de alertas es **{r['channel']}** con **{int(r['alerts']):,} casos**."
            )

        if not top_rate_channel.empty:
            r = top_rate_channel.iloc[0]
            insights.append(
                f"El canal con mayor intensidad relativa de alertado es **{r['channel']}** con un alert rate de **{r['alert_rate']:.2%}**."
            )

    if not country_df.empty and {"country", "alerts"}.issubset(country_df.columns):
        top_country = country_df.sort_values("alerts", ascending=False).head(1)
        if not top_country.empty:
            r = top_country.iloc[0]
            insights.append(
                f"El país con mayor concentración de alertas es **{r['country']}** con **{int(r['alerts']):,} alertas**."
            )

    if not cross_df.empty and {"cross_border_group", "alert_rate"}.issubset(cross_df.columns):
        top_cross = cross_df.sort_values("alert_rate", ascending=False).head(1)
        if not top_cross.empty:
            r = top_cross.iloc[0]
            insights.append(
                f"En el corte doméstico vs internacional, **{r['cross_border_group']}** presenta el mayor alert rate (**{r['alert_rate']:.2%}**)."
            )

    amount_col = scored_meta.get("amount_col")
    risk_bucket_col = scored_meta.get("risk_bucket_col")
    if (
        not scored_filtered_df.empty
        and amount_col is not None
        and risk_bucket_col is not None
        and amount_col in scored_filtered_df.columns
        and risk_bucket_col in scored_filtered_df.columns
    ):
        tmp = scored_filtered_df.copy()
        tmp[amount_col] = pd.to_numeric(tmp[amount_col], errors="coerce")
        tmp = tmp[tmp[amount_col].notna()].copy()
        if not tmp.empty:
            exposure_row = (
                tmp.groupby(risk_bucket_col, as_index=False)[amount_col]
                .sum()
                .sort_values(amount_col, ascending=False)
                .head(1)
            )
            if not exposure_row.empty:
                r = exposure_row.iloc[0]
                insights.append(
                    f"El mayor importe agregado se concentra en el bucket **{r[risk_bucket_col]}**, con una exposición de **{r[amount_col]:,.2f}**."
                )

    return insights[:6]


# =========================================================
# NORMALIZATION / PREP
# =========================================================

daily_monitoring_df = normalize_columns(daily_monitoring_df)
daily_channel_monitoring_df = normalize_columns(daily_channel_monitoring_df)
scored_df = normalize_columns(scored_df)
alert_queue_df = normalize_columns(alert_queue_df)
country_summary_df = normalize_columns(country_summary_df)
channel_summary_df = normalize_columns(channel_summary_df)
executive_kpis_df = normalize_columns(executive_kpis_df)

daily_meta = infer_daily_columns(daily_monitoring_df)
daily_channel_meta = infer_channel_columns(daily_channel_monitoring_df)
scored_meta = infer_scored_columns(scored_df)
country_meta = infer_country_summary_columns(country_summary_df)
channel_meta = infer_channel_summary_columns(channel_summary_df)

champion_threshold = extract_kpi_value(
    executive_kpis_df,
    candidate_keys=["champion_threshold", "selected_threshold", "operating_threshold", "model_threshold"],
)
if champion_threshold is None:
    champion_threshold = 0.8987

if not daily_monitoring_df.empty and daily_meta["date_col"] is not None:
    daily_monitoring_df[daily_meta["date_col"]] = pd.to_datetime(
        daily_monitoring_df[daily_meta["date_col"]], errors="coerce"
    )

    for c in [daily_meta["tx_col"], daily_meta["alerts_col"], daily_meta["alert_rate_col"], daily_meta["avg_score_col"]]:
        if c is not None and c in daily_monitoring_df.columns:
            daily_monitoring_df[c] = pd.to_numeric(daily_monitoring_df[c], errors="coerce")

    if daily_meta["alert_rate_col"] is None and daily_meta["tx_col"] and daily_meta["alerts_col"]:
        daily_monitoring_df["alert_rate"] = daily_monitoring_df[daily_meta["alerts_col"]] / daily_monitoring_df[daily_meta["tx_col"]]
        daily_meta["alert_rate_col"] = "alert_rate"

    rename_map = {}
    if daily_meta["date_col"]:
        rename_map[daily_meta["date_col"]] = "monitoring_date"
    if daily_meta["tx_col"]:
        rename_map[daily_meta["tx_col"]] = "transactions"
    if daily_meta["alerts_col"]:
        rename_map[daily_meta["alerts_col"]] = "alerts"
    if daily_meta["alert_rate_col"]:
        rename_map[daily_meta["alert_rate_col"]] = "alert_rate"
    if daily_meta["avg_score_col"]:
        rename_map[daily_meta["avg_score_col"]] = "avg_score"

    daily_monitoring_df = daily_monitoring_df.rename(columns=rename_map)
    daily_monitoring_df = daily_monitoring_df.sort_values("monitoring_date").reset_index(drop=True)
else:
    daily_monitoring_df = build_daily_from_scored(scored_df, alert_queue_df, scored_meta)

if not daily_channel_monitoring_df.empty and daily_channel_meta["date_col"] is not None:
    daily_channel_monitoring_df[daily_channel_meta["date_col"]] = pd.to_datetime(
        daily_channel_monitoring_df[daily_channel_meta["date_col"]], errors="coerce"
    )
    for c in [daily_channel_meta["tx_col"], daily_channel_meta["alerts_col"], daily_channel_meta["alert_rate_col"]]:
        if c is not None and c in daily_channel_monitoring_df.columns:
            daily_channel_monitoring_df[c] = pd.to_numeric(daily_channel_monitoring_df[c], errors="coerce")

    if daily_channel_meta["alert_rate_col"] is None and daily_channel_meta["tx_col"] and daily_channel_meta["alerts_col"]:
        daily_channel_monitoring_df["alert_rate"] = (
            daily_channel_monitoring_df[daily_channel_meta["alerts_col"]] /
            daily_channel_monitoring_df[daily_channel_meta["tx_col"]]
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
    channel_summary_df = build_channel_summary_from_daily_channel(
        daily_channel_monitoring_df,
        {"channel_col": "channel", "tx_col": "transactions", "alerts_col": "alerts", "alert_rate_col": "alert_rate"},
    )

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

cross_border_summary_df = compute_cross_border_summary(scored_df, alert_queue_df, scored_meta)

date_min = None
date_max = None
if not daily_monitoring_df.empty and "monitoring_date" in daily_monitoring_df.columns:
    date_min = pd.to_datetime(daily_monitoring_df["monitoring_date"], errors="coerce").min()
    date_max = pd.to_datetime(daily_monitoring_df["monitoring_date"], errors="coerce").max()


# =========================================================
# SIDEBAR FILTERS
# =========================================================

st.title("📈 Monitorización transaccional")
st.caption("Seguimiento operativo del volumen, alertas, exposición económica y concentración del riesgo")

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

    geo_metric = st.selectbox(
        "Métrica del mapa",
        options=["alerts", "alert_rate", "transactions", "importe_total"],
        format_func=lambda x: {
            "alerts": "Alertas",
            "alert_rate": "Alert rate",
            "transactions": "Transacciones",
            "importe_total": "Importe total",
        }[x],
    )

    heatmap_metric = st.radio(
        "Heatmap por canal",
        options=["alerts", "transactions", "alert_rate"],
        horizontal=False,
        format_func=lambda x: {
            "alerts": "Alertas",
            "transactions": "Transacciones",
            "alert_rate": "Alert rate",
        }[x],
    )

    st.markdown("---")
    st.markdown(
        """
        **Objetivo de esta página**
        
        - vigilar el comportamiento diario del flujo
        - detectar focos por canal, país y franja temporal
        - entender dónde se concentra el importe expuesto
        - traducir el riesgo modelizado a decisiones operativas
        """
    )


# =========================================================
# APPLY FILTERS
# =========================================================

daily_filtered = daily_monitoring_df.copy()
daily_channel_filtered = daily_channel_monitoring_df.copy()
channel_summary_filtered = channel_summary_df.copy()
country_summary_filtered = country_summary_df.copy()

start_date = None
end_date = None

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

scored_filtered = apply_scored_filters(
    scored_df,
    scored_meta,
    start_date=start_date,
    end_date=end_date,
    selected_channels=selected_channels,
    selected_countries=selected_countries,
)

alert_filtered = apply_scored_filters(
    alert_queue_df,
    scored_meta,
    start_date=start_date,
    end_date=end_date,
    selected_channels=selected_channels,
    selected_countries=selected_countries,
)

cross_border_summary_filtered = compute_cross_border_summary(scored_filtered, alert_filtered, scored_meta)
geo_metric_df = build_geo_metric_df(scored_filtered, alert_filtered, scored_meta)
risk_bucket_summary_df = build_risk_bucket_summary(scored_filtered, alert_filtered, scored_meta)


# =========================================================
# KPI BLOCK
# =========================================================

total_transactions = None
total_alerts = None
alert_rate = None
avg_score = None
peak_alert_day = None
peak_alerts = None
amount_col = scored_meta["amount_col"]
score_col = scored_meta["score_col"]

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

if total_transactions in [None, 0] or pd.isna(total_transactions):
    if not scored_filtered.empty:
        total_transactions = len(scored_filtered)

if total_alerts in [None] or pd.isna(total_alerts):
    if not alert_filtered.empty:
        total_alerts = len(alert_filtered)

if (alert_rate is None or pd.isna(alert_rate)) and total_transactions not in [None, 0] and total_alerts is not None:
    alert_rate = total_alerts / total_transactions

if avg_score is None or pd.isna(avg_score):
    if score_col is not None and score_col in scored_filtered.columns:
        avg_score = pd.to_numeric(scored_filtered[score_col], errors="coerce").mean()

total_amount = None
alert_amount = None
if amount_col is not None and amount_col in scored_filtered.columns:
    total_amount = pd.to_numeric(scored_filtered[amount_col], errors="coerce").sum()

if amount_col is not None and amount_col in alert_filtered.columns:
    alert_amount = pd.to_numeric(alert_filtered[amount_col], errors="coerce").sum()

st.markdown("## Vista ejecutiva")

k1, k2, k3, k4, k5, k6 = st.columns(6)

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
with k6:
    st.metric("Importe alertado", f"{alert_amount:,.2f}" if alert_amount is not None and pd.notna(alert_amount) else "N/A")

if peak_alert_day is not None:
    st.caption(f"Día con mayor volumen de alertas en la ventana seleccionada: **{peak_alert_day}**")

st.info(
    """
    **Cómo leer esta cabecera:** resume el tamaño del flujo, la presión de alertado, el nivel medio de riesgo y el importe económico asociado a la cola priorizada.
    """
)

st.markdown("---")


# =========================================================
# EXECUTIVE READING
# =========================================================

st.markdown("## Lectura ejecutiva")

exec_insights = make_exec_reading(
    daily_df=daily_filtered,
    channel_df=channel_summary_filtered,
    country_df=country_summary_filtered,
    cross_df=cross_border_summary_filtered,
    scored_filtered_df=scored_filtered,
    scored_meta=scored_meta,
)

if exec_insights:
    for insight in exec_insights:
        st.markdown(f"- {insight}")
else:
    st.info("Todavía no se pudieron generar insights automáticos con los datos disponibles.")

if total_amount is not None and pd.notna(total_amount) and alert_amount is not None and pd.notna(alert_amount):
    share_alert_amount = safe_divide(alert_amount, total_amount)
    if share_alert_amount is not None:
        st.success(
            f"En la ventana seleccionada, el importe asociado a alertas representa **{share_alert_amount:.2%}** del importe total procesado."
        )

st.markdown("---")


# =========================================================
# TIME EVOLUTION
# =========================================================

st.markdown("## Evolución temporal")
info_box("Mira aquí la dinámica del sistema: volumen, alertas y alert rate. Sirve para detectar cambios de régimen, picos operativos y periodos anómalos.")

if daily_filtered.empty:
    st.warning("No hay datos de monitoring diario disponibles para la ventana seleccionada.")
else:
    combo_fig = make_subplots(specs=[[{"secondary_y": True}]])
    if {"monitoring_date", "alerts"}.issubset(daily_filtered.columns):
        combo_fig.add_trace(
            go.Bar(
                x=daily_filtered["monitoring_date"],
                y=daily_filtered["alerts"],
                name="Alertas",
                opacity=0.55,
            ),
            secondary_y=False,
        )
    if {"monitoring_date", "transactions"}.issubset(daily_filtered.columns):
        combo_fig.add_trace(
            go.Scatter(
                x=daily_filtered["monitoring_date"],
                y=daily_filtered["transactions"],
                mode="lines+markers",
                name="Transacciones",
            ),
            secondary_y=True,
        )
    combo_fig.update_layout(
        title="Volumen y alertas en el tiempo",
        xaxis_title="Fecha",
        yaxis_title="Alertas",
        yaxis2_title="Transacciones",
        height=470,
        legend_title_text="Métrica",
    )
    st.plotly_chart(combo_fig, use_container_width=True)

    t1, t2 = st.columns(2)

    with t1:
        if {"monitoring_date", "alert_rate"}.issubset(daily_filtered.columns):
            fig_alert_rate = px.area(
                daily_filtered,
                x="monitoring_date",
                y="alert_rate",
                title="Alert rate diario",
            )
            fig_alert_rate.update_layout(
                xaxis_title="Fecha",
                yaxis_title="Alert rate",
                yaxis_tickformat=".2%",
                height=400,
            )
            st.plotly_chart(fig_alert_rate, use_container_width=True)

    with t2:
        if {"monitoring_date", "avg_score"}.issubset(daily_filtered.columns):
            fig_score = px.line(
                daily_filtered,
                x="monitoring_date",
                y="avg_score",
                markers=True,
                title="Score medio diario",
            )
            fig_score.update_layout(
                xaxis_title="Fecha",
                yaxis_title="Score medio",
                height=400,
            )
            st.plotly_chart(fig_score, use_container_width=True)

st.markdown("---")


# =========================================================
# CHANNEL EVOLUTION
# =========================================================

st.markdown("## Evolución por canal")
info_box("Esta sección permite distinguir entre canales con mucho volumen absoluto y canales con una intensidad relativa de riesgo más alta de lo normal.")

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
                yaxis_title="Alertas",
                height=400,
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
                height=400,
            )
            st.plotly_chart(fig_channel_rate, use_container_width=True)

    scatter_df = build_channel_scatter_df(channel_summary_filtered)
    if not scatter_df.empty:
        fig_scatter_channel = px.scatter(
            scatter_df,
            x="transactions",
            y="alert_rate",
            size="alerts",
            color="channel",
            hover_name="channel",
            title="Canales: volumen vs alert rate",
        )
        fig_scatter_channel.update_layout(
            xaxis_title="Transacciones",
            yaxis_title="Alert rate",
            yaxis_tickformat=".2%",
            height=480,
        )
        st.plotly_chart(fig_scatter_channel, use_container_width=True)

if daily_channel_filtered.empty:
    st.info("No hay datos diarios por canal disponibles.")
else:
    st.markdown("### Heatmap canal × fecha")
    info_box("Cuanto más intensa la celda, mayor concentración de la métrica en ese canal y ese día. Es una forma rápida de detectar focos persistentes o picos aislados.")

    heatmap_df = build_channel_heatmap(daily_channel_filtered, heatmap_metric)
    if not heatmap_df.empty:
        fig_heatmap = go.Figure(
            data=go.Heatmap(
                z=heatmap_df.values,
                x=list(heatmap_df.columns),
                y=list(heatmap_df.index),
                hoverongaps=False,
            )
        )
        fig_heatmap.update_layout(
            title=f"Heatmap por canal · {heatmap_metric}",
            xaxis_title="Fecha",
            yaxis_title="Canal",
            height=430,
        )
        if heatmap_metric == "alert_rate":
            fig_heatmap.update_coloraxes(colorbar_tickformat=".2%")
        st.plotly_chart(fig_heatmap, use_container_width=True)
    else:
        st.info("No se pudo construir el heatmap por canal.")

    st.markdown("### Peso relativo de cada canal en el tiempo")
    info_box("Aquí no importa solo el volumen total, sino quién gana peso dentro del mix operativo diario.")

    share_metric = "alerts" if "alerts" in daily_channel_filtered.columns else "transactions"
    share_df = build_channel_share_over_time(daily_channel_filtered, share_metric)
    if not share_df.empty:
        fig_area = px.area(
            share_df,
            x="monitoring_date",
            y="share",
            color="channel",
            title=f"Participación diaria por canal · {share_metric}",
        )
        fig_area.update_layout(
            xaxis_title="Fecha",
            yaxis_title="Participación",
            yaxis_tickformat=".0%",
            height=430,
        )
        st.plotly_chart(fig_area, use_container_width=True)
    else:
        st.info("No se pudo construir la evolución relativa por canal.")

st.markdown("---")


# =========================================================
# GEOGRAPHIC ANALYSIS
# =========================================================

st.markdown("## Análisis geográfico")
info_box("Úsalo para separar países con mucho tráfico de países con una intensidad relativa de riesgo desproporcionada. El mapa es útil para detectar clusters regionales.")

if geo_metric_df.empty:
    st.info("No hay resumen geográfico disponible.")
else:
    g1, g2 = st.columns(2)

    with g1:
        choropleth_fig = px.choropleth(
            geo_metric_df,
            locations="iso_alpha",
            color=geo_metric,
            hover_name="country",
            scope="europe",
            title=f"Mapa de Europa · {geo_metric}",
            color_continuous_scale="Viridis",
            hover_data={
                "transactions": ":,.0f",
                "alerts": ":,.0f",
                "alert_rate": ":.2%",
                "importe_total": ":,.2f",
                "iso_alpha": False,
            },
        )
        choropleth_fig.update_layout(height=520, margin=dict(l=0, r=0, t=60, b=0))
        st.plotly_chart(choropleth_fig, use_container_width=True)

    with g2:
        bubble_metric_size = "alerts" if "alerts" in geo_metric_df.columns else "transactions"
        bubble_fig = px.scatter_geo(
            geo_metric_df,
            locations="iso_alpha",
            hover_name="country",
            size=bubble_metric_size,
            color="alert_rate",
            scope="europe",
            title="Mapa burbuja · alertas y alert rate",
            projection="natural earth",
            hover_data={
                "transactions": ":,.0f",
                "alerts": ":,.0f",
                "alert_rate": ":.2%",
                "importe_total": ":,.2f",
                "iso_alpha": False,
            },
        )
        bubble_fig.update_layout(height=520, margin=dict(l=0, r=0, t=60, b=0))
        st.plotly_chart(bubble_fig, use_container_width=True)

    top_country_alerts = geo_metric_df.sort_values("alerts", ascending=False).head(15)
    if not top_country_alerts.empty:
        geo1, geo2 = st.columns(2)

        with geo1:
            fig_country_alerts = px.bar(
                top_country_alerts,
                x="country",
                y="alerts",
                title="Top países por alertas",
            )
            fig_country_alerts.update_layout(
                xaxis_title="País",
                yaxis_title="Alertas",
                height=390,
            )
            st.plotly_chart(fig_country_alerts, use_container_width=True)

        with geo2:
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
                height=390,
            )
            st.plotly_chart(fig_country_rate, use_container_width=True)

    show_country_df = geo_metric_df.copy()
    if "alert_rate" in show_country_df.columns:
        show_country_df["alert_rate"] = show_country_df["alert_rate"].map(lambda x: f"{x:.2%}" if pd.notna(x) else "N/A")
    st.dataframe(show_country_df.head(25), use_container_width=True, hide_index=True)

st.markdown("---")


# =========================================================
# DOMESTIC VS CROSS-BORDER
# =========================================================

st.markdown("## Corte doméstico vs cross-border")
info_box("Compara el comportamiento relativo de tráfico doméstico e internacional. Es útil para decidir si conviene aplicar reglas más exigentes en transacciones transfronterizas.")

if cross_border_summary_filtered.empty:
    st.info("No se encontró una columna de cross-border en el dataset scored.")
else:
    cb1, cb2 = st.columns(2)

    with cb1:
        if {"cross_border_group", "alerts"}.issubset(cross_border_summary_filtered.columns):
            fig_cb_alerts = px.bar(
                cross_border_summary_filtered,
                x="cross_border_group",
                y="alerts",
                title="Alertas por grupo",
            )
            fig_cb_alerts.update_layout(
                xaxis_title="Grupo",
                yaxis_title="Alertas",
                height=360,
            )
            st.plotly_chart(fig_cb_alerts, use_container_width=True)

    with cb2:
        if {"cross_border_group", "alert_rate"}.issubset(cross_border_summary_filtered.columns):
            fig_cb_rate = px.bar(
                cross_border_summary_filtered,
                x="cross_border_group",
                y="alert_rate",
                title="Alert rate por grupo",
            )
            fig_cb_rate.update_layout(
                xaxis_title="Grupo",
                yaxis_title="Alert rate",
                yaxis_tickformat=".2%",
                height=360,
            )
            st.plotly_chart(fig_cb_rate, use_container_width=True)

    show_cross_df = cross_border_summary_filtered.copy()
    if "alert_rate" in show_cross_df.columns:
        show_cross_df["alert_rate"] = show_cross_df["alert_rate"].map(lambda x: f"{x:.2%}" if pd.notna(x) else "N/A")
    st.dataframe(show_cross_df, use_container_width=True, hide_index=True)

st.markdown("---")


# =========================================================
# MODELED RISK & ECONOMIC EXPOSURE
# =========================================================

st.markdown("## Riesgo modelizado y exposición económica")
info_box("Este bloque traduce el output del modelo a decisiones de negocio: dónde está el score alto, dónde se acumulan las alertas y en qué buckets se concentra más importe.")

score_distribution_df = build_score_distribution(scored_filtered, scored_meta["score_col"], bins=25)
amount_by_bucket_df = build_amount_by_bucket(scored_filtered, scored_meta["risk_bucket_col"], scored_meta["amount_col"])

r1, r2 = st.columns(2)

with r1:
    if not score_distribution_df.empty:
        fig_score_dist = px.bar(
            score_distribution_df,
            x="score_bin",
            y="transactions",
            title="Distribución del score de fraude",
        )
        fig_score_dist.add_vline(
            x=champion_threshold if champion_threshold is not None else 0.8987,
            line_dash="dash",
            annotation_text="Threshold",
            annotation_position="top left",
        )
        fig_score_dist.update_layout(
            xaxis_title="Intervalo de score",
            yaxis_title="Transacciones",
            height=420,
        )
        st.plotly_chart(fig_score_dist, use_container_width=True)
    else:
        st.info("No se pudo construir la distribución del score.")

with r2:
    if not amount_by_bucket_df.empty:
        fig_amount_bucket = px.treemap(
            amount_by_bucket_df,
            path=["bucket_riesgo"],
            values="importe_total",
            title="Treemap de exposición por bucket",
        )
        fig_amount_bucket.update_layout(height=420)
        st.plotly_chart(fig_amount_bucket, use_container_width=True)
    else:
        st.info("No se pudo construir el análisis de importe por bucket.")

if not risk_bucket_summary_df.empty:
    rb1, rb2 = st.columns(2)

    with rb1:
        if {"bucket_riesgo", "alerts", "alert_rate"}.issubset(risk_bucket_summary_df.columns):
            fig_bucket_alert_rate = px.bar(
                risk_bucket_summary_df,
                x="bucket_riesgo",
                y="alert_rate",
                hover_data={"alerts": ":,.0f", "transactions": ":,.0f"},
                title="Alert rate por bucket de riesgo",
            )
            fig_bucket_alert_rate.update_layout(
                xaxis_title="Bucket",
                yaxis_title="Alert rate",
                yaxis_tickformat=".2%",
                height=390,
            )
            st.plotly_chart(fig_bucket_alert_rate, use_container_width=True)

    with rb2:
        if {"bucket_riesgo", "score_medio", "alert_rate", "importe_total"}.issubset(risk_bucket_summary_df.columns):
            fig_bucket_bubble = px.scatter(
                risk_bucket_summary_df,
                x="score_medio",
                y="alert_rate",
                size="importe_total",
                color="bucket_riesgo",
                hover_name="bucket_riesgo",
                hover_data={
                    "transactions": ":,.0f",
                    "alerts": ":,.0f",
                    "importe_total": ":,.2f",
                    "score_medio": ":.4f",
                    "alert_rate": ":.2%",
                },
                title="Buckets: score medio vs alert rate vs exposición",
            )
            fig_bucket_bubble.update_layout(
                xaxis_title="Score medio",
                yaxis_title="Alert rate",
                yaxis_tickformat=".2%",
                height=390,
            )
            st.plotly_chart(fig_bucket_bubble, use_container_width=True)

    show_risk_df = risk_bucket_summary_df.copy()
    if "alert_rate" in show_risk_df.columns:
        show_risk_df["alert_rate"] = show_risk_df["alert_rate"].map(lambda x: f"{x:.2%}" if pd.notna(x) else "N/A")
    if "score_medio" in show_risk_df.columns:
        show_risk_df["score_medio"] = show_risk_df["score_medio"].map(lambda x: f"{x:.4f}" if pd.notna(x) else "N/A")
    st.dataframe(show_risk_df, use_container_width=True, hide_index=True)

st.markdown("---")


# =========================================================
# BUSINESS PATTERNS
# =========================================================

st.markdown("## Patrones de negocio")
info_box("Aquí se localizan medios de pago y categorías de comercio donde conviene reforzar reglas, controles o revisión manual.")

payment_alerts_df = build_payment_type_alerts(alert_filtered, scored_meta["payment_type_col"])
merchant_category_alerts_df = build_merchant_category_alerts(alert_filtered, scored_meta["merchant_category_col"], top_n=15)

p1, p2 = st.columns(2)

with p1:
    if not payment_alerts_df.empty:
        fig_payment = px.bar(
            payment_alerts_df,
            x="payment_type",
            y="alerts",
            title="Alertas por tipo de pago",
        )
        fig_payment.update_layout(
            xaxis_title="Tipo de pago",
            yaxis_title="Alertas",
            height=400,
        )
        st.plotly_chart(fig_payment, use_container_width=True)
    else:
        st.info("No se encontró una columna robusta de tipo de pago.")

with p2:
    if not merchant_category_alerts_df.empty:
        fig_merchant_cat = px.bar(
            merchant_category_alerts_df,
            x="merchant_category",
            y="alerts",
            title="Top categorías de comercio por alertas",
        )
        fig_merchant_cat.update_layout(
            xaxis_title="Categoría de comercio",
            yaxis_title="Alertas",
            height=400,
        )
        st.plotly_chart(fig_merchant_cat, use_container_width=True)
    else:
        st.info("No se encontró una columna robusta de categoría de comercio.")

st.markdown("---")


# =========================================================
# TEMPORAL MICRO-PATTERNS
# =========================================================

st.markdown("## Micro-patrones temporales")
info_box("Sirve para detectar franjas horarias y días de semana donde se acumula más actividad sospechosa y donde podrían tener sentido reglas temporales específicas.")

weekday_alerts_df = build_weekday_alerts(alert_filtered, scored_meta["date_col"])
hourly_alerts_df = build_hourly_alerts(alert_filtered, scored_meta["date_col"])

m1, m2 = st.columns(2)

with m1:
    if not weekday_alerts_df.empty:
        fig_weekday = px.bar(
            weekday_alerts_df,
            x="día_semana",
            y="alerts",
            title="Alertas por día de la semana",
        )
        fig_weekday.update_layout(
            xaxis_title="Día de la semana",
            yaxis_title="Alertas",
            height=390,
        )
        st.plotly_chart(fig_weekday, use_container_width=True)
    else:
        st.info("No fue posible construir el patrón semanal.")

with m2:
    if not hourly_alerts_df.empty:
        fig_hour = px.line(
            hourly_alerts_df,
            x="hour",
            y="alerts",
            markers=True,
            title="Alertas por hora del día",
        )
        fig_hour.update_layout(
            xaxis_title="Hora",
            yaxis_title="Alertas",
            height=390,
        )
        st.plotly_chart(fig_hour, use_container_width=True)
    else:
        st.info("No fue posible construir el patrón horario.")

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

    st.markdown("### dashboard_scored_transactions")
    st.dataframe(scored_filtered.head(20), use_container_width=True, hide_index=True)

    st.markdown("### alert_queue")
    st.dataframe(alert_filtered.head(20), use_container_width=True, hide_index=True)
