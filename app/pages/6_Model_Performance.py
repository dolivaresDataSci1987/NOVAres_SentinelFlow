from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))


# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="Model Performance | SentinelFlow",
    page_icon="🎯",
    layout="wide",
)


# =========================================================
# HELPERS - LOADING
# =========================================================
def _safe_read_csv(path: Path) -> Optional[pd.DataFrame]:
    try:
        if path.exists() and path.stat().st_size > 0:
            return pd.read_csv(path)
    except Exception:
        return None
    return None


@st.cache_data(show_spinner=False)
def load_dashboard_data() -> pd.DataFrame:
    exports_dir = ROOT / "artifacts" / "dashboard_exports"

    preferred_files = [
        "dashboard_scored_transactions.csv",
        "scored_transactions_dashboard.csv",
        "transactions_scored_dashboard.csv",
        "transaction_monitoring_base.csv",
        "alerts_dashboard.csv",
    ]

    for fname in preferred_files:
        df = _safe_read_csv(exports_dir / fname)
        if df is not None and not df.empty:
            return df

    if exports_dir.exists():
        for csv_path in sorted(exports_dir.glob("*.csv")):
            df = _safe_read_csv(csv_path)
            if df is None or df.empty:
                continue

            cols = [c.lower() for c in df.columns]
            has_score = any(
                x in cols
                for x in [
                    "risk_score",
                    "fraud_score",
                    "score",
                    "predicted_probability",
                    "fraud_probability",
                    "model_score",
                    "risk_probability",
                    "prediction_score",
                ]
            )
            has_label = any(
                x in cols
                for x in [
                    "fraud_label",
                    "is_fraud",
                    "label",
                    "target",
                    "y_true",
                    "actual_fraud",
                ]
            )
            if has_score and has_label:
                return df

    return pd.DataFrame()


# =========================================================
# HELPERS - SCHEMA
# =========================================================
def find_first_existing(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    cols_lower = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in cols_lower:
            return cols_lower[cand.lower()]
    return None


def infer_columns(df: pd.DataFrame) -> Dict[str, Optional[str]]:
    return {
        "label": find_first_existing(
            df,
            [
                "fraud_label",
                "is_fraud",
                "label",
                "target",
                "y_true",
                "actual_fraud",
                "fraud_flag",
            ],
        ),
        "score": find_first_existing(
            df,
            [
                "risk_score",
                "fraud_score",
                "score",
                "predicted_probability",
                "fraud_probability",
                "model_score",
                "risk_probability",
                "prediction_score",
                "fraud_prob",
                "pred_proba",
            ],
        ),
        "pred": find_first_existing(
            df,
            [
                "fraud_prediction",
                "predicted_label",
                "prediction",
                "is_alert",
                "alert_flag",
                "model_prediction",
            ],
        ),
        "timestamp": find_first_existing(
            df,
            [
                "transaction_ts",
                "transaction_timestamp",
                "event_ts",
                "timestamp",
                "created_at",
                "transaction_date",
                "event_date",
                "date",
            ],
        ),
        "channel": find_first_existing(
            df,
            [
                "channel",
                "payment_channel",
                "transaction_channel",
                "entry_channel",
            ],
        ),
        "country": find_first_existing(
            df,
            [
                "country",
                "merchant_country",
                "transaction_country",
                "issuer_country",
                "billing_country",
                "country_code",
            ],
        ),
        "risk_bucket": find_first_existing(
            df,
            [
                "risk_bucket",
                "risk_band",
                "score_band",
                "risk_segment",
                "alert_priority",
            ],
        ),
        "amount": find_first_existing(
            df,
            [
                "amount",
                "transaction_amount",
                "amount_eur",
                "amount_usd",
                "authorized_amount",
                "txn_amount",
            ],
        ),
    }


def coerce_binary_label(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        out = pd.to_numeric(series, errors="coerce").fillna(0)
        return (out > 0).astype(int)

    s = series.astype(str).str.strip().str.lower()
    mapping_true = {"1", "true", "yes", "y", "fraud", "positive", "positivo", "si", "sí"}
    return s.isin(mapping_true).astype(int)


def coerce_score(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    s = s.replace([np.inf, -np.inf], np.nan)

    if s.notna().sum() == 0:
        return s

    q99 = s.quantile(0.99)
    if pd.notna(q99) and q99 > 1.5 and q99 <= 100:
        s = s / 100.0

    return s.clip(lower=0, upper=1)


def derive_prediction_from_score(score: pd.Series, threshold: float) -> pd.Series:
    return (score >= threshold).astype(int)


def ensure_datetime(df: pd.DataFrame, col: Optional[str]) -> pd.Series:
    if col is None or col not in df.columns:
        return pd.Series(pd.NaT, index=df.index)
    return pd.to_datetime(df[col], errors="coerce")


def add_score_bands(df: pd.DataFrame, score_col: str) -> pd.DataFrame:
    out = df.copy()
    bins = [-0.001, 0.1, 0.2, 0.4, 0.6, 0.8, 0.9, 1.0]
    labels = [
        "0.00–0.10",
        "0.10–0.20",
        "0.20–0.40",
        "0.40–0.60",
        "0.60–0.80",
        "0.80–0.90",
        "0.90–1.00",
    ]
    out["score_band_auto"] = pd.cut(
        out[score_col],
        bins=bins,
        labels=labels,
        include_lowest=True,
        ordered=True,
    )
    return out


def add_score_deciles(df: pd.DataFrame, score_col: str) -> pd.DataFrame:
    out = df.copy()
    ranked = out[score_col].rank(method="first")
    try:
        out["score_decile"] = pd.qcut(ranked, 10, labels=[f"D{i}" for i in range(1, 11)])
    except Exception:
        out["score_decile"] = pd.cut(
            out[score_col],
            bins=np.linspace(0, 1, 11),
            labels=[f"D{i}" for i in range(1, 11)],
            include_lowest=True,
        )
    return out


def format_pct(x: Optional[float]) -> str:
    if x is None or pd.isna(x):
        return "N/A"
    return f"{x:.1%}"


def format_num(x: Optional[float]) -> str:
    if x is None or pd.isna(x):
        return "N/A"
    return f"{x:,.0f}"


# =========================================================
# METRICS WITHOUT SKLEARN
# =========================================================
def confusion_counts(y_true: pd.Series, y_pred: pd.Series) -> Dict[str, int]:
    y_true = y_true.astype(int).to_numpy()
    y_pred = y_pred.astype(int).to_numpy()

    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())

    return {"tp": tp, "fp": fp, "tn": tn, "fn": fn}


def compute_confusion_metrics(y_true: pd.Series, y_pred: pd.Series) -> Dict[str, float]:
    if len(y_true) == 0:
        return {
            "precision": np.nan,
            "recall": np.nan,
            "f1": np.nan,
            "fpr": np.nan,
            "alert_rate": np.nan,
            "fraud_capture_rate": np.nan,
            "tp": 0,
            "fp": 0,
            "tn": 0,
            "fn": 0,
        }

    c = confusion_counts(y_true, y_pred)
    tp, fp, tn, fn = c["tp"], c["fp"], c["tn"], c["fn"]

    precision = tp / (tp + fp) if (tp + fp) > 0 else np.nan
    recall = tp / (tp + fn) if (tp + fn) > 0 else np.nan
    f1 = (
        2 * precision * recall / (precision + recall)
        if pd.notna(precision) and pd.notna(recall) and (precision + recall) > 0
        else np.nan
    )
    fpr = fp / (fp + tn) if (fp + tn) > 0 else np.nan
    alert_rate = (tp + fp) / len(y_true) if len(y_true) > 0 else np.nan

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "fpr": fpr,
        "alert_rate": alert_rate,
        "fraud_capture_rate": recall,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
    }


def _prepare_binary_arrays(y_true: pd.Series, score: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    valid = ~(y_true.isna() | score.isna())
    y = y_true[valid].astype(int).to_numpy()
    s = score[valid].astype(float).to_numpy()
    return y, s


def roc_curve_manual(y_true: pd.Series, score: pd.Series) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    y, s = _prepare_binary_arrays(y_true, score)

    if len(y) == 0 or np.unique(y).size < 2:
        return np.array([]), np.array([]), np.array([])

    thresholds = np.unique(s)[::-1]
    thresholds = np.r_[np.inf, thresholds, -np.inf]

    tpr_list = []
    fpr_list = []

    pos = (y == 1).sum()
    neg = (y == 0).sum()

    for thr in thresholds:
        pred = (s >= thr).astype(int)
        c = confusion_counts(pd.Series(y), pd.Series(pred))
        tpr = c["tp"] / pos if pos > 0 else np.nan
        fpr = c["fp"] / neg if neg > 0 else np.nan
        tpr_list.append(tpr)
        fpr_list.append(fpr)

    return np.array(fpr_list), np.array(tpr_list), thresholds


def precision_recall_curve_manual(y_true: pd.Series, score: pd.Series) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    y, s = _prepare_binary_arrays(y_true, score)

    if len(y) == 0 or np.unique(y).size < 2:
        return np.array([]), np.array([]), np.array([])

    thresholds = np.unique(s)[::-1]

    precision_list = []
    recall_list = []

    pos = (y == 1).sum()

    for thr in thresholds:
        pred = (s >= thr).astype(int)
        c = confusion_counts(pd.Series(y), pd.Series(pred))
        precision = c["tp"] / (c["tp"] + c["fp"]) if (c["tp"] + c["fp"]) > 0 else 1.0
        recall = c["tp"] / pos if pos > 0 else np.nan
        precision_list.append(precision)
        recall_list.append(recall)

    precision_arr = np.array(precision_list)
    recall_arr = np.array(recall_list)

    order = np.argsort(recall_arr)
    return precision_arr[order], recall_arr[order], thresholds


def auc_trapezoid(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2 or len(y) < 2:
        return np.nan
    valid = ~(np.isnan(x) | np.isnan(y))
    x = x[valid]
    y = y[valid]
    if len(x) < 2:
        return np.nan
    return float(np.trapz(y, x))


def roc_auc_manual(y_true: pd.Series, score: pd.Series) -> float:
    fpr, tpr, _ = roc_curve_manual(y_true, score)
    return auc_trapezoid(fpr, tpr)


def pr_auc_manual(y_true: pd.Series, score: pd.Series) -> float:
    precision, recall, _ = precision_recall_curve_manual(y_true, score)
    return auc_trapezoid(recall, precision)


def compute_auc_metrics(y_true: pd.Series, score: pd.Series) -> Dict[str, float]:
    valid = ~(y_true.isna() | score.isna())
    y = y_true[valid]
    s = score[valid]

    if len(y) == 0:
        return {"roc_auc": np.nan, "pr_auc": np.nan, "base_rate": np.nan}

    if y.nunique() < 2:
        return {"roc_auc": np.nan, "pr_auc": np.nan, "base_rate": y.mean()}

    return {
        "roc_auc": roc_auc_manual(y, s),
        "pr_auc": pr_auc_manual(y, s),
        "base_rate": y.mean(),
    }


def build_threshold_table(
    y_true: pd.Series,
    score: pd.Series,
    thresholds: Optional[List[float]] = None,
) -> pd.DataFrame:
    valid = ~(y_true.isna() | score.isna())
    y = y_true[valid].astype(int)
    s = score[valid].astype(float)

    if thresholds is None:
        thresholds = list(np.round(np.arange(0.05, 1.00, 0.05), 2))

    rows = []
    total_fraud = int(y.sum())
    total_n = int(len(y))

    for thr in thresholds:
        pred = (s >= thr).astype(int)
        m = compute_confusion_metrics(y, pred)

        rows.append(
            {
                "threshold": thr,
                "alertas": int(pred.sum()),
                "alert_rate": m["alert_rate"],
                "precision": m["precision"],
                "recall": m["recall"],
                "f1": m["f1"],
                "false_positive_rate": m["fpr"],
                "fraud_capture_rate": m["fraud_capture_rate"],
                "fraudes_capturados": int(m["tp"]),
                "fraudes_totales": total_fraud,
                "observaciones": total_n,
            }
        )

    return pd.DataFrame(rows)


def build_lift_table(df: pd.DataFrame, label_col: str, score_col: str) -> pd.DataFrame:
    tmp = df[[label_col, score_col]].dropna().copy()
    if tmp.empty:
        return pd.DataFrame()

    tmp = tmp.sort_values(score_col, ascending=False).reset_index(drop=True)
    n = len(tmp)
    base_fraud_rate = tmp[label_col].mean()

    bands = [
        ("Top 1%", max(1, int(n * 0.01))),
        ("Top 5%", max(1, int(n * 0.05))),
        ("Top 10%", max(1, int(n * 0.10))),
        ("Top 20%", max(1, int(n * 0.20))),
    ]

    rows = []
    for label, k in bands:
        subset = tmp.head(k)
        fraud_rate = subset[label_col].mean()
        lift = fraud_rate / base_fraud_rate if base_fraud_rate > 0 else np.nan
        captured = subset[label_col].sum()
        total_fraud = tmp[label_col].sum()
        capture = captured / total_fraud if total_fraud > 0 else np.nan

        rows.append(
            {
                "banda": label,
                "n_obs": k,
                "fraud_rate": fraud_rate,
                "lift": lift,
                "fraud_capture_rate": capture,
                "fraudes_capturados": int(captured),
            }
        )

    return pd.DataFrame(rows)


def build_temporal_metrics(
    df: pd.DataFrame,
    date_col: str,
    label_col: str,
    score_col: str,
    threshold: float,
    freq: str = "W",
) -> pd.DataFrame:
    tmp = df[[date_col, label_col, score_col]].dropna().copy()
    if tmp.empty:
        return pd.DataFrame()

    tmp["period"] = pd.to_datetime(tmp[date_col], errors="coerce").dt.to_period(freq).dt.to_timestamp()
    tmp = tmp.dropna(subset=["period"])

    rows = []
    for period, g in tmp.groupby("period"):
        y = g[label_col].astype(int)
        s = g[score_col].astype(float)
        pred = (s >= threshold).astype(int)

        conf = compute_confusion_metrics(y, pred)
        aucs = compute_auc_metrics(y, s)

        rows.append(
            {
                "period": period,
                "n": len(g),
                "fraud_rate_real": y.mean(),
                "score_medio": s.mean(),
                "roc_auc": aucs["roc_auc"],
                "pr_auc": aucs["pr_auc"],
                "precision": conf["precision"],
                "recall": conf["recall"],
                "f1": conf["f1"],
                "alert_rate": conf["alert_rate"],
                "fraud_capture_rate": conf["fraud_capture_rate"],
            }
        )

    return pd.DataFrame(rows).sort_values("period")


def build_segment_performance(
    df: pd.DataFrame,
    segment_col: str,
    label_col: str,
    score_col: str,
    threshold: float,
    min_rows: int = 200,
) -> pd.DataFrame:
    tmp = df[[segment_col, label_col, score_col]].dropna().copy()
    if tmp.empty:
        return pd.DataFrame()

    rows = []
    for seg, g in tmp.groupby(segment_col):
        if len(g) < min_rows:
            continue

        y = g[label_col].astype(int)
        s = g[score_col].astype(float)
        pred = (s >= threshold).astype(int)

        conf = compute_confusion_metrics(y, pred)
        aucs = compute_auc_metrics(y, s)

        rows.append(
            {
                "segmento": str(seg),
                "n": len(g),
                "fraud_rate_real": y.mean(),
                "score_medio": s.mean(),
                "roc_auc": aucs["roc_auc"],
                "precision": conf["precision"],
                "recall": conf["recall"],
                "f1": conf["f1"],
                "alert_rate": conf["alert_rate"],
                "fraud_capture_rate": conf["fraud_capture_rate"],
            }
        )

    return pd.DataFrame(rows).sort_values(["fraud_capture_rate", "precision"], ascending=False)


def detect_deterioration_signals(
    temporal_df: pd.DataFrame,
    segment_df: Optional[pd.DataFrame] = None,
) -> List[str]:
    signals = []

    if temporal_df is not None and not temporal_df.empty and len(temporal_df) >= 4:
        recent = temporal_df.tail(4).copy()

        if recent["roc_auc"].notna().sum() >= 2:
            start_auc = recent["roc_auc"].iloc[0]
            end_auc = recent["roc_auc"].iloc[-1]
            if pd.notna(start_auc) and pd.notna(end_auc) and end_auc < start_auc - 0.05:
                signals.append("Caída reciente del AUC ROC superior a 5 puntos: posible pérdida de capacidad discriminativa.")

        if recent["precision"].notna().sum() >= 2:
            start_p = recent["precision"].iloc[0]
            end_p = recent["precision"].iloc[-1]
            if pd.notna(start_p) and pd.notna(end_p) and end_p < start_p - 0.05:
                signals.append("Descenso reciente de precision: el modelo podría estar generando más ruido operativo.")

        if recent["alert_rate"].notna().sum() >= 2:
            start_a = recent["alert_rate"].iloc[0]
            end_a = recent["alert_rate"].iloc[-1]
            if pd.notna(start_a) and pd.notna(end_a) and end_a > start_a + 0.05:
                signals.append("Subida del alert rate: la carga enviada a revisión manual está creciendo.")

        if recent["score_medio"].notna().sum() >= 2:
            std_recent = recent["score_medio"].std()
            if pd.notna(std_recent) and std_recent > 0.08:
                signals.append("Variabilidad relevante en score medio por periodo: revisar estabilidad del input o drift del tráfico.")

    if segment_df is not None and not segment_df.empty:
        low_perf = segment_df[
            (segment_df["roc_auc"].notna()) & (segment_df["roc_auc"] < 0.60)
        ]
        if len(low_perf) > 0:
            signals.append("Hay segmentos con AUC ROC bajo (<0.60): el modelo no separa bien en todas las poblaciones.")

        spread_precision = (
            segment_df["precision"].max() - segment_df["precision"].min()
            if segment_df["precision"].notna().sum() >= 2
            else np.nan
        )
        if pd.notna(spread_precision) and spread_precision > 0.20:
            signals.append("Gran dispersión de precision entre segmentos: conviene revisar thresholds o tratamiento segmentado.")

    if not signals:
        signals.append("No se observan señales fuertes de deterioro con los indicadores disponibles en esta muestra.")

    return signals


# =========================================================
# PLOTS
# =========================================================
def plot_score_distribution(df: pd.DataFrame, score_col: str) -> go.Figure:
    fig = px.histogram(
        df,
        x=score_col,
        nbins=40,
        title="Distribución global del score",
        labels={score_col: "Score del modelo"},
    )
    fig.update_layout(height=350, margin=dict(l=10, r=10, t=60, b=10))
    return fig


def plot_score_by_class(df: pd.DataFrame, score_col: str, label_col: str) -> go.Figure:
    tmp = df[[score_col, label_col]].dropna().copy()
    tmp["Clase real"] = tmp[label_col].map({0: "No fraude", 1: "Fraude"})
    fig = px.box(
        tmp,
        x="Clase real",
        y=score_col,
        points=False,
        title="Separación del score por clase real",
        labels={score_col: "Score del modelo"},
    )
    fig.update_layout(height=350, margin=dict(l=10, r=10, t=60, b=10))
    return fig


def plot_fraud_by_decile(df: pd.DataFrame, label_col: str, score_col: str) -> go.Figure:
    tmp = add_score_deciles(df[[label_col, score_col]].dropna().copy(), score_col)
    if tmp.empty:
        return go.Figure()

    dec = (
        tmp.groupby("score_decile", observed=False)
        .agg(
            observaciones=(label_col, "size"),
            fraudes=(label_col, "sum"),
            fraud_rate=(label_col, "mean"),
        )
        .reset_index()
    )

    order_map = {f"D{i}": i for i in range(1, 11)}
    dec["sort_key"] = dec["score_decile"].astype(str).map(order_map)
    dec = dec.sort_values("sort_key", ascending=False)

    fig = px.bar(
        dec,
        x="score_decile",
        y="fraud_rate",
        title="Fraude real por decil de score",
        labels={"score_decile": "Decil de score", "fraud_rate": "Tasa real de fraude"},
        text_auto=".1%",
    )
    fig.update_layout(height=350, margin=dict(l=10, r=10, t=60, b=10))
    return fig


def plot_roc_curve(y_true: pd.Series, score: pd.Series) -> go.Figure:
    fpr, tpr, _ = roc_curve_manual(y_true, score)

    fig = go.Figure()
    if len(fpr) == 0 or len(tpr) == 0:
        fig.update_layout(title="Curva ROC no disponible")
        return fig

    auc_value = auc_trapezoid(fpr, tpr)

    fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"Modelo (AUC={auc_value:.3f})"))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Azar", line=dict(dash="dash")))

    fig.update_layout(
        title="Curva ROC",
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        height=350,
        margin=dict(l=10, r=10, t=60, b=10),
    )
    return fig


def plot_pr_curve(y_true: pd.Series, score: pd.Series) -> go.Figure:
    precision, recall, _ = precision_recall_curve_manual(y_true, score)

    fig = go.Figure()
    if len(precision) == 0 or len(recall) == 0:
        fig.update_layout(title="Curva Precision-Recall no disponible")
        return fig

    base_rate = y_true.mean() if len(y_true) > 0 else np.nan
    pr_auc = auc_trapezoid(recall, precision)

    fig.add_trace(go.Scatter(x=recall, y=precision, mode="lines", name=f"Modelo (AUC={pr_auc:.3f})"))
    if pd.notna(base_rate):
        fig.add_trace(
            go.Scatter(
                x=[0, 1],
                y=[base_rate, base_rate],
                mode="lines",
                name="Base rate",
                line=dict(dash="dash"),
            )
        )

    fig.update_layout(
        title="Curva Precision-Recall",
        xaxis_title="Recall",
        yaxis_title="Precision",
        height=350,
        margin=dict(l=10, r=10, t=60, b=10),
    )
    return fig


def plot_threshold_tradeoff(threshold_df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    if threshold_df.empty:
        fig.update_layout(title="Trade-off por threshold no disponible")
        return fig

    fig.add_trace(go.Scatter(
        x=threshold_df["threshold"],
        y=threshold_df["precision"],
        mode="lines+markers",
        name="Precision",
    ))
    fig.add_trace(go.Scatter(
        x=threshold_df["threshold"],
        y=threshold_df["recall"],
        mode="lines+markers",
        name="Recall",
    ))
    fig.add_trace(go.Scatter(
        x=threshold_df["threshold"],
        y=threshold_df["alert_rate"],
        mode="lines+markers",
        name="Alert rate",
    ))

    fig.update_layout(
        title="Trade-off entre threshold, precision, recall y volumen de alertas",
        xaxis_title="Threshold",
        yaxis_title="Métrica",
        height=380,
        margin=dict(l=10, r=10, t=60, b=10),
    )
    return fig


def plot_temporal_performance(temporal_df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    if temporal_df.empty:
        fig.update_layout(title="Evolución temporal no disponible")
        return fig

    fig.add_trace(go.Scatter(
        x=temporal_df["period"],
        y=temporal_df["precision"],
        mode="lines+markers",
        name="Precision",
    ))
    fig.add_trace(go.Scatter(
        x=temporal_df["period"],
        y=temporal_df["recall"],
        mode="lines+markers",
        name="Recall",
    ))
    fig.add_trace(go.Scatter(
        x=temporal_df["period"],
        y=temporal_df["alert_rate"],
        mode="lines+markers",
        name="Alert rate",
    ))
    fig.add_trace(go.Scatter(
        x=temporal_df["period"],
        y=temporal_df["fraud_capture_rate"],
        mode="lines+markers",
        name="Fraud capture",
    ))

    fig.update_layout(
        title="Evolución temporal de métricas operativas",
        xaxis_title="Periodo",
        yaxis_title="Valor",
        height=380,
        margin=dict(l=10, r=10, t=60, b=10),
    )
    return fig


def plot_temporal_stability(temporal_df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    if temporal_df.empty:
        fig.update_layout(title="Estabilidad temporal no disponible")
        return fig

    fig.add_trace(go.Scatter(
        x=temporal_df["period"],
        y=temporal_df["score_medio"],
        mode="lines+markers",
        name="Score medio",
    ))
    fig.add_trace(go.Scatter(
        x=temporal_df["period"],
        y=temporal_df["fraud_rate_real"],
        mode="lines+markers",
        name="Fraud rate real",
    ))
    fig.add_trace(go.Scatter(
        x=temporal_df["period"],
        y=temporal_df["roc_auc"],
        mode="lines+markers",
        name="ROC AUC",
    ))

    fig.update_layout(
        title="Estabilidad temporal del score y la discriminación",
        xaxis_title="Periodo",
        yaxis_title="Valor",
        height=380,
        margin=dict(l=10, r=10, t=60, b=10),
    )
    return fig


def plot_segment_comparison(segment_df: pd.DataFrame, x_col: str = "segmento") -> go.Figure:
    fig = go.Figure()
    if segment_df.empty:
        fig.update_layout(title="Comparación segmentada no disponible")
        return fig

    fig = px.bar(
        segment_df.sort_values("fraud_capture_rate", ascending=False),
        x=x_col,
        y="fraud_capture_rate",
        hover_data=["n", "roc_auc", "precision", "recall", "alert_rate"],
        title=f"Fraud capture por {x_col}",
        labels={x_col: x_col.capitalize(), "fraud_capture_rate": "Fraud capture rate"},
    )
    fig.update_layout(height=360, margin=dict(l=10, r=10, t=60, b=10))
    return fig


# =========================================================
# UI HELPERS
# =========================================================
def info_box(text: str) -> None:
    st.markdown(
        f"""
        <div style="
            background:#F6F8FB;
            border:1px solid #E5EAF2;
            padding:12px 14px;
            border-radius:10px;
            margin-bottom:8px;
            color:#22303C;
            font-size:0.95rem;">
            {text}
        </div>
        """,
        unsafe_allow_html=True,
    )


def metric_card(label: str, value: str, help_text: str = "") -> None:
    st.metric(label, value)
    if help_text:
        st.caption(help_text)


# =========================================================
# MAIN
# =========================================================
df_raw = load_dashboard_data()

st.title("Model Performance")
st.caption("Seguimiento operativo del rendimiento del modelo champion de fraude en producción.")

if df_raw.empty:
    st.error(
        "No se encontró un dataset válido en artifacts/dashboard_exports/ con columnas de score y label."
    )
    st.stop()

schema = infer_columns(df_raw)

if schema["label"] is None or schema["score"] is None:
    st.error(
        "El dataset cargado no contiene las columnas mínimas necesarias para esta página: label y score."
    )
    st.write("Columnas detectadas:", list(df_raw.columns))
    st.stop()

df = df_raw.copy()
df["__label__"] = coerce_binary_label(df[schema["label"]])
df["__score__"] = coerce_score(df[schema["score"]])
df["__timestamp__"] = ensure_datetime(df, schema["timestamp"])

df = df.dropna(subset=["__label__", "__score__"]).copy()

if df.empty:
    st.error("Tras la limpieza no quedan registros utilizables para evaluar el modelo.")
    st.stop()

if schema["risk_bucket"] is None:
    df = add_score_bands(df, "__score__")
    risk_bucket_col = "score_band_auto"
else:
    risk_bucket_col = schema["risk_bucket"]

st.sidebar.header("Filtros")

if df["__timestamp__"].notna().sum() > 0:
    min_date = df["__timestamp__"].min().date()
    max_date = df["__timestamp__"].max().date()
    date_range = st.sidebar.date_input(
        "Rango temporal",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date,
    )
    if isinstance(date_range, tuple) and len(date_range) == 2:
        start_date, end_date = date_range
        df = df[
            (df["__timestamp__"].dt.date >= start_date)
            & (df["__timestamp__"].dt.date <= end_date)
        ].copy()

if schema["channel"] is not None:
    channel_values = sorted([x for x in df[schema["channel"]].dropna().astype(str).unique()])
    selected_channels = st.sidebar.multiselect(
        "Canal",
        options=channel_values,
        default=channel_values,
    )
    if selected_channels:
        df = df[df[schema["channel"]].astype(str).isin(selected_channels)].copy()

if schema["country"] is not None:
    country_values = sorted([x for x in df[schema["country"]].dropna().astype(str).unique()])
    selected_countries = st.sidebar.multiselect(
        "País",
        options=country_values,
        default=country_values[:15] if len(country_values) > 15 else country_values,
    )
    if selected_countries:
        df = df[df[schema["country"]].astype(str).isin(selected_countries)].copy()

threshold = st.sidebar.slider(
    "Threshold operativo",
    min_value=0.05,
    max_value=0.95,
    value=0.70,
    step=0.05,
)

freq_label = st.sidebar.selectbox(
    "Granularidad temporal",
    options=["Semanal", "Mensual"],
    index=0,
)
freq_map = {"Semanal": "W", "Mensual": "M"}
freq = freq_map[freq_label]

df["__pred__"] = derive_prediction_from_score(df["__score__"], threshold)

if len(df) == 0:
    st.warning("No hay datos después de aplicar los filtros.")
    st.stop()

st.markdown("## 1) Resumen ejecutivo")
info_box(
    "Este bloque resume si el modelo está aportando valor operativo: qué capacidad tiene para separar fraude de no fraude, "
    "qué parte del fraude consigue capturar y cuánto ruido envía a revisión manual."
)

auc_metrics = compute_auc_metrics(df["__label__"], df["__score__"])
conf_metrics = compute_confusion_metrics(df["__label__"], df["__pred__"])
lift_df = build_lift_table(df, "__label__", "__score__")

total_alerts = int(df["__pred__"].sum())
total_fraud = int(df["__label__"].sum())
captured_fraud = int(((df["__pred__"] == 1) & (df["__label__"] == 1)).sum())

c1, c2, c3, c4 = st.columns(4)
with c1:
    metric_card("AUC ROC", f"{auc_metrics['roc_auc']:.3f}" if pd.notna(auc_metrics["roc_auc"]) else "N/A",
                "Capacidad global de discriminación del score.")
with c2:
    metric_card("Precision", format_pct(conf_metrics["precision"]),
                "De las alertas generadas, qué proporción termina siendo fraude real.")
with c3:
    metric_card("Recall", format_pct(conf_metrics["recall"]),
                "Del fraude real observado, qué proporción consigue capturar el modelo.")
with c4:
    metric_card("F1", f"{conf_metrics['f1']:.3f}" if pd.notna(conf_metrics["f1"]) else "N/A",
                "Equilibrio entre precision y recall.")

c5, c6, c7, c8 = st.columns(4)
with c5:
    metric_card("Fraud Capture Rate", format_pct(conf_metrics["fraud_capture_rate"]),
                "Equivale al recall en esta vista operativa.")
with c6:
    metric_card("False Positive Rate", format_pct(conf_metrics["fpr"]),
                "Parte del no fraude que está entrando como alerta.")
with c7:
    metric_card("Alert Rate", format_pct(conf_metrics["alert_rate"]),
                "Porcentaje del tráfico total que se envía a revisión.")
with c8:
    metric_card("Fraudes capturados", f"{captured_fraud:,} / {total_fraud:,}",
                "Fraude real recuperado al threshold seleccionado.")

c9, c10, c11 = st.columns(3)
with c9:
    metric_card("Alertas generadas", format_num(total_alerts),
                "Volumen operativo que impacta directamente en Alert Queue y Review Capacity.")
with c10:
    metric_card("PR AUC", f"{auc_metrics['pr_auc']:.3f}" if pd.notna(auc_metrics["pr_auc"]) else "N/A",
                "Más útil que ROC cuando el fraude es muy minoritario.")
with c11:
    top10_capture = lift_df.loc[lift_df["banda"] == "Top 10%", "fraud_capture_rate"]
    metric_card(
        "Fraude en Top 10%",
        format_pct(top10_capture.iloc[0]) if len(top10_capture) else "N/A",
        "Cuánto fraude total se concentra en el 10% superior del score."
    )

st.markdown("## 2) Calidad de separación del score")
info_box(
    "Aquí se observa si el score realmente ordena bien el riesgo. "
    "Un buen modelo concentra más fraude en las bandas altas y separa visualmente fraude vs no fraude."
)

col_a, col_b = st.columns(2)
with col_a:
    st.plotly_chart(plot_score_distribution(df, "__score__"), use_container_width=True)
    st.caption("Lectura rápida: muestra cómo se reparte el score en toda la cartera.")
with col_b:
    st.plotly_chart(plot_score_by_class(df, "__score__", "__label__"), use_container_width=True)
    st.caption("Lectura rápida: cuanto más separadas estén ambas clases, mejor discrimina el modelo.")

col_c, col_d = st.columns(2)
with col_c:
    st.plotly_chart(plot_fraud_by_decile(df, "__label__", "__score__"), use_container_width=True)
    st.caption("Lectura rápida: el fraude real debería concentrarse en los deciles más altos del score.")
with col_d:
    if not lift_df.empty:
        lift_show = lift_df.copy()
        lift_show["fraud_rate"] = lift_show["fraud_rate"].map(lambda x: f"{x:.1%}" if pd.notna(x) else "N/A")
        lift_show["fraud_capture_rate"] = lift_show["fraud_capture_rate"].map(lambda x: f"{x:.1%}" if pd.notna(x) else "N/A")
        lift_show["lift"] = lift_show["lift"].map(lambda x: f"{x:.2f}x" if pd.notna(x) else "N/A")
        st.dataframe(lift_show, use_container_width=True, hide_index=True)
    else:
        st.info("No fue posible calcular lift / concentración.")
    st.caption("Lectura rápida: ayuda a entender cuánto valor genera priorizar las alertas por score.")

st.markdown("## 3) Curvas de rendimiento y trade-off operativo")
info_box(
    "Este bloque ayuda a decidir dónde poner el threshold. "
    "Subirlo suele mejorar precision pero reduce recall; bajarlo captura más fraude pero incrementa carga y ruido."
)

col_e, col_f = st.columns(2)
with col_e:
    st.plotly_chart(plot_roc_curve(df["__label__"], df["__score__"]), use_container_width=True)
    st.caption("Lectura rápida: resume la capacidad discriminativa global del modelo.")
with col_f:
    st.plotly_chart(plot_pr_curve(df["__label__"], df["__score__"]), use_container_width=True)
    st.caption("Lectura rápida: especialmente útil en fraude, donde la clase positiva suele ser muy minoritaria.")

threshold_df = build_threshold_table(df["__label__"], df["__score__"])

st.plotly_chart(plot_threshold_tradeoff(threshold_df), use_container_width=True)
st.caption("Lectura rápida: permite ver el equilibrio entre calidad de la alerta, captura y carga operativa.")

st.markdown("### Thresholds sugeridos")
if not threshold_df.empty:
    thr_show = threshold_df.copy()
    for col in ["alert_rate", "precision", "recall", "f1", "false_positive_rate", "fraud_capture_rate"]:
        thr_show[col] = thr_show[col].map(lambda x: f"{x:.1%}" if pd.notna(x) else "N/A")

    best_f1_row = threshold_df.loc[threshold_df["f1"].idxmax()] if threshold_df["f1"].notna().any() else None
    best_precision_row = threshold_df[threshold_df["alertas"] > 0].sort_values(["precision", "recall"], ascending=False).head(1)
    best_recall_row = threshold_df.sort_values(["recall", "precision"], ascending=False).head(1)

    s1, s2, s3 = st.columns(3)
    with s1:
        if best_recall_row is not None and not best_recall_row.empty:
            r = best_recall_row.iloc[0]
            st.success(
                f"**Objetivo: capturar más fraude**  \nThreshold orientativo: **{r['threshold']:.2f}**  \nRecall: **{r['recall']:.1%}** | Alert rate: **{r['alert_rate']:.1%}**"
            )
    with s2:
        if best_f1_row is not None:
            st.info(
                f"**Objetivo: equilibrio calidad-volumen**  \nThreshold orientativo: **{best_f1_row['threshold']:.2f}**  \nF1: **{best_f1_row['f1']:.3f}** | Precision: **{best_f1_row['precision']:.1%}**"
            )
    with s3:
        if best_precision_row is not None and not best_precision_row.empty:
            p = best_precision_row.iloc[0]
            st.warning(
                f"**Objetivo: reducir ruido operativo**  \nThreshold orientativo: **{p['threshold']:.2f}**  \nPrecision: **{p['precision']:.1%}** | Alert rate: **{p['alert_rate']:.1%}**"
            )

    st.dataframe(
        thr_show[
            [
                "threshold",
                "alertas",
                "alert_rate",
                "precision",
                "recall",
                "f1",
                "false_positive_rate",
                "fraud_capture_rate",
                "fraudes_capturados",
            ]
        ],
        use_container_width=True,
        hide_index=True,
    )

st.markdown("## 4) Rendimiento y estabilidad en el tiempo")

if df["__timestamp__"].notna().sum() > 0:
    info_box(
        "Un modelo útil no solo debe rendir bien en promedio; también debe mantenerse estable. "
        "Este bloque ayuda a detectar degradación, cambios en el tráfico o pérdida de capacidad discriminativa."
    )

    temporal_df = build_temporal_metrics(
        df=df,
        date_col="__timestamp__",
        label_col="__label__",
        score_col="__score__",
        threshold=threshold,
        freq=freq,
    )

    col_g, col_h = st.columns(2)
    with col_g:
        st.plotly_chart(plot_temporal_performance(temporal_df), use_container_width=True)
        st.caption("Lectura rápida: muestra cómo cambian precision, recall y carga de alertas por periodo.")
    with col_h:
        st.plotly_chart(plot_temporal_stability(temporal_df), use_container_width=True)
        st.caption("Lectura rápida: permite detectar cambios anómalos en score medio, fraude real o AUC.")

    if not temporal_df.empty:
        temp_show = temporal_df.copy()
        for col in ["fraud_rate_real", "score_medio", "roc_auc", "pr_auc", "precision", "recall", "f1", "alert_rate", "fraud_capture_rate"]:
            if col in temp_show.columns:
                temp_show[col] = temp_show[col].map(lambda x: f"{x:.3f}" if pd.notna(x) else "N/A")
        st.dataframe(temp_show, use_container_width=True, hide_index=True)
else:
    temporal_df = pd.DataFrame()
    st.info("No se detectó una columna temporal utilizable para análisis de estabilidad.")

st.markdown("## 5) Comparación segmentada")
info_box(
    "Este bloque ayuda a ver si el modelo funciona igual de bien en todos los contextos. "
    "Diferencias fuertes por canal, país o banda de riesgo pueden indicar necesidad de ajuste o supervisión adicional."
)

segment_tables = []

seg_cols_to_try = []
if schema["channel"] is not None:
    seg_cols_to_try.append(("Canal", schema["channel"]))
if schema["country"] is not None:
    seg_cols_to_try.append(("País", schema["country"]))
if risk_bucket_col is not None:
    seg_cols_to_try.append(("Score band / risk bucket", risk_bucket_col))

if seg_cols_to_try:
    tabs = st.tabs([name for name, _ in seg_cols_to_try])

    for tab, (tab_name, seg_col) in zip(tabs, seg_cols_to_try):
        with tab:
            seg_perf = build_segment_performance(
                df=df,
                segment_col=seg_col,
                label_col="__label__",
                score_col="__score__",
                threshold=threshold,
                min_rows=100 if tab_name != "País" else 50,
            )
            segment_tables.append((tab_name, seg_perf))

            if seg_perf.empty:
                st.info(f"No hay suficiente volumen para mostrar comparación por {tab_name.lower()}.")
            else:
                st.plotly_chart(plot_segment_comparison(seg_perf), use_container_width=True)
                st.caption(f"Lectura rápida: permite detectar dónde el modelo captura mejor el fraude en {tab_name.lower()}.")

                show_seg = seg_perf.copy()
                for col in ["fraud_rate_real", "score_medio", "precision", "recall", "f1", "alert_rate", "fraud_capture_rate"]:
                    show_seg[col] = show_seg[col].map(lambda x: f"{x:.1%}" if pd.notna(x) else "N/A")
                show_seg["roc_auc"] = show_seg["roc_auc"].map(lambda x: f"{x:.3f}" if pd.notna(x) else "N/A")
                st.dataframe(show_seg, use_container_width=True, hide_index=True)
else:
    st.info("No se detectaron columnas segmentables como canal, país o risk bucket.")

st.markdown("## 6) Señales de deterioro o monitorización de estabilidad")

segment_for_signals = None
if segment_tables:
    for _, seg_df in segment_tables:
        if seg_df is not None and not seg_df.empty:
            segment_for_signals = seg_df
            break

signals = detect_deterioration_signals(temporal_df, segment_for_signals)

for s in signals:
    st.markdown(f"- {s}")

st.markdown("## 7) Lectura ejecutiva y lectura técnica")

exec_msg_parts = []
if pd.notna(auc_metrics["roc_auc"]):
    exec_msg_parts.append(f"el modelo presenta un AUC ROC de {auc_metrics['roc_auc']:.3f}")
if pd.notna(conf_metrics["precision"]):
    exec_msg_parts.append(f"una precision operativa de {conf_metrics['precision']:.1%}")
if pd.notna(conf_metrics["fraud_capture_rate"]):
    exec_msg_parts.append(f"y captura {conf_metrics['fraud_capture_rate']:.1%} del fraude observado al threshold actual")
if pd.notna(conf_metrics["alert_rate"]):
    exec_msg_parts.append(f"con un alert rate de {conf_metrics['alert_rate']:.1%}")

exec_text = "En esta muestra, " + ", ".join(exec_msg_parts) + "." if exec_msg_parts else "No hay suficientes métricas para una lectura ejecutiva."

tech_text = (
    "Desde analítica, la prioridad es validar tres frentes: "
    "1) si el score sigue separando bien fraude vs no fraude, "
    "2) si el threshold actual está alineado con la capacidad del equipo, "
    "y 3) si hay desviaciones por periodo o por segmento que sugieran drift o degradación."
)

col_i, col_j = st.columns(2)
with col_i:
    st.markdown("### Para Dirección / Negocio")
    info_box(
        exec_text
        + " La pregunta clave es si el modelo está ayudando a concentrar la revisión manual en casos de mayor valor y a reducir ruido frente a una cola no priorizada."
    )
with col_j:
    st.markdown("### Para Fraude / Analytics")
    info_box(
        tech_text
        + " Esta página debe leerse junto con Alert Queue y Review Capacity, porque cualquier cambio de threshold afecta directamente al volumen revisable y a la eficiencia del equipo."
    )

with st.expander("Ver columnas detectadas en el dataset"):
    detected = pd.DataFrame(
        {
            "rol": list(schema.keys()) + ["risk_bucket_final"],
            "columna_detectada": list(schema.values()) + [risk_bucket_col],
        }
    )
    st.dataframe(detected, use_container_width=True, hide_index=True)
