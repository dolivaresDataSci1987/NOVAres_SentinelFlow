from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DASHBOARD_EXPORTS_DIR = PROJECT_ROOT / "artifacts" / "dashboard_exports"


CSV_FILES = {
    "dashboard_scored_transactions": "dashboard_scored_transactions.csv",
    "alert_queue": "alert_queue.csv",
    "review_queue_top_1pct": "review_queue_top_1pct.csv",
    "review_queue_top_3pct": "review_queue_top_3pct.csv",
    "review_queue_top_5pct": "review_queue_top_5pct.csv",
    "review_queue_top_10pct": "review_queue_top_10pct.csv",
    "executive_kpis": "executive_kpis.csv",
    "daily_monitoring": "daily_monitoring.csv",
    "daily_channel_monitoring": "daily_channel_monitoring.csv",
    "channel_summary": "channel_summary.csv",
    "merchant_category_summary": "merchant_category_summary.csv",
    "customer_segment_summary": "customer_segment_summary.csv",
    "payment_type_summary": "payment_type_summary.csv",
    "merchant_risk_level_summary": "merchant_risk_level_summary.csv",
    "transaction_country_summary": "transaction_country_summary.csv",
    "suspicious_customers": "suspicious_customers.csv",
    "suspicious_merchants": "suspicious_merchants.csv",
    "suspicious_devices": "suspicious_devices.csv",
    "suspicious_payment_methods": "suspicious_payment_methods.csv",
    "alert_composition": "alert_composition.csv",
    "alert_mix_by_channel": "alert_mix_by_channel.csv",
    "alert_mix_by_merchant_category": "alert_mix_by_merchant_category.csv",
    "queue_summary": "queue_summary.csv",
    "top_scored_transactions": "top_scored_transactions.csv",
    "top_true_positive_cases": "top_true_positive_cases.csv",
    "top_false_positive_cases": "top_false_positive_cases.csv",
    "top_false_negative_cases": "top_false_negative_cases.csv",
    "output_inventory": "output_inventory.csv",
}


def _try_read_csv(path: Path, **kwargs) -> pd.DataFrame:
    try:
        return pd.read_csv(path, **kwargs)
    except Exception:
        return pd.DataFrame()


def _safe_read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()

    attempts = [
        {"low_memory": False},
        {"sep": ";", "low_memory": False},
        {"encoding": "latin-1", "low_memory": False},
        {"sep": ";", "encoding": "latin-1", "low_memory": False},
    ]

    for kwargs in attempts:
        df = _try_read_csv(path, **kwargs)
        if not df.empty or path.stat().st_size == 0:
            return df

    return pd.DataFrame()


def get_dashboard_exports_dir() -> Path:
    return DASHBOARD_EXPORTS_DIR


def list_expected_files() -> pd.DataFrame:
    rows = []
    for dataset_name, filename in CSV_FILES.items():
        path = DASHBOARD_EXPORTS_DIR / filename
        rows.append(
            {
                "dataset_name": dataset_name,
                "filename": filename,
                "exists": path.exists(),
                "size_mb": round(path.stat().st_size / (1024 * 1024), 3) if path.exists() else None,
                "path": str(path),
            }
        )
    return pd.DataFrame(rows)


def load_dataset(dataset_name: str) -> pd.DataFrame:
    if dataset_name not in CSV_FILES:
        raise ValueError(f"Unknown dataset_name: {dataset_name}")

    path = DASHBOARD_EXPORTS_DIR / CSV_FILES[dataset_name]
    return _safe_read_csv(path)


def load_selected_datasets(dataset_names: List[str]) -> Dict[str, pd.DataFrame]:
    return {name: load_dataset(name) for name in dataset_names}


def load_core_dashboard_data() -> Dict[str, pd.DataFrame]:
    core_names = [
        "dashboard_scored_transactions",
        "alert_queue",
        "review_queue_top_1pct",
        "review_queue_top_3pct",
        "review_queue_top_5pct",
        "review_queue_top_10pct",
        "executive_kpis",
        "daily_monitoring",
        "channel_summary",
        "merchant_category_summary",
        "customer_segment_summary",
        "payment_type_summary",
        "merchant_risk_level_summary",
        "transaction_country_summary",
        "suspicious_customers",
        "suspicious_merchants",
        "suspicious_devices",
        "suspicious_payment_methods",
        "alert_composition",
        "queue_summary",
        "top_scored_transactions",
        "top_true_positive_cases",
        "top_false_positive_cases",
        "top_false_negative_cases",
        "output_inventory",
    ]
    return load_selected_datasets(core_names)


def load_all_dashboard_data() -> Dict[str, pd.DataFrame]:
    return {name: load_dataset(name) for name in CSV_FILES.keys()}


def extract_kpi_value(
    executive_kpis_df: pd.DataFrame,
    candidate_keys: List[str],
    value_column_candidates: Optional[List[str]] = None,
) -> Optional[float]:
    if executive_kpis_df is None or executive_kpis_df.empty:
        return None

    df = executive_kpis_df.copy()
    df.columns = [str(c).strip() for c in df.columns]

    possible_key_cols = [
        c for c in df.columns
        if c.lower() in {"kpi", "metric", "metric_name", "name", "key", "metric_key"}
    ]
    possible_value_cols = value_column_candidates or [
        c for c in df.columns
        if c.lower() in {"value", "metric_value", "kpi_value", "metric_result"}
    ]

    if possible_key_cols and possible_value_cols:
        key_col = possible_key_cols[0]
        value_col = possible_value_cols[0]

        df[key_col] = df[key_col].astype(str).str.strip().str.lower()
        for key in candidate_keys:
            mask = df[key_col] == key.strip().lower()
            if mask.any():
                value = df.loc[mask, value_col].iloc[0]
                try:
                    return float(value)
                except Exception:
                    pass

    flat_candidates = []
    for col in df.columns:
        for _, row in df.iterrows():
            cell = str(row[col]).strip().lower()
            if cell in [k.strip().lower() for k in candidate_keys]:
                for maybe_val_col in df.columns:
                    if maybe_val_col == col:
                        continue
                    try:
                        value = float(row[maybe_val_col])
                        flat_candidates.append(value)
                    except Exception:
                        continue

    if flat_candidates:
        return flat_candidates[0]

    return None


def infer_main_transaction_date_column(df: pd.DataFrame) -> Optional[str]:
    candidates = [
        "transaction_date",
        "transaction_ts",
        "event_date",
        "date",
        "timestamp",
    ]
    lower_map = {str(c).lower(): c for c in df.columns}
    for col in candidates:
        if col.lower() in lower_map:
            return lower_map[col.lower()]
    return None


def parse_datetime_column(df: pd.DataFrame, col: str) -> pd.DataFrame:
    out = df.copy()
    if col in out.columns:
        out[col] = pd.to_datetime(out[col], errors="coerce")
    return out
