from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DASHBOARD_EXPORTS_DIR = ROOT / "artifacts" / "dashboard_exports"

DATASET_FILE_MAP: dict[str, str] = {
    "dashboard_scored_transactions": "dashboard_scored_transactions.csv",
    "executive_kpis": "executive_kpis.csv",
    "alert_queue": "alert_queue.csv",
    "review_queue_top_1pct": "review_queue_top_1pct.csv",
    "review_queue_top_3pct": "review_queue_top_3pct.csv",
    "review_queue_top_5pct": "review_queue_top_5pct.csv",
    "review_queue_top_10pct": "review_queue_top_10pct.csv",
    "queue_summary": "queue_summary.csv",
    "output_inventory": "output_inventory.csv",
}


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df

    out = df.copy()
    out.columns = [str(c).replace("\ufeff", "").strip() for c in out.columns]
    return out


def _drop_fully_unnamed_columns(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df

    keep_cols = [c for c in df.columns if not str(c).lower().startswith("unnamed:")]
    return df.loc[:, keep_cols].copy()


def _read_csv_robust(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()

    read_attempts: list[dict[str, Any]] = [
        {"sep": ",", "encoding": "utf-8"},
        {"sep": None, "engine": "python", "encoding": "utf-8"},
        {"sep": ",", "encoding": "utf-8-sig"},
        {"sep": ";", "encoding": "utf-8"},
        {"sep": ";", "encoding": "utf-8-sig"},
        {"sep": None, "engine": "python", "encoding": "latin-1"},
    ]

    best_df = pd.DataFrame()
    best_score = -1

    for kwargs in read_attempts:
        try:
            df = pd.read_csv(path, **kwargs)
            df = _normalize_columns(df)
            df = _drop_fully_unnamed_columns(df)

            if df.empty:
                continue

            n_cols = len(df.columns)
            n_rows = len(df)
            single_col_penalty = 1000 if n_cols == 1 else 0
            score = (n_cols * 1000) + n_rows - single_col_penalty

            if score > best_score:
                best_df = df
                best_score = score

            if n_cols > 1:
                return df

        except Exception:
            continue

    return best_df


def load_dataset(dataset_name: str) -> pd.DataFrame:
    filename = DATASET_FILE_MAP.get(dataset_name, f"{dataset_name}.csv")
    path = DASHBOARD_EXPORTS_DIR / filename
    return _read_csv_robust(path)


def load_selected_datasets(dataset_names: list[str]) -> dict[str, pd.DataFrame]:
    return {dataset_name: load_dataset(dataset_name) for dataset_name in dataset_names}


def list_expected_files() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for dataset_name, filename in DATASET_FILE_MAP.items():
        path = DASHBOARD_EXPORTS_DIR / filename
        exists = path.exists()
        size_bytes = path.stat().st_size if exists else None

        detected_shape = None
        detected_columns = None

        if exists and size_bytes and size_bytes > 0:
            try:
                sample_df = _read_csv_robust(path)
                detected_shape = tuple(sample_df.shape)
                detected_columns = len(sample_df.columns)
            except Exception:
                detected_shape = None
                detected_columns = None

        rows.append(
            {
                "dataset_name": dataset_name,
                "filename": filename,
                "exists": exists,
                "size_bytes": size_bytes,
                "detected_shape": detected_shape,
                "detected_columns": detected_columns,
                "path": str(path),
            }
        )

    return pd.DataFrame(rows)


def first_existing_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    if df is None or df.empty:
        return None

    lower_map = {str(c).lower(): c for c in df.columns}
    for candidate in candidates:
        match = lower_map.get(str(candidate).lower())
        if match is not None:
            return match
    return None


def infer_main_transaction_date_column(df: pd.DataFrame) -> str | None:
    return first_existing_col(
        df,
        [
            "transaction_ts",
            "transaction_datetime",
            "event_ts",
            "event_datetime",
            "scored_at",
            "transaction_date",
            "date",
        ],
    )


def parse_datetime_column(df: pd.DataFrame, column: str) -> pd.DataFrame:
    if df is None or df.empty or column not in df.columns:
        return df

    out = df.copy()
    out[column] = pd.to_datetime(out[column], errors="coerce")
    return out


def extract_kpi_value(
    df: pd.DataFrame,
    candidate_keys: list[str],
) -> float | None:
    if df is None or df.empty:
        return None

    normalized_keys = {str(k).strip().lower() for k in candidate_keys}

    # 1) Wide format: KPI names as columns
    wide_col_map = {str(c).strip().lower(): c for c in df.columns}
    for key in candidate_keys:
        col = wide_col_map.get(str(key).strip().lower())
        if col is not None:
            series = pd.to_numeric(df[col], errors="coerce").dropna()
            if not series.empty:
                return float(series.iloc[0])

    # 2) Long format: key/value table
    possible_key_cols = ["metric", "kpi", "key", "name", "metric_name", "kpi_name"]
    possible_value_cols = ["value", "metric_value", "kpi_value", "metric_result", "score"]

    key_col = first_existing_col(df, possible_key_cols)
    value_col = first_existing_col(df, possible_value_cols)

    if key_col is not None and value_col is not None:
        tmp = df.copy()
        tmp[key_col] = tmp[key_col].astype(str).str.strip().str.lower()
        match = tmp[tmp[key_col].isin(normalized_keys)]
        if not match.empty:
            series = pd.to_numeric(match[value_col], errors="coerce").dropna()
            if not series.empty:
                return float(series.iloc[0])

    # 3) Two-column fallback
    if df.shape[1] >= 2:
        col1, col2 = df.columns[:2]
        tmp = df[[col1, col2]].copy()
        tmp[col1] = tmp[col1].astype(str).str.strip().str.lower()
        match = tmp[tmp[col1].isin(normalized_keys)]
        if not match.empty:
            series = pd.to_numeric(match[col2], errors="coerce").dropna()
            if not series.empty:
                return float(series.iloc[0])

    return None
