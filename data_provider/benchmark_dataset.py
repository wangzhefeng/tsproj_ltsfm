"""Shared dataset utilities for time-series foundation model benchmarks."""

from __future__ import annotations

import io
import json
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_TIME_COLUMN_CANDIDATES = ("date", "time", "timestamp", "datetime")


@dataclass(slots=True)
class SeriesWindowBatch:
    dataset_name: str
    target_col: str
    time_col: str | None
    contexts: np.ndarray
    targets: np.ndarray
    start_indices: np.ndarray
    time_index: list[str] | None


def infer_time_column(frame: pd.DataFrame, explicit_time_col: str | None = None) -> str | None:
    # 参数配置提供了时间特征名称
    if explicit_time_col:
        if explicit_time_col not in frame.columns:
            raise ValueError(f"time column {explicit_time_col!r} not found in {list(frame.columns)}")
        return explicit_time_col
    # 推断时间特征名称
    lowered = {column.lower(): column for column in frame.columns}
    for candidate in DEFAULT_TIME_COLUMN_CANDIDATES:
        if candidate in lowered:
            return lowered[candidate]
    return None


def infer_target_column(frame: pd.DataFrame, explicit_target_col: str | None = None) -> str:
    # 参数配置提供了目标特征名称
    if explicit_target_col:
        if explicit_target_col not in frame.columns:
            raise ValueError(f"target column {explicit_target_col!r} not found in {list(frame.columns)}")
        return explicit_target_col
    # 推断目标特征名称
    excluded = {column for column in frame.columns if column.lower() in DEFAULT_TIME_COLUMN_CANDIDATES}
    numeric_columns = [
        column for column in frame.columns
        if column not in excluded and pd.api.types.is_numeric_dtype(frame[column])
    ]
    if not numeric_columns:
        raise ValueError("no numeric target column found in dataset")
    return numeric_columns[0]


# TODO 未使用
def summarize_frame(frame: pd.DataFrame, target_col: str | None = None, time_col: str | None = None) -> dict:
    selected_time_col = infer_time_column(frame, explicit_time_col=time_col)
    selected_target_col = infer_target_column(frame, explicit_target_col=target_col)
    series = pd.to_numeric(frame[selected_target_col], errors="coerce").dropna()
    return {
        "rows": int(frame.shape[0]),
        "columns": frame.columns.tolist(),
        "target_col": selected_target_col,
        "time_col": selected_time_col,
        "target_mean": float(series.mean()),
        "target_std": float(series.std(ddof=0)),
        "target_min": float(series.min()),
        "target_max": float(series.max()),
    }

# TODO未使用
def dumps_json(data: dict) -> str:
    return json.dumps(data, ensure_ascii=False, indent=2)


# TODO 待删除
def _read_from_zip(path: Path, zip_member: str | None = None) -> pd.DataFrame:
    with zipfile.ZipFile(path) as zf:
        members = [member for member in zf.namelist() if not member.endswith("/")]
        if not members:
            raise ValueError(f"zip archive {path} does not contain any files")
        member = zip_member or (members[0] if len(members) == 1 else None)
        if member is None:
            raise ValueError(
                f"zip archive {path} contains multiple files; please set --zip-member from {members}"
            )
        with zf.open(member) as handle:
            data = handle.read()
    buffer = io.BytesIO(data)
    if member.endswith(".jsonl"):
        return pd.read_json(buffer, lines=True)
    return pd.read_csv(buffer)


def _sort_time_column(frame: pd.DataFrame, time_col: str | None = None) -> pd.DataFrame:
    selected_time_col = infer_time_column(frame, explicit_time_col=time_col)
    if selected_time_col is None:
        return frame

    sorted_frame = frame.copy()
    try:
        sorted_frame[selected_time_col] = pd.to_datetime(sorted_frame[selected_time_col])
        sorted_frame = sorted_frame.sort_values(by=selected_time_col).reset_index(drop=True)
        sorted_frame[selected_time_col] = sorted_frame[selected_time_col].astype(str)
    except Exception:
        sorted_frame = sorted_frame.sort_values(by=selected_time_col).reset_index(drop=True)
    
    return sorted_frame


def read_time_series_frame(data_path: str | Path, zip_member: str | None = None, time_col: str | None = None) -> pd.DataFrame:
    """
    Read CSV or JSONL time-series data from plain files or ZIP archives.
    """
    path = Path(data_path)
    suffixes = path.suffixes

    if suffixes[-1:] == [".zip"]:
        frame = _read_from_zip(path, zip_member=zip_member)
    elif suffixes[-1:] == [".jsonl"]:
        frame = pd.read_json(path, lines=True)
    elif suffixes[-1:] == [".csv"]:
        frame = pd.read_csv(path)

    return _sort_time_column(frame, time_col=time_col)


def build_series_windows(
    frame: pd.DataFrame,
    context_length: int,
    prediction_length: int,
    target_col: str | None = None,
    time_col: str | None = None,
    stride: int = 1,
    sample_limit: int | None = None,
) -> SeriesWindowBatch:
    if context_length <= 0 or prediction_length <= 0:
        raise ValueError("context_length and prediction_length must be positive")
    if stride <= 0:
        raise ValueError("stride must be positive")
    # 时间特征
    selected_time_col = infer_time_column(frame, explicit_time_col=time_col)
    # 目标特征
    selected_target_col = infer_target_column(frame, explicit_target_col=target_col)
    # 数据清洗
    clean_frame = frame.copy()
    clean_frame[selected_target_col] = pd.to_numeric(clean_frame[selected_target_col], errors="coerce")
    clean_frame = clean_frame.dropna(subset=[selected_target_col]).reset_index(drop=True)
    values = clean_frame[selected_target_col].to_numpy(dtype=np.float32)
    
    # 输入数据窗口长度
    total_length = context_length + prediction_length
    if len(values) < total_length:
        raise ValueError(f"dataset length {len(values)} is shorter than required window size {total_length}")

    starts = list(range(0, len(values) - total_length + 1, stride))
    if sample_limit is not None:
        starts = starts[:sample_limit]
    
    # 输入数据窗口：[context, target]
    contexts = np.stack([values[start:start + context_length] for start in starts]).astype(np.float32)
    targets = np.stack([values[start + context_length:start + total_length] for start in starts]).astype(np.float32)
    time_index = None
    if selected_time_col is not None:
        raw_times = clean_frame[selected_time_col].astype(str).tolist()
        time_index = [raw_times[start + context_length - 1] for start in starts]

    return SeriesWindowBatch(
        dataset_name="unknown",
        target_col=selected_target_col,
        time_col=selected_time_col,
        contexts=contexts,
        targets=targets,
        start_indices=np.asarray(starts, dtype=np.int64),
        time_index=time_index,
    )
