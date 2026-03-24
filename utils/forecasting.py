"""Evaluation helpers shared by model-specific CLI entrypoints."""

from __future__ import annotations

import json
import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path("/tmp/matplotlib-cache")))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


@dataclass(slots=True)
class EvalConfig:
    model_name: str
    checkpoint: str
    data_path: str
    dataset_name: str
    env: str
    device: str
    device_map: str
    dtype: str
    context_length: int
    prediction_length: int
    batch_size: int
    num_samples: int
    stride: int
    sample_limit: int | None
    target_col: str
    time_col: str | None
    zip_member: str | None


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    error = y_pred - y_true
    mae = np.mean(np.abs(error))
    mse = np.mean(np.square(error))
    rmse = math.sqrt(mse)
    denom = np.abs(y_true) + 1e-8
    mape = np.mean(np.abs(error) / denom)
    return {
        "mae": float(mae),
        "mse": float(mse),
        "rmse": float(rmse),
        "mape": float(mape),
    }


def save_run_artifacts(
    output_dir: str | Path,
    config: EvalConfig,
    metrics: dict[str, float],
    contexts: np.ndarray,
    targets: np.ndarray,
    predictions: np.ndarray,
    start_indices: np.ndarray,
    time_index: list[str] | None = None,
    save_plot: bool = True,
) -> dict[str, str]:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    config_path = out_dir / "config.json"
    metrics_path = out_dir / "metrics.json"
    predictions_path = out_dir / "predictions.csv"
    summary_path = out_dir / "summary.md"
    plot_path = out_dir / "plot.png"

    with config_path.open("w", encoding="utf-8") as handle:
        json.dump(asdict(config), handle, ensure_ascii=False, indent=2)

    payload = {
        "model_name": config.model_name,
        "dataset_name": config.dataset_name,
        "env": config.env,
        "device_config": f"device={config.device},device_map={config.device_map},dtype={config.dtype}",
        "model_checkpoint": config.checkpoint,
        "num_windows": int(len(contexts)),
        "context_length": config.context_length,
        "prediction_length": config.prediction_length,
        "metrics": metrics,
    }
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)

    predictions_frame = _build_predictions_frame(
        contexts=contexts,
        targets=targets,
        predictions=predictions,
        start_indices=start_indices,
        time_index=time_index,
    )
    predictions_frame.to_csv(predictions_path, index=False)

    summary = [
        f"# {config.model_name} benchmark summary",
        "",
        f"- dataset: `{config.dataset_name}`",
        f"- env: `{config.env}`",
        f"- checkpoint: `{config.checkpoint}`",
        f"- windows: `{len(contexts)}`",
        f"- context_length: `{config.context_length}`",
        f"- prediction_length: `{config.prediction_length}`",
        f"- metrics: `{metrics}`",
    ]
    summary_path.write_text("\n".join(summary) + "\n", encoding="utf-8")

    if save_plot and len(contexts) > 0:
        _save_example_plot(contexts[0], targets[0], predictions[0], plot_path)

    return {
        "config": str(config_path),
        "metrics": str(metrics_path),
        "predictions": str(predictions_path),
        "summary": str(summary_path),
        "plot": str(plot_path),
    }


def _build_predictions_frame(
    contexts: np.ndarray,
    targets: np.ndarray,
    predictions: np.ndarray,
    start_indices: np.ndarray,
    time_index: list[str] | None = None,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for idx in range(len(contexts)):
        row = {
            "window_idx": idx,
            "start_idx": int(start_indices[idx]),
            "context": json.dumps(contexts[idx].tolist(), ensure_ascii=False),
            "target": json.dumps(targets[idx].tolist(), ensure_ascii=False),
            "prediction": json.dumps(predictions[idx].tolist(), ensure_ascii=False),
        }
        if time_index is not None:
            row["context_end_time"] = time_index[idx]
        rows.append(row)
    return pd.DataFrame(rows)


def _save_example_plot(
    context: np.ndarray,
    target: np.ndarray,
    prediction: np.ndarray,
    output_path: Path,
) -> None:
    x_context = np.arange(len(context))
    x_future = np.arange(len(context), len(context) + len(target))
    plt.figure(figsize=(10, 4))
    plt.plot(x_context, context, label="context", linewidth=1.8)
    plt.plot(x_future, target, label="target", linewidth=1.8)
    plt.plot(x_future, prediction, label="prediction", linewidth=1.8)
    plt.xlabel("time step")
    plt.ylabel("value")
    plt.title("Forecast example")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()
