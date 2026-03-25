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


def compute_probabilistic_metrics(
    y_true: np.ndarray,
    sample_predictions: np.ndarray,
    quantiles: tuple[float, ...] = (0.1, 0.5, 0.9),
    interval_quantiles: tuple[float, float] = (0.05, 0.95),
) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=np.float64)
    sample_predictions = np.asarray(sample_predictions, dtype=np.float64)
    if sample_predictions.ndim != 3:
        raise ValueError(
            f"expected probabilistic predictions with shape [windows, samples, horizon], got {sample_predictions.shape}"
        )

    metrics: dict[str, float] = {}
    for quantile in quantiles:
        quantile_prediction = np.quantile(sample_predictions, quantile, axis=1)
        error = y_true - quantile_prediction
        pinball = np.maximum(quantile * error, (quantile - 1.0) * error)
        metrics[f"pinball_loss_q{int(quantile * 100):02d}"] = float(np.mean(pinball))

    lower_q, upper_q = interval_quantiles
    lower = np.quantile(sample_predictions, lower_q, axis=1)
    upper = np.quantile(sample_predictions, upper_q, axis=1)
    coverage = np.mean((y_true >= lower) & (y_true <= upper))
    width = np.mean(upper - lower)
    metrics[f"interval_coverage_q{int(lower_q * 100):02d}_q{int(upper_q * 100):02d}"] = float(coverage)
    metrics[f"interval_width_q{int(lower_q * 100):02d}_q{int(upper_q * 100):02d}"] = float(width)
    return metrics


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
    extra_summary_lines: list[str] | None = None,
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
    if extra_summary_lines:
        summary.extend(["", *extra_summary_lines])
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


def save_probabilistic_artifacts(
    output_dir: str | Path,
    sample_predictions: np.ndarray,
    targets: np.ndarray,
    start_indices: np.ndarray,
    probabilistic_metrics: dict[str, float],
    time_index: list[str] | None = None,
    quantiles: tuple[float, ...] = (0.05, 0.1, 0.5, 0.9, 0.95),
    save_plot: bool = True,
) -> dict[str, str]:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    sample_predictions_path = out_dir / "sample_predictions.npz"
    probabilistic_metrics_path = out_dir / "probabilistic_metrics.json"
    quantiles_path = out_dir / "quantiles.csv"
    probabilistic_plot_path = out_dir / "probabilistic_plot.png"

    np.savez_compressed(sample_predictions_path, sample_predictions=sample_predictions)
    with probabilistic_metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(probabilistic_metrics, handle, ensure_ascii=False, indent=2)

    quantiles_frame = _build_quantiles_frame(
        sample_predictions=sample_predictions,
        targets=targets,
        start_indices=start_indices,
        time_index=time_index,
        quantiles=quantiles,
    )
    quantiles_frame.to_csv(quantiles_path, index=False)

    if save_plot and len(sample_predictions) > 0:
        median = np.quantile(sample_predictions[0], 0.5, axis=0)
        lower = np.quantile(sample_predictions[0], 0.05, axis=0)
        upper = np.quantile(sample_predictions[0], 0.95, axis=0)
        _save_probabilistic_plot(
            target=targets[0],
            median_prediction=median,
            interval_lower=lower,
            interval_upper=upper,
            output_path=probabilistic_plot_path,
        )

    return {
        "sample_predictions": str(sample_predictions_path),
        "probabilistic_metrics": str(probabilistic_metrics_path),
        "quantiles": str(quantiles_path),
        "probabilistic_plot": str(probabilistic_plot_path),
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


def _build_quantiles_frame(
    sample_predictions: np.ndarray,
    targets: np.ndarray,
    start_indices: np.ndarray,
    time_index: list[str] | None,
    quantiles: tuple[float, ...],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for idx in range(len(sample_predictions)):
        row = {
            "window_idx": idx,
            "start_idx": int(start_indices[idx]),
            "target": json.dumps(targets[idx].tolist(), ensure_ascii=False),
        }
        if time_index is not None:
            row["context_end_time"] = time_index[idx]
        for quantile in quantiles:
            quantile_prediction = np.quantile(sample_predictions[idx], quantile, axis=0)
            row[f"q{int(quantile * 100):02d}"] = json.dumps(quantile_prediction.tolist(), ensure_ascii=False)
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


def _save_probabilistic_plot(
    target: np.ndarray,
    median_prediction: np.ndarray,
    interval_lower: np.ndarray,
    interval_upper: np.ndarray,
    output_path: Path,
) -> None:
    x_future = np.arange(len(target))
    plt.figure(figsize=(10, 4))
    plt.fill_between(x_future, interval_lower, interval_upper, alpha=0.3, color="royalblue", label="q05-q95")
    plt.plot(x_future, target, label="target", linewidth=1.8)
    plt.plot(x_future, median_prediction, label="q50", linewidth=1.8)
    plt.xlabel("forecast step")
    plt.ylabel("value")
    plt.title("Probabilistic forecast example")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()
