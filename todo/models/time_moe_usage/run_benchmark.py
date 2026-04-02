# -*- coding: utf-8 -*-

"""Time-MoE benchmark CLI."""

from __future__ import annotations

import argparse
import os
import sys
import types
from pathlib import Path

import numpy as np
ROOT = Path.cwd()
os.environ.setdefault("HF_MODULES_CACHE", str(ROOT / ".hf_modules_cache"))
import torch
from transformers import AutoModelForCausalLM

ROOT_STR = str(ROOT)
if ROOT_STR not in sys.path:
    sys.path.append(ROOT_STR)

from data_provider.benchmark_dataset import build_series_windows, read_time_series_frame
from utils.device import recommended_dtype, resolve_runtime_device
from utils.forecasting import (
    EvalConfig,
    compute_metrics,
    save_run_artifacts,
)
from utils.log_util import logger

import warnings
warnings.filterwarnings("ignore")


DEFAULT_CHECKPOINT = "pretrain_models/TimeMoE-50M"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark Time-MoE on a univariate series.")
    parser.add_argument("--data", required=True, help="Path to CSV/JSONL/ZIP dataset.")
    parser.add_argument("--zip-member", default=None, help="Member path inside ZIP archives.")
    parser.add_argument("--dataset-name", default="unknown")
    parser.add_argument("--target-col", default=None)
    parser.add_argument("--time-col", default=None)
    parser.add_argument("--context-length", type=int, default=512)
    parser.add_argument("--prediction-length", type=int, default=96)
    parser.add_argument("--stride", type=int, default=16)
    parser.add_argument("--sample-limit", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--device-map", default="auto", choices=["none", "auto"])
    parser.add_argument("--dtype", default="auto", choices=["auto", "float32", "float16", "bfloat16"])
    parser.add_argument("--env", default="local")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--save-plot", action="store_true")
    return parser


def load_model(checkpoint: str, device: str, device_map: str, dtype: str):
    kwargs = {
        "trust_remote_code": True,
    }
    torch_dtype = _resolve_dtype(dtype)
    if torch_dtype is not None:
        kwargs["torch_dtype"] = torch_dtype
    if device_map == "auto":
        kwargs["device_map"] = "auto"
    
    model = AutoModelForCausalLM.from_pretrained(checkpoint, **kwargs)
    if device_map == "none":
        model = model.to(device)
    if device == "cpu" and dtype == "float32":
        model = model.float()
    if not hasattr(model, "_extract_past_from_model_output"):
        model._extract_past_from_model_output = types.MethodType(_extract_past_from_model_output, model)
    model.eval()
    
    return model


def run_forecast(
    model,
    contexts: np.ndarray,
    prediction_length: int,
    batch_size: int,
    device: str,
    dtype: str,
) -> np.ndarray:
    outputs: list[np.ndarray] = []
    model_dtype = next(model.parameters()).dtype
    tensor_dtype = _resolve_tensor_dtype(dtype) or model_dtype
    for start in range(0, len(contexts), batch_size):
        batch = torch.from_numpy(contexts[start:start + batch_size])
        batch = batch.to(dtype=tensor_dtype)
        batch = batch.to(device)

        mean = batch.mean(dim=-1, keepdim=True)
        std = batch.std(dim=-1, keepdim=True)
        std = torch.where(std < 1e-6, torch.ones_like(std), std)
        normalized = (batch - mean) / std

        normalized_predictions = _autoregressive_forecast(
            model=model,
            normalized_context=normalized,
            prediction_length=prediction_length,
        )
        predictions = normalized_predictions * std + mean
        outputs.append(predictions.detach().float().cpu().numpy())
    return np.concatenate(outputs, axis=0)


def _autoregressive_forecast(model, normalized_context: torch.Tensor, prediction_length: int) -> torch.Tensor:
    running = normalized_context
    steps: list[torch.Tensor] = []
    for _ in range(prediction_length):
        model_inputs = running.clone()
        with torch.no_grad():
            outputs = model(
                input_ids=model_inputs,
                use_cache=False,
                return_dict=True,
                max_horizon_length=1,
            )
        logits = outputs.logits
        if logits.ndim != 3:
            raise ValueError(f"unexpected Time-MoE logits shape: {tuple(logits.shape)}")
        next_step = logits[:, -1, 0].unsqueeze(-1)
        steps.append(next_step)
        if running.ndim == 3:
            running = running.squeeze(-1)
        running = torch.cat([running, next_step], dim=-1)
    return torch.cat(steps, dim=-1)


def _resolve_dtype(dtype: str):
    mapping = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    return mapping.get(dtype)


def _resolve_tensor_dtype(dtype: str):
    if dtype == "float16" and not torch.cuda.is_available():
        return torch.float32
    return _resolve_dtype(dtype)


def _extract_past_from_model_output(self, outputs, standardize_cache_format: bool = False):
    del standardize_cache_format
    return getattr(outputs, "past_key_values", None)


def main() -> None:
    args = build_parser().parse_args()

    # 输入数据
    frame = read_time_series_frame(args.data, zip_member=args.zip_member, time_col=args.time_col)
    logger.info(f"input data: {frame.head()}")
    # 预测数据窗口
    windows = build_series_windows(
        frame=frame,
        context_length=args.context_length,
        prediction_length=args.prediction_length,
        target_col=args.target_col,
        time_col=args.time_col,
        stride=args.stride,
        sample_limit=args.sample_limit,
    )
    windows.dataset_name = args.dataset_name
    logger.info(f"input windows: {windows}")
    logger.info(f"input windows context: {len(windows.contexts)}")
    logger.info(f"input windows prediction_length: {len(windows.prediction_length)}")
    # 模型运行配置
    runtime = resolve_runtime_device(args.device, args.device_map)
    resolved_dtype = recommended_dtype(runtime.device, args.dtype)
    # 模型加载
    model = load_model(
        args.checkpoint,
        device=runtime.device,
        device_map=runtime.device_map,
        dtype=resolved_dtype,
    )
    # 模型预测
    predictions = run_forecast(
        model=model,
        contexts=windows.contexts,
        prediction_length=args.prediction_length,
        batch_size=args.batch_size,
        device=runtime.device,
        dtype=resolved_dtype,
    )
    # 模型预测评估
    metrics = compute_metrics(windows.targets, predictions)
    config = EvalConfig(
        model_name="Time-MoE",
        checkpoint=args.checkpoint,
        data_path=args.data,
        dataset_name=args.dataset_name,
        env=args.env,
        device=runtime.device,
        device_map=runtime.device_map,
        dtype=resolved_dtype,
        context_length=args.context_length,
        prediction_length=args.prediction_length,
        batch_size=args.batch_size,
        num_samples=1,
        stride=args.stride,
        sample_limit=args.sample_limit,
        target_col=windows.target_col,
        time_col=windows.time_col,
        zip_member=args.zip_member,
    )
    save_run_artifacts(
        output_dir=args.output_dir,
        config=config,
        metrics=metrics,
        contexts=windows.contexts,
        targets=windows.targets,
        predictions=predictions,
        start_indices=windows.start_indices,
        time_index=windows.time_index,
        save_plot=args.save_plot,
    )

    logger.info(metrics)

if __name__ == "__main__":
    main()
