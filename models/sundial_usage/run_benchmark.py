# -*- coding: utf-8 -*-

"""Sundial benchmark CLI."""

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
from transformers.cache_utils import Cache, DynamicCache
from transformers import AutoModelForCausalLM

ROOT_STR = str(ROOT)
if ROOT_STR not in sys.path:
    sys.path.append(ROOT_STR)

from data_provider.benchmark_dataset import build_series_windows, read_time_series_frame
from utils.device import recommended_dtype, resolve_runtime_device
from utils.forecasting import (
    EvalConfig,
    compute_metrics,
    compute_probabilistic_metrics,
    save_probabilistic_artifacts,
    save_run_artifacts,
)
from utils.log_util import logger

import warnings
warnings.filterwarnings("ignore")


DEFAULT_CHECKPOINT = "pretrain_models/sundial-base-128m"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark Sundial on a univariate series.")
    parser.add_argument("--data", required=True, help="Path to CSV/JSONL/ZIP dataset.")
    parser.add_argument("--zip-member", default=None, help="Member path inside ZIP archives.")
    parser.add_argument("--dataset-name", default="unknown")
    parser.add_argument("--target-col", default=None)
    parser.add_argument("--time-col", default=None)
    parser.add_argument("--context-length", type=int, default=960)
    parser.add_argument("--prediction-length", type=int, default=96)
    parser.add_argument("--stride", type=int, default=16)
    parser.add_argument("--sample-limit", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-samples", type=int, default=20)
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
    model.prepare_inputs_for_generation = types.MethodType(
        _prepare_inputs_for_generation_compatible,
        model,
    )
    model.eval()
    return model


def run_forecast(
    model,
    contexts: np.ndarray,
    prediction_length: int,
    batch_size: int,
    num_samples: int,
    device: str,
    dtype: str,
) -> tuple[np.ndarray, np.ndarray]:
    point_outputs: list[np.ndarray] = []
    sample_outputs: list[np.ndarray] = []
    tensor_dtype = _resolve_tensor_dtype(dtype) or next(model.parameters()).dtype
    for start in range(0, len(contexts), batch_size):
        batch = torch.from_numpy(contexts[start:start + batch_size])
        batch = batch.to(dtype=tensor_dtype)
        batch = batch.to(device)
        with torch.no_grad():
            generated = model(
                input_ids=batch,
                use_cache=False,
                return_dict=True,
                max_output_length=prediction_length,
                revin=True,
                num_samples=num_samples,
            )
        candidates = _normalize_sundial_candidates(
            generated=generated.logits,
            prediction_length=prediction_length,
        )
        point_outputs.append(candidates.mean(dim=1).detach().float().cpu().numpy())
        sample_outputs.append(candidates.detach().float().cpu().numpy())
    return np.concatenate(point_outputs, axis=0), np.concatenate(sample_outputs, axis=0)


def _normalize_sundial_candidates(generated, prediction_length: int) -> torch.Tensor:
    if generated.ndim == 2:
        if generated.shape[-1] == prediction_length:
            return generated.unsqueeze(1)
        if generated.shape[-1] > prediction_length:
            return generated[:, :prediction_length].unsqueeze(1)
    if generated.ndim == 3:
        candidates = generated
        if candidates.shape[-1] > prediction_length:
            candidates = candidates[..., :prediction_length]
        elif candidates.shape[-1] != prediction_length:
            raise ValueError(f"unexpected Sundial candidate shape: {tuple(generated.shape)}")
        return candidates
    raise ValueError(f"unexpected Sundial output shape: {tuple(generated.shape)}")


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


def _prepare_inputs_for_generation_compatible(
    self,
    input_ids,
    past_key_values=None,
    attention_mask=None,
    inputs_embeds=None,
    revin=False,
    num_samples=1,
    **kwargs,
):
    if past_key_values is not None:
        if isinstance(past_key_values, Cache):
            cache_length = past_key_values.get_seq_length()
            if isinstance(past_key_values, DynamicCache):
                past_length = getattr(past_key_values, "seen_tokens", cache_length)
            else:
                past_length = cache_length
            max_cache_length = (
                past_key_values.get_max_length()
                if hasattr(past_key_values, "get_max_length")
                else None
            )
        else:
            cache_length = past_length = past_key_values[0][0].shape[2]
            max_cache_length = None

        input_token_len = self.config.input_token_len
        if attention_mask is not None and attention_mask.shape[1] > (input_ids.shape[1] // input_token_len):
            input_ids = input_ids[:, -(attention_mask.shape[1] - past_length) * input_token_len:]
        elif past_length < (input_ids.shape[1] // input_token_len):
            input_ids = input_ids[:, past_length * input_token_len:]

        if (
            max_cache_length is not None
            and attention_mask is not None
            and cache_length + (input_ids.shape[1] // input_token_len) > max_cache_length
        ):
            attention_mask = attention_mask[:, -max_cache_length:]

    position_ids = kwargs.get("position_ids", None)
    if attention_mask is not None and position_ids is None:
        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 1)
        if past_key_values:
            position_ids = position_ids[:, -(input_ids.shape[1] // self.config.input_token_len):]

    if inputs_embeds is not None and past_key_values is None:
        model_inputs = {"inputs_embeds": inputs_embeds}
    else:
        model_inputs = {"input_ids": input_ids}

    model_inputs.update(
        {
            "position_ids": position_ids,
            "past_key_values": past_key_values,
            "use_cache": kwargs.get("use_cache"),
            "attention_mask": attention_mask,
            "revin": revin,
            "num_samples": num_samples,
        }
    )
    return model_inputs


def main() -> None:
    args = build_parser().parse_args()

    frame = read_time_series_frame(args.data, zip_member=args.zip_member, time_col=args.time_col)
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
    runtime = resolve_runtime_device(args.device, args.device_map)
    resolved_dtype = recommended_dtype(runtime.device, args.dtype)

    model = load_model(
        args.checkpoint,
        device=runtime.device,
        device_map=runtime.device_map,
        dtype=resolved_dtype,
    )
    predictions, sample_predictions = run_forecast(
        model=model,
        contexts=windows.contexts,
        prediction_length=args.prediction_length,
        batch_size=args.batch_size,
        num_samples=args.num_samples,
        device=runtime.device,
        dtype=resolved_dtype,
    )

    metrics = compute_metrics(windows.targets, predictions)
    probabilistic_metrics = compute_probabilistic_metrics(windows.targets, sample_predictions)
    config = EvalConfig(
        model_name="Sundial",
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
        num_samples=args.num_samples,
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
        extra_summary_lines=[
            f"- probabilistic_metrics: `{probabilistic_metrics}`",
            "- probabilistic_outputs: `sample_predictions.npz`, `quantiles.csv`, `probabilistic_metrics.json`, `probabilistic_plot.png`",
        ],
    )
    save_probabilistic_artifacts(
        output_dir=args.output_dir,
        sample_predictions=sample_predictions,
        targets=windows.targets,
        start_indices=windows.start_indices,
        probabilistic_metrics=probabilistic_metrics,
        time_index=windows.time_index,
        save_plot=args.save_plot,
    )

    logger.info(metrics)
    logger.info(probabilistic_metrics)

if __name__ == "__main__":
    main()
