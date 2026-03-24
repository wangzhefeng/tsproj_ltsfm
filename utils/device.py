"""Device selection helpers for local and server benchmarks."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(slots=True)
class RuntimeDevice:
    device: str
    device_map: str


def resolve_runtime_device(device: str, device_map: str) -> RuntimeDevice:
    resolved_device = _resolve_device(device)
    resolved_device_map = device_map
    if resolved_device != "cuda":
        resolved_device_map = "none"
    return RuntimeDevice(device=resolved_device, device_map=resolved_device_map)


def recommended_dtype(device: str, requested_dtype: str) -> str:
    if requested_dtype != "auto":
        if requested_dtype == "float16" and device == "cpu":
            return "float32"
        if requested_dtype == "bfloat16" and device == "mps":
            return "float32"
        return requested_dtype

    if device == "cuda":
        return "bfloat16"
    if device == "mps":
        return "float16"
    return "float32"


def _resolve_device(requested: str) -> str:
    if requested == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but not available")
    if requested == "mps" and not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
        raise ValueError("MPS requested but not available")
    return requested
