from pathlib import Path

import torch
from torch import nn
from chronos import BaseChronosPipeline


DEFAULT_LOCAL_CHECKPOINT = "pretrain_models/chronos-bolt-base"
DEFAULT_REMOTE_CHECKPOINT = "amazon/chronos-bolt-base"


def _is_valid_checkpoint_dir(candidate_path: Path) -> bool:
    expected_files = ("config.json", "model.safetensors", "pytorch_model.bin")
    return (candidate_path / "config.json").exists() and any(
        (candidate_path / file_name).exists() for file_name in expected_files[1:]
    )


def _resolve_model_source(configs) -> str:
    configured = getattr(configs, "pretrain_checkpoints", None)
    candidates = [configured, DEFAULT_LOCAL_CHECKPOINT]
    for candidate in candidates:
        if not candidate:
            continue
        candidate_path = Path(candidate)
        if candidate_path.is_dir() and _is_valid_checkpoint_dir(candidate_path):
            return str(candidate_path)
    if configured and configured != "./pretrain_checkpoints/":
        return configured
    return DEFAULT_REMOTE_CHECKPOINT


class Model(nn.Module):
    def __init__(self, configs):
        """
        patch_len: int, patch len for patch_embedding
        stride: int, stride for patch_embedding
        """
        super().__init__()
        device_map = "cpu"
        if getattr(configs, "use_gpu", False):
            if getattr(configs, "gpu_type", "cuda") == "cuda" and torch.cuda.is_available():
                device_map = "cuda"
            elif getattr(configs, "gpu_type", "") == "mps" and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device_map = "mps"

        self.model = BaseChronosPipeline.from_pretrained(
            _resolve_model_source(configs),
            device_map=device_map,
            torch_dtype=torch.bfloat16 if device_map == "cuda" else torch.float32,
        )
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        outputs = []
        for i in range(x_enc.shape[-1]):
            output = self.model.predict(x_enc[..., i], prediction_length=self.pred_len)
            output = output.mean(dim=1)
            outputs.append(output)
        dec_out = torch.stack(outputs, dim=-1)

        return dec_out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'zero_shot_forecast':
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out
        return None
