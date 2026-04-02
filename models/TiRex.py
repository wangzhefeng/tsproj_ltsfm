from pathlib import Path

import torch
from torch import nn


DEFAULT_LOCAL_CHECKPOINT = "pretrain_models/TiRex"
DEFAULT_REMOTE_CHECKPOINT = "NX-AI/TiRex"


def _is_valid_checkpoint_dir(candidate_path: Path) -> bool:
    expected_files = (
        "config.json",
        "model.safetensors",
        "pytorch_model.bin",
        "torch_model.ckpt",
        "model.ckpt",
        "tirex.onnx",
    )
    return any((candidate_path / file_name).exists() for file_name in expected_files)


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
        try:
            from tirex import load_model
        except ImportError as exc:
            raise ImportError(
                "TiRex requires an inference package exposing `tirex.load_model`, but the current environment "
                "does not provide it. Please install `tirex-ts` and place the "
                "checkpoint under `pretrain_models/TiRex` if you want local loading."
            ) from exc

        if load_model is None:
            raise ImportError(
                "Current `tirex` package does not expose `load_model`. The installed package appears incompatible "
                "with the TiRex forecasting wrapper in this project."
            )

        self.model = load_model(_resolve_model_source(configs))
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc.sub(means)
        stdev = torch.sqrt(
            torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc = x_enc.div(stdev)

        B, L, C = x_enc.shape
        x_enc = torch.reshape(x_enc, (B*C, L))
        quantiles, output = self.model.forecast(x_enc, prediction_length=self.pred_len)
        dec_out = torch.reshape(output, (B, output.shape[-1], C)).to(x_enc.device)
        dec_out = dec_out * \
                  (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        dec_out = dec_out + \
                  (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        return dec_out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'zero_shot_forecast':
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out
        return None
