import os
from pathlib import Path

ROOT = Path.cwd()
os.environ.setdefault("HF_MODULES_CACHE", str(ROOT / ".hf_modules_cache"))

import torch
from torch import nn
from transformers import AutoModelForCausalLM


DEFAULT_LOCAL_CHECKPOINT = "pretrain_models/TimeMoE-200M"
DEFAULT_REMOTE_CHECKPOINT = "Maple728/TimeMoE-200M"


def _resolve_model_source(configs) -> str:
    configured = getattr(configs, "pretrain_checkpoints", None)
    candidates = [configured, DEFAULT_LOCAL_CHECKPOINT]
    for candidate in candidates:
        if not candidate:
            continue
        candidate_path = Path(candidate)
        if candidate_path.is_dir() and (candidate_path / "config.json").exists():
            return str(candidate_path)
    if configured and configured != "./pretrain_checkpoints/":
        return configured
    return DEFAULT_REMOTE_CHECKPOINT


class Model(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.model = AutoModelForCausalLM.from_pretrained(
            _resolve_model_source(configs),
            trust_remote_code=True,
        )
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc.sub(means)
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc = x_enc.div(stdev)

        batch_size, seq_len, channels = x_enc.shape
        x_enc = torch.reshape(x_enc, (batch_size * channels, seq_len))
        try:
            output = self.model.generate(x_enc, max_new_tokens=self.pred_len)
        except AttributeError:
            output = _autoregressive_generate(self.model, x_enc, self.pred_len)
        dec_out = torch.reshape(output, (batch_size, output.shape[-1], channels))
        dec_out = dec_out[:, -self.pred_len:, :]
        dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        return dec_out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'zero_shot_forecast':
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out
        return None


def _autoregressive_generate(model, inputs: torch.Tensor, pred_len: int) -> torch.Tensor:
    running = inputs
    steps = []
    for _ in range(pred_len):
        model_inputs = running.clone()
        with torch.no_grad():
            outputs = model(
                input_ids=model_inputs,
                use_cache=False,
                return_dict=True,
                max_horizon_length=1,
            )
        logits = outputs.logits
        next_step = logits[:, -1, 0].unsqueeze(-1)
        steps.append(next_step)
        if running.ndim == 3:
            running = running.squeeze(-1)
        running = torch.cat([running, next_step], dim=-1)
    return torch.cat([inputs, torch.cat(steps, dim=-1)], dim=-1)
