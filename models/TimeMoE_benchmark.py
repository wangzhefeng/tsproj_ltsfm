import os
from pathlib import Path

ROOT = Path.cwd()
os.environ.setdefault("HF_MODULES_CACHE", str(ROOT / ".hf_modules_cache"))

import torch
from torch import nn
from transformers import AutoModelForCausalLM


DEFAULT_LOCAL_CHECKPOINT = "pretrain_models/TimeMoE-50M"
DEFAULT_REMOTE_CHECKPOINT = "Maple728/TimeMoE-50M"


def _resolve_model_source(configs, default_local: str, default_remote: str) -> str:
    configured = getattr(configs, "pretrain_checkpoints", None)
    candidates = [configured, default_local]
    for candidate in candidates:
        if not candidate:
            continue
        candidate_path = Path(candidate)
        if candidate_path.is_dir() and (candidate_path / "config.json").exists():
            return str(candidate_path)
    if configured and configured != "./pretrain_checkpoints/":
        return configured
    return default_remote


class Model(nn.Module):
    def __init__(self, configs):
        """
        patch_len: int, patch len for patch_embedding
        stride: int, stride for patch_embedding
        """
        super().__init__()
        self.model = AutoModelForCausalLM.from_pretrained(
            _resolve_model_source(configs, DEFAULT_LOCAL_CHECKPOINT, DEFAULT_REMOTE_CHECKPOINT),
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

        B, L, C = x_enc.shape
        running = torch.reshape(x_enc, (B * C, L))
        steps = []
        for _ in range(self.pred_len):
            model_inputs = running.clone()
            with torch.no_grad():
                outputs = self.model(
                    input_ids=model_inputs,
                    use_cache=False,
                    return_dict=True,
                    max_horizon_length=1,
                )
            logits = outputs.logits
            if logits.ndim != 3:
                raise ValueError(f"unexpected TimeMoE logits shape: {tuple(logits.shape)}")
            next_step = logits[:, -1, 0].unsqueeze(-1)
            steps.append(next_step)
            if running.ndim == 3:
                running = running.squeeze(-1)
            running = torch.cat([running, next_step], dim=-1)

        dec_out = torch.cat(steps, dim=-1)
        dec_out = torch.reshape(dec_out, (B, self.pred_len, C))
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
