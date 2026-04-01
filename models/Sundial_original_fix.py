import os
from pathlib import Path
import types

ROOT = Path.cwd()
os.environ.setdefault("HF_MODULES_CACHE", str(ROOT / ".hf_modules_cache"))

import torch
from torch import nn
from transformers.cache_utils import Cache, DynamicCache
from transformers import AutoModelForCausalLM


DEFAULT_LOCAL_CHECKPOINT = "pretrain_models/sundial-base-128m"
DEFAULT_REMOTE_CHECKPOINT = "thuml/sundial-base-128m"


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


class Model(nn.Module):
    def __init__(self, configs):
        """
        patch_len: int, patch len for patch_embedding
        stride: int, stride for patch_embedding
        """
        super().__init__()
        self.model = AutoModelForCausalLM.from_pretrained(_resolve_model_source(configs), trust_remote_code=True)
        if not hasattr(self.model, "_extract_past_from_model_output"):
            self.model._extract_past_from_model_output = types.MethodType(_extract_past_from_model_output, self.model)
        self.model.prepare_inputs_for_generation = types.MethodType(
            _prepare_inputs_for_generation_compatible,
            self.model,
        )
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.num_samples = getattr(configs, "num_samples", 20)

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        outputs = []
        for i in range(x_enc.shape[-1]):
            try:
                output = self.model.generate(x_enc[...,i], max_new_tokens=self.pred_len, num_samples=self.num_samples)
                output = output.mean(dim=1)
            except AttributeError:
                with torch.no_grad():
                    generated = self.model(
                        input_ids=x_enc[..., i],
                        use_cache=False,
                        return_dict=True,
                        max_output_length=self.pred_len,
                        revin=True,
                        num_samples=self.num_samples,
                    )
                output = _normalize_sundial_candidates(generated.logits, self.pred_len).mean(dim=1)
            outputs.append(output)
        dec_out = torch.stack(outputs, dim=-1)
        return dec_out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'zero_shot_forecast':
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out
        return None


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
