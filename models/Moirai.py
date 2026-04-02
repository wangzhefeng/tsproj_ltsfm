import numpy as np
import torch
from torch import nn
from pathlib import Path


DEFAULT_LOCAL_CHECKPOINT = "pretrain_models/moirai-2.0-R-small"
DEFAULT_REMOTE_CHECKPOINT = "Salesforce/moirai-2.0-R-small"


def _is_valid_checkpoint_dir(candidate_path: Path) -> bool:
    expected_files = ("config.json", "model.safetensors", "pytorch_model.bin")
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


def _resolve_device(configs) -> str:
    if getattr(configs, "use_gpu", False):
        if getattr(configs, "gpu_type", "cuda") == "cuda" and torch.cuda.is_available():
            return "cuda"
        if getattr(configs, "gpu_type", "") == "mps" and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    return "cpu"

class Model(nn.Module):
    def __init__(self, configs):
        """
        patch_len: int, patch len for patch_embedding
        stride: int, stride for patch_embedding
        """
        super().__init__()
        try:
            from uni2ts.model.moirai2 import Moirai2Forecast, Moirai2Module
        except ImportError as exc:
            raise ImportError(
                "Moirai requires the `uni2ts` package, but it is not installed in the current environment. "
                "Please install it from `git+https://github.com/SalesforceAIResearch/uni2ts.git` and place the checkpoint under "
                "`pretrain_models/moirai-2.0-R-small`."
            ) from exc

        device = _resolve_device(configs)
        self.model = Moirai2Forecast(
            module=Moirai2Module.from_pretrained(
                _resolve_model_source(configs),
            ),
            prediction_length=configs.pred_len,
            context_length=configs.seq_len,
            target_dim=1,
            feat_dynamic_real_dim=0,
            past_feat_dynamic_real_dim=0,
        ).to(device)

        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        outputs = []
        for i in range(x_enc.shape[-1]):
            output = self.model.predict(x_enc[..., i].detach().cpu().numpy())
            output = np.mean(output, axis=1)
            outputs.append(torch.Tensor(output).to(x_enc.device))
        dec_out = torch.stack(outputs, dim=-1)

        return dec_out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'zero_shot_forecast':
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out
        return None
