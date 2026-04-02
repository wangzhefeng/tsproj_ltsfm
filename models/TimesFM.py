import timesfm
import torch
from pathlib import Path
from torch import nn
from safetensors.torch import load_file as load_safetensors


DEFAULT_LOCAL_CHECKPOINT = "pretrain_models/timesfm-2.5-200m-pytorch"
DEFAULT_REMOTE_CHECKPOINT = "google/timesfm-2.5-200m-pytorch"


def _resolve_model_source(configs):
    configured = getattr(configs, "pretrain_checkpoints", None)
    candidates = [configured, DEFAULT_LOCAL_CHECKPOINT]
    for candidate in candidates:
        if not candidate:
            continue
        candidate_path = Path(candidate)
        checkpoint_path = _materialize_checkpoint(candidate_path)
        if checkpoint_path.exists():
            return {"path": str(checkpoint_path)}
    if configured and configured != "./pretrain_checkpoints/":
        candidate_path = Path(configured)
        checkpoint_path = _materialize_checkpoint(candidate_path)
        if checkpoint_path.exists():
            return {"path": str(checkpoint_path)}
    return {"huggingface_repo_id": DEFAULT_REMOTE_CHECKPOINT, "local_dir": DEFAULT_LOCAL_CHECKPOINT}


def _materialize_checkpoint(candidate_path: Path) -> Path:
    checkpoint_path = candidate_path / "torch_model.ckpt"
    if checkpoint_path.exists():
        return checkpoint_path

    safetensors_path = candidate_path / "model.safetensors"
    if safetensors_path.exists():
        state_dict = load_safetensors(str(safetensors_path))
        torch.save(state_dict, checkpoint_path)
    return checkpoint_path


def _resolve_backend(configs) -> str:
    if getattr(configs, "use_gpu", False) and getattr(configs, "gpu_type", "cuda") == "cuda" and torch.cuda.is_available():
        return "gpu"
    return "cpu"


class Model(nn.Module):
    def __init__(self, configs):
        """
        patch_len: int, patch len for patch_embedding
        stride: int, stride for patch_embedding
        """
        super().__init__()
        backend = _resolve_backend(configs)
        checkpoint_kwargs = _resolve_model_source(configs)
        self.model = timesfm.TimesFm(
            hparams=timesfm.TimesFmHparams(
                context_len=configs.seq_len,
                horizon_len=configs.pred_len,
                backend=backend,
                per_core_batch_size=32,
            ),
            checkpoint=timesfm.TimesFmCheckpoint(
                version="torch",
                path=checkpoint_kwargs.get("path"),
                huggingface_repo_id=checkpoint_kwargs.get("huggingface_repo_id"),
                local_dir=checkpoint_kwargs.get("local_dir"),
            ),
        )

        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.freq_code = timesfm.freq_map(getattr(configs, "freq", "h"))

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc.sub(means)
        stdev = torch.sqrt(
            torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc = x_enc.div(stdev)

        B, L, C = x_enc.shape
        device = x_enc.device
        flat_contexts = torch.reshape(x_enc, (B * C, L)).detach().cpu().numpy()
        inputs = [flat_contexts[idx] for idx in range(flat_contexts.shape[0])]
        output, _ = self.model.forecast(
            inputs=inputs,
            freq=[self.freq_code] * len(inputs),
            forecast_context_len=self.seq_len,
        )
        output = torch.Tensor(output).to(device)

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
