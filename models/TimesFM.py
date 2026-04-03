import timesfm
import torch
from huggingface_hub import hf_hub_download
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
        if candidate_path.is_dir():
            return {"model_dir": str(candidate_path)}
        checkpoint_path = _materialize_checkpoint(candidate_path)
        if checkpoint_path.exists():
            return {"path": str(checkpoint_path), "model_dir": str(candidate_path)}
    if configured and configured != "./pretrain_checkpoints/":
        candidate_path = Path(configured)
        if candidate_path.is_dir():
            return {"model_dir": str(candidate_path)}
        checkpoint_path = _materialize_checkpoint(candidate_path)
        if checkpoint_path.exists():
            return {"path": str(checkpoint_path), "model_dir": str(candidate_path)}
    return {
        "huggingface_repo_id": DEFAULT_REMOTE_CHECKPOINT,
        "local_dir": DEFAULT_LOCAL_CHECKPOINT,
    }


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


def _resolve_timesfm_2p5_checkpoint(model_source: dict) -> str:
    model_dir = model_source.get("model_dir")
    if model_dir:
        checkpoint_path = Path(model_dir) / "model.safetensors"
        if checkpoint_path.exists():
            return str(checkpoint_path)

    return hf_hub_download(
        repo_id=model_source.get("huggingface_repo_id", DEFAULT_REMOTE_CHECKPOINT),
        filename="model.safetensors",
        local_dir=model_source.get("local_dir"),
    )


class Model(nn.Module):
    def __init__(self, configs):
        """
        patch_len: int, patch len for patch_embedding
        stride: int, stride for patch_embedding
        """
        super().__init__()
        backend = _resolve_backend(configs)
        model_source = _resolve_model_source(configs)
        self._use_timesfm_2p5 = hasattr(timesfm, "TimesFM_2p5_200M_torch")

        if self._use_timesfm_2p5:
            checkpoint_path = _resolve_timesfm_2p5_checkpoint(model_source)
            self.model = timesfm.TimesFM_2p5_200M_torch(torch_compile=False)
            self.model.model.load_checkpoint(checkpoint_path, torch_compile=False)
            self.model.compile(
                timesfm.ForecastConfig(
                    max_context=configs.seq_len,
                    max_horizon=configs.pred_len,
                    normalize_inputs=True,
                    per_core_batch_size=32,
                    use_continuous_quantile_head=True,
                    force_flip_invariance=True,
                    infer_is_positive=True,
                    fix_quantile_crossing=True,
                )
            )
        else:
            self.model = timesfm.TimesFm(
                hparams=timesfm.TimesFmHparams(
                    context_len=configs.seq_len,
                    horizon_len=configs.pred_len,
                    backend=backend,
                    per_core_batch_size=32,
                ),
                checkpoint=timesfm.TimesFmCheckpoint(
                    version="torch",
                    path=model_source.get("path"),
                    huggingface_repo_id=model_source.get("huggingface_repo_id"),
                    local_dir=model_source.get("local_dir"),
                ),
            )

        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.freq_code = None
        if not self._use_timesfm_2p5:
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
        if self._use_timesfm_2p5:
            output, _ = self.model.forecast(
                horizon=self.pred_len,
                inputs=inputs,
            )
        else:
            output, _ = self.model.forecast(
                inputs=inputs,
                freq=[self.freq_code] * len(inputs),
                forecast_context_len=self.seq_len,
            )
        output = torch.tensor(output, device=device)

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
