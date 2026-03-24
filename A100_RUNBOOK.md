# A100 Runbook

本文件给出在 A100 8GPU 服务器上执行本项目正式实验的建议步骤。

## 1. 同步项目

建议同步以下目录：

- `dataset/`
- `pretrain_models/`
- `models/`
- `scripts/`
- `utils/`
- `data_provider/`
- `pyproject.toml`

如果服务器上希望单独下载更大的 checkpoint，也可以只同步代码和数据，不同步 `pretrain_models/`。

## 2. 环境准备

建议使用独立虚拟环境，并安装项目依赖：

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .
pip install accelerate
```

推荐环境变量：

```bash
export HF_HOME=/path/to/hf_cache
export HF_HUB_CACHE=/path/to/hf_cache/hub
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
```

## 3. 基础检查

```bash
python models/time_moe_usage/run_benchmark.py --help
python models/sundial_usage/run_benchmark.py --help
python - <<'PY'
import torch
print('cuda', torch.cuda.is_available())
print('count', torch.cuda.device_count())
PY
```

## 4. 推荐执行顺序

### 4.1 标准对比

先运行 ETTh1：

```bash
bash scripts/time_moe/a100/50m/etth1_a100.sh
bash scripts/time_moe/a100/200m/etth1_200m_a100.sh
bash scripts/sundial/a100/etth1_a100.sh
```

推荐顺序：

1. `TimeMoE-50M` 先作为稳定基线
2. `TimeMoE-200M` 再做放大实验
3. `Sundial-base-128m` 做对齐比较

### 4.2 扩展数据

在标准对比成功后，再扩展到：

- `traffic`
- `electricity`
- `weather`
- `ETTh2`

推荐扩展脚本：

```bash
bash scripts/time_moe/a100/50m/etth2_a100.sh
bash scripts/time_moe/a100/50m/electricity_a100.sh
bash scripts/time_moe/a100/50m/traffic_a100.sh
bash scripts/time_moe/a100/50m/weather_a100.sh
bash scripts/time_moe/a100/200m/etth2_200m_a100.sh
bash scripts/time_moe/a100/200m/electricity_200m_a100.sh
bash scripts/time_moe/a100/200m/traffic_200m_a100.sh
bash scripts/time_moe/a100/200m/weather_200m_a100.sh
bash scripts/sundial/a100/etth2_a100.sh
bash scripts/sundial/a100/electricity_a100.sh
bash scripts/sundial/a100/traffic_a100.sh
bash scripts/sundial/a100/weather_a100.sh
```

建议优先使用本地 smoke 脚本作为模板，单独复制一份服务器版脚本，再把：

- `context-length`
- `prediction-length`
- `sample-limit`
- `batch-size`

逐步放大。

## 5. 结果回传

建议至少回传以下目录：

- `results/time_moe/`
- `results/sundial/`

确保每次正式运行都保留：

- `config.json`
- `metrics.json`
- `predictions.csv`
- `summary.md`
- `plot.png`（如果生成）

## 6. 推荐记录项

正式实验建议同时记录：

- GPU 型号与数量
- 总显存与单卡峰值显存
- 运行时长
- batch size
- context length
- prediction length
- checkpoint 名称
