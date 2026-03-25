# Sundial Scripts

`scripts/sundial/` 按运行环境分类，避免本地 smoke 和服务器正式实验脚本混在同一层目录。

当前结构：

- `local/`：本地 smoke test 脚本
- `a100/`：A100 正式实验脚本

覆盖范围：

- 数据：`ETTh1`、`ETTh2`、`electricity`、`traffic`、`weather`
- 模型：`sundial-base-128m`
- 环境：本地 smoke、A100 正式实验
- 概率预测：`etth1_probabilistic*.sh` 用于显式测试多样本生成、分位数和预测区间产物

命名约定：

- `etth1.sh`、`etth2.sh`：ETT 数据集脚本
- `etth1_probabilistic.sh`、`etth1_probabilistic_a100.sh`：显式概率预测脚本
- `*_smoke.sh`：本地轻量验证脚本
- `*_a100.sh`：服务器正式实验脚本

所有脚本都支持用环境变量覆盖默认参数，例如：

```bash
DEVICE=mps DTYPE=float32 bash scripts/sundial/local/etth1.sh
DEVICE=cpu NUM_SAMPLES=2 CONTEXT_LENGTH=256 bash scripts/sundial/local/weather_smoke.sh
DEVICE=cuda DTYPE=bfloat16 NUM_SAMPLES=32 SAMPLE_LIMIT=1024 bash scripts/sundial/a100/traffic_a100.sh
```

后续新增脚本时，继续按这套分层规则放置。
