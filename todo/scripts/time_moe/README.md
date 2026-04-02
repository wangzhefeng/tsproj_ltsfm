# Time-MoE Scripts

`scripts/time_moe/` 按运行环境和模型规模分类，避免本地 smoke、A100 正式实验和不同 checkpoint 混在同一层目录。

当前结构：

- `local/50m/`：`TimeMoE-50M` 的本地 smoke test 脚本
- `local/200m/`：`TimeMoE-200M` 的本地 smoke test 脚本
- `a100/50m/`：`TimeMoE-50M` 的 A100 正式实验脚本
- `a100/200m/`：`TimeMoE-200M` 的 A100 正式实验脚本

目录清单：

- `local/50m/`：`etth1.sh`、`etth2.sh`、`electricity_smoke.sh`、`traffic_smoke.sh`、`weather_smoke.sh`
- `local/200m/`：`etth1_200m.sh`、`etth2_200m.sh`、`electricity_200m_smoke.sh`、`traffic_200m_smoke.sh`、`weather_200m_smoke.sh`
- `a100/50m/`：`etth1_a100.sh`、`etth2_a100.sh`、`electricity_a100.sh`、`traffic_a100.sh`、`weather_a100.sh`
- `a100/200m/`：`etth1_200m_a100.sh`、`etth2_200m_a100.sh`、`electricity_200m_a100.sh`、`traffic_200m_a100.sh`、`weather_200m_a100.sh`

覆盖范围：

- 数据：`ETTh1`、`ETTh2`、`electricity`、`traffic`、`weather`
- 模型：`TimeMoE-50M`、`TimeMoE-200M`
- 环境：本地 smoke、A100 正式实验

命名约定：

- `etth1.sh`、`etth2.sh`：ETT 数据集脚本
- `*_smoke.sh`：本地轻量验证脚本
- `*_a100.sh`：服务器正式实验脚本
- `*_200m*.sh`：使用 `pretrain_models/TimeMoE-200M`

所有脚本都支持用环境变量覆盖默认参数，例如：

```bash
DEVICE=mps DTYPE=float32 bash scripts/time_moe/local/50m/etth1.sh
DEVICE=cpu CONTEXT_LENGTH=256 PREDICTION_LENGTH=48 bash scripts/time_moe/local/200m/etth2_200m.sh
DEVICE=cuda DTYPE=bfloat16 SAMPLE_LIMIT=1024 bash scripts/time_moe/a100/50m/traffic_a100.sh
```

后续新增脚本时，优先保持这套分层规则，不再回到扁平目录。
