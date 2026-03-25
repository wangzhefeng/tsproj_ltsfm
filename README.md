# tsproj_ltsfm

> Large Time Series Forecast Model（Foundation Model）

本项目围绕两个时间序列基础模型开展方法调研、脚本接入和实验管理：

- `Time-MoE`
- `Sundial`

当前仓库采用“两级实验”策略：

- 本地开发机：负责代码开发、本地 smoke test、数据流验证、README 命令校验。
- A100 8GPU 服务器：负责较大 checkpoint、较长上下文、较多样本的正式对比实验。

当前已将已下载的预训练模型副本复制到 `pretrain_models/`：

- `pretrain_models/TimeMoE-50M`
- `pretrain_models/TimeMoE-200M`
- `pretrain_models/sundial-base-128m`

体积说明见 `pretrain_models/README.md`。

---

# 1.模型方法、原理调研报告

## Time-MOE

参考资料：

- Paper: [Time-MoE: Billion-Scale Time Series Foundation Models with Mixture of Experts](https://arxiv.org/abs/2409.16040)
- GitHub: <https://github.com/Time-MoE/Time-MoE>
- Hugging Face:
  - `50M`: <https://huggingface.co/Maple728/TimeMoE-50M>
  - `200M`: <https://huggingface.co/Maple728/TimeMoE-200M>
- Blog: [全球首次！时序大模型突破十亿参数，华人团队发布Time-MoE，预训练数据达3000亿个时间点](https://mp.weixin.qq.com/s/LaYn0IJAOlN9Ufp_qus96Q)

### 1.1 模型结构

`Time-MoE` 是面向时间序列预测的自回归基础模型，核心思想是将语言模型中的 `Mixture of Experts (MoE)` 结构迁移到连续值序列建模任务中。其主要特点包括：

- 使用 decoder-only 自回归结构，将时间序列按时间步输入，逐步预测未来值。
- 在前馈层引入 MoE 路由，使不同专家网络学习不同模式的时间变化、周期性与趋势变化。
- 对外提供 `50M`、`200M` 等公开 checkpoint，并在论文中讨论更大规模模型。

### 1.2 训练目标与数据

- 训练目标本质上是连续值的自回归预测，不是离散 token 分类。
- 官方提出 `Time-300B` 预训练语料，覆盖大规模、多领域时间序列。
- 论文和仓库将其定位为时间序列 foundation model，强调 zero-shot / transfer forecasting 能力。

### 1.3 推理范式

- 推理时通常先取历史窗口 `context`。
- 对窗口做逐序列标准化，然后进行未来步长生成，再做逆标准化恢复原尺度。
- 适合统一的滑窗评测接口，因此本仓库将其接入为通用 CLI benchmark。

### 1.4 优势

- MoE 结构提升模型容量，同时允许更灵活地适配不同时间序列模式。
- 官方已给出公开 checkpoint 与 benchmark 脚本，工程复现入口相对明确。
- 与统一滑窗评测方式天然兼容，便于和其他 TSFM 横向比较。

### 1.5 局限

- 官方远程代码与较新的 `transformers` 在 `generate()` 路径上存在兼容性风险，本仓库已通过直接自回归前向推理做兼容适配。
- 大模型与较长上下文在 CPU 上推理较慢，正式实验更适合放在 A100 服务器上执行。
- 目前本仓库的 benchmark 接口按“单目标列预测”实现，对于多变量数据集默认抽取一个目标列进行对比。

### 1.6 适用场景

- 长期预测（long-term forecasting）
- 零样本或少样本 forecasting 基线
- 多数据集统一评测与 foundation model 横向比较

## Sundial(日晷)

参考资料：

- Paper: [Sundial: A Family of Highly Capable Time Series Foundation Models](https://arxiv.org/abs/2502.00816)
- GitHub: <https://github.com/thuml/Sundial>
- Hugging Face: <https://huggingface.co/thuml/sundial-base-128m>
- Official Example: [quickstart_zero_shot_generation.ipynb](https://github.com/thuml/Sundial/blob/main/examples/quickstart_zero_shot_generation.ipynb)
- Talk Slides: [ICML2025 Oral Sundial PPT](https://cloud.tsinghua.edu.cn/f/8d526337afde465e87c9/)
- Blog: [AI论文速读 | 日晷(Sundial)：一系列高性能时间序列基础模型](https://zhuanlan.zhihu.com/p/22292409357)

### 1.1 模型结构

`Sundial` 是 THUML 发布的一族时间序列基础模型。它的公开形态更偏向“通用生成式预测器”：

- 仍然采用基于 Transformer 的生成式建模路线。
- 强调 zero-shot forecasting 与概率式样本生成能力。
- 官方公开 `sundial-base-128m` 等推理入口，直接支持从历史上下文生成多个未来样本。

### 1.2 训练目标与数据

- 官方论文将其作为 family of foundation models 介绍，强调统一的时间序列建模能力。
- 论文与模型卡强调其在 zero-shot 评测中的竞争力，尤其是跨数据集泛化表现。
- 从工程使用角度看，`Sundial` 比较接近“输入历史序列，直接采样未来轨迹”的生成式接口。

### 1.3 推理范式

- 输入固定长度的历史窗口。
- 通过 `num_samples` 生成多个未来候选轨迹，再聚合为均值预测或保留分布信息。
- 本仓库 benchmark 现在同时保留生成样本分布，除均值点预测外，还会落盘分位数、预测区间和概率指标。

### 1.4 优势

- 官方公开结果强调 zero-shot 能力强，适合作为 foundation model 对比基线。
- 天然支持多样本生成，除了点预测，还可以进一步扩展为不确定性分析。
- 与 Time-MoE 一样，可以接入统一滑窗评测框架。

### 1.5 局限

- 本地 CPU 冷启动和首次权重加载较慢，正式实验更推荐直接在 A100 服务器上执行。
- 与不同版本 `transformers` 的兼容性需要额外关注，建议在服务器上使用隔离环境。
- 当前仓库实现默认将多变量数据集转为“单目标列”评测，尚未扩展到联合多变量生成。

### 1.6 适用场景

- 零样本 forecasting
- 生成式未来轨迹采样
- 与其他 TSFM 在统一滑窗协议下的横向比较

---

# 2.模型使用方法及脚本

## 2.1 环境说明

### 本地开发机

适合执行：

- 数据读取与窗口切分验证
- CLI 和结果落盘验证
- 小样本 smoke test
- MacBook Pro M 系列上的 MPS 推理验证

当前项目依赖见 `pyproject.toml`，模型推理入口基于：

- `torch`
- `transformers`
- `pandas`
- `numpy`
- `matplotlib`

推荐先在项目根目录执行：

```bash
uv sync
```

设备选择策略：

- `--device auto`：优先 `CUDA`，其次 `MPS`，最后 `CPU`
- 在 MacBook Pro M2 上，若当前 `torch` 构建支持 MPS，则会自动走 `mps`
- 在 NVIDIA 服务器上，`--device auto --device-map auto` 会优先走 `cuda`

### A100 8GPU 服务器

适合执行：

- 更大 checkpoint
- 更长 context length
- 更大 sample limit
- 正式对比实验与吞吐/显存记录

建议额外准备：

- `accelerate`
- 独立 Python 环境，尽量对齐官方 README / model card 推荐版本
- 充足的 Hugging Face 缓存目录
- 如需多卡执行，可先用 `CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7` 固定 GPU 可见范围
- 建议把 `HF_HOME`、`HF_HUB_CACHE` 指向服务器上的大容量磁盘

## 2.2 统一 Benchmark CLI

本仓库新增了统一的数据和结果接口：

- `data_provider/benchmark_dataset.py`
- `utils/forecasting.py`
- `models/time_moe_usage/run_benchmark.py`
- `models/sundial_usage/run_benchmark.py`

统一参数约定：

- `--data`：数据文件路径，支持 `csv/jsonl/zip`
- `--target-col`：目标列名
- `--time-col`：时间列名，默认自动识别
- `--context-length`：输入窗口长度
- `--prediction-length`：预测步长
- `--stride`：滑窗步长
- `--sample-limit`：样本窗口上限
- `--batch-size`：batch 大小
- `--checkpoint`：Hugging Face 模型名或本地模型目录
- `--device`：`auto` / `cpu` / `cuda` / `mps`
- `--device-map`：`none` / `auto`
- `--dtype`：`auto` / `float32` / `float16` / `bfloat16`
- `--env`：`local` 或 `a100`
- `--output-dir`：结果输出目录

结果输出统一包含：

- `config.json`
- `metrics.json`
- `predictions.csv`
- `summary.md`
- `plot.png`（可选）

## 2.3 Time-MoE 使用方法

### 脚本分类说明

`scripts/time_moe/` 按“运行环境 + 模型规模”分类：

- `scripts/time_moe/local/50m/`：`TimeMoE-50M` 本地轻量验证
- `scripts/time_moe/local/200m/`：`TimeMoE-200M` 本地轻量验证
- `scripts/time_moe/a100/50m/`：`TimeMoE-50M` A100 正式实验
- `scripts/time_moe/a100/200m/`：`TimeMoE-200M` A100 正式实验

命名规则：

- `etth1.sh`、`etth2.sh`：ETT 数据集脚本
- `*_smoke.sh`：本地轻量验证脚本
- `*_a100.sh`：服务器正式实验脚本
- `*_200m*.sh`：使用 `pretrain_models/TimeMoE-200M`

### 本地 smoke test

`scripts/time_moe/local/50m/`：

- `scripts/time_moe/local/50m/etth1.sh`
- `scripts/time_moe/local/50m/etth2.sh`
- `scripts/time_moe/local/50m/electricity_smoke.sh`
- `scripts/time_moe/local/50m/traffic_smoke.sh`
- `scripts/time_moe/local/50m/weather_smoke.sh`

`scripts/time_moe/local/200m/`：

- `scripts/time_moe/local/200m/etth1_200m.sh`
- `scripts/time_moe/local/200m/etth2_200m.sh`
- `scripts/time_moe/local/200m/electricity_200m_smoke.sh`
- `scripts/time_moe/local/200m/traffic_200m_smoke.sh`
- `scripts/time_moe/local/200m/weather_200m_smoke.sh`

示例：

```bash
python models/time_moe_usage/run_benchmark.py \
  --data dataset/ETTh1.csv \
  --dataset-name ETTh1 \
  --target-col OT \
  --context-length 128 \
  --prediction-length 24 \
  --stride 256 \
  --sample-limit 2 \
  --batch-size 1 \
  --checkpoint pretrain_models/TimeMoE-50M \
  --device auto \
  --dtype auto \
  --env local \
  --output-dir results/time_moe/etth1_smoke \
  --save-plot
```

说明：

- 本仓库对 `Time-MoE` 做了 `transformers 5.x` 兼容适配，避免直接走官方 `generate()` 路径。
- 默认 checkpoint 使用项目内的 `pretrain_models/TimeMoE-50M`。
- 如需测试更大规模 checkpoint，可改用 `pretrain_models/TimeMoE-200M`。
- `--device auto` 会自动适配 `cuda/mps/cpu`。
- 所有脚本都支持通过环境变量覆盖默认参数，例如 `DEVICE=mps DTYPE=float32 CONTEXT_LENGTH=256 bash scripts/time_moe/local/50m/etth1.sh`。
- 正式结果建议在 A100 上跑更大配置。

### A100 正式实验

`scripts/time_moe/a100/50m/`：

- `scripts/time_moe/a100/50m/etth1_a100.sh`
- `scripts/time_moe/a100/50m/etth2_a100.sh`
- `scripts/time_moe/a100/50m/electricity_a100.sh`
- `scripts/time_moe/a100/50m/traffic_a100.sh`
- `scripts/time_moe/a100/50m/weather_a100.sh`

`scripts/time_moe/a100/200m/`：

- `scripts/time_moe/a100/200m/etth1_200m_a100.sh`
- `scripts/time_moe/a100/200m/etth2_200m_a100.sh`
- `scripts/time_moe/a100/200m/electricity_200m_a100.sh`
- `scripts/time_moe/a100/200m/traffic_200m_a100.sh`
- `scripts/time_moe/a100/200m/weather_200m_a100.sh`

建议命令：

```bash
python models/time_moe_usage/run_benchmark.py \
  --data dataset/ETTh1.csv \
  --dataset-name ETTh1 \
  --target-col OT \
  --context-length 1024 \
  --prediction-length 96 \
  --stride 96 \
  --sample-limit 512 \
  --batch-size 32 \
  --checkpoint pretrain_models/TimeMoE-50M \
  --device cuda \
  --device-map auto \
  --dtype bfloat16 \
  --env a100 \
  --output-dir results/time_moe/etth1_a100 \
  --save-plot
```

如需测试 `200M` 模型，可直接执行：

```bash
bash scripts/time_moe/local/200m/etth1_200m.sh
bash scripts/time_moe/a100/200m/etth1_200m_a100.sh
```

扩展到更多数据集时，可直接使用：

```bash
bash scripts/time_moe/a100/50m/etth2_a100.sh
bash scripts/time_moe/a100/50m/electricity_a100.sh
bash scripts/time_moe/a100/50m/traffic_a100.sh
bash scripts/time_moe/a100/50m/weather_a100.sh
bash scripts/time_moe/a100/200m/etth2_200m_a100.sh
bash scripts/time_moe/a100/200m/electricity_200m_a100.sh
bash scripts/time_moe/a100/200m/traffic_200m_a100.sh
bash scripts/time_moe/a100/200m/weather_200m_a100.sh
```

## 2.4 Sundial 使用方法

### 脚本分类说明

`scripts/sundial/` 按“运行环境 + 功能类型”分类：

- `scripts/sundial/local/`：本地脚本
- `scripts/sundial/a100/`：A100 正式实验脚本

命名规则：

- `etth1.sh`、`etth2.sh`：常规点预测脚本
- `*_smoke.sh`：本地轻量验证脚本
- `*_a100.sh`：服务器正式实验脚本
- `etth1_probabilistic.sh`、`etth1_probabilistic_a100.sh`：显式概率预测脚本，保留样本分布并生成分位数与预测区间产物

### 本地 smoke test

`scripts/sundial/local/`：

- `scripts/sundial/local/etth1.sh`
- `scripts/sundial/local/etth1_probabilistic.sh`
- `scripts/sundial/local/etth2.sh`
- `scripts/sundial/local/electricity_smoke.sh`
- `scripts/sundial/local/traffic_smoke.sh`
- `scripts/sundial/local/weather_smoke.sh`

示例：

```bash
python models/sundial_usage/run_benchmark.py \
  --data dataset/ETTh1.csv \
  --dataset-name ETTh1 \
  --target-col OT \
  --context-length 128 \
  --prediction-length 24 \
  --stride 512 \
  --sample-limit 1 \
  --batch-size 1 \
  --num-samples 4 \
  --checkpoint pretrain_models/sundial-base-128m \
  --device auto \
  --dtype auto \
  --env local \
  --output-dir results/sundial/etth1_auto_device_smoke \
  --save-plot
```

说明：

- `Sundial` 支持多样本未来轨迹生成，本仓库在保留均值点预测指标的同时，也会保存样本分布、分位数和预测区间。
- 默认 checkpoint 使用项目内的 `pretrain_models/sundial-base-128m`。
- 所有脚本都支持通过环境变量覆盖默认参数，例如 `DEVICE=mps NUM_SAMPLES=2 bash scripts/sundial/local/etth1.sh`。
- 本地冷启动仍可能较慢；如果追求更快反馈，建议直接迁移到 A100 服务器上跑。
- 官方 `quickstart_zero_shot_generation.ipynb` 展示的是概率预测范式：生成多条未来样本，再查看 `mean`、`quantile` 和 `prediction interval`；本仓库已按同样思路补齐对应落盘产物。

### A100 正式实验

`scripts/sundial/a100/`：

- `scripts/sundial/a100/etth1_a100.sh`
- `scripts/sundial/a100/etth1_probabilistic_a100.sh`
- `scripts/sundial/a100/etth2_a100.sh`
- `scripts/sundial/a100/electricity_a100.sh`
- `scripts/sundial/a100/traffic_a100.sh`
- `scripts/sundial/a100/weather_a100.sh`

建议命令：

```bash
python models/sundial_usage/run_benchmark.py \
  --data dataset/ETTh1.csv \
  --dataset-name ETTh1 \
  --target-col OT \
  --context-length 2880 \
  --prediction-length 96 \
  --stride 96 \
  --sample-limit 512 \
  --batch-size 16 \
  --num-samples 20 \
  --checkpoint pretrain_models/sundial-base-128m \
  --device cuda \
  --device-map auto \
  --dtype bfloat16 \
  --env a100 \
  --output-dir results/sundial/etth1_a100 \
  --save-plot
```

### A100 执行建议

推荐流程：

1. 将 `dataset/`、`pretrain_models/`、`models/`、`scripts/`、`utils/` 同步到服务器。
2. 在服务器上先验证 `python models/time_moe_usage/run_benchmark.py --help` 和 `python models/sundial_usage/run_benchmark.py --help`。
3. 先跑 `scripts/time_moe/a100/50m/etth1_a100.sh` 与 `scripts/sundial/a100/etth1_a100.sh`，确认标准对比实验可落盘。
4. 再扩展到 `traffic`、`electricity` 等更大数据脚本。

推荐环境变量示例：

```bash
export HF_HOME=/path/to/hf_cache
export HF_HUB_CACHE=/path/to/hf_cache/hub
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
```

更完整的服务器执行步骤见 `A100_RUNBOOK.md`。

---

# 3.模型数据说明

本仓库当前将所有评测数据统一放在 `dataset/` 根目录下。对多变量数据集，当前实现默认通过 `--target-col` 选取一个目标列作为单变量预测任务。

| 数据文件 | 记录数 | 字段数 | 推荐目标列 | 说明 |
| --- | ---: | ---: | --- | --- |
| `dataset/ETTh1.csv` | 17420 | 8 | `OT` | ETT 基准之一，适合做快速 long-term forecasting smoke test |
| `dataset/electricity.csv` | 26304 | 322 | `0` | 电力负载，多变量宽表，可抽单列做统一比较 |
| `dataset/traffic.csv` | 17544 | 863 | `0` | 交通流量，多变量宽表，适合服务器正式实验 |
| `dataset/weather.csv` | 52696 | 22 | `T (degC)` | 天气数据，适合本地和服务器两级验证 |
| `dataset/global_temp_test.jsonl` | 1000 条序列 | 1 | `sequence` | JSONL 序列列表格式，适合后续扩展非表格输入 |

数据组织策略：

- 本地阶段优先使用 `ETTh1` 与 `weather` 做通路验证。
- 服务器阶段优先使用 `ETTh1`、`traffic`、`electricity` 做正式 benchmark。
- 不强制把大模型权重和所有原始数据纳入 git；仓库内重点保留脚本、配置与结果摘要。

---

# 4.模型测试结果说明

## 4.1 结果文件位置

### Time-MoE

- 本地 smoke 结果：`results/time_moe/etth1_smoke/`
- A100 正式实验待执行：`results/time_moe/etth1_a100_pending.md`
- TimeMoE-200M A100 待执行：`results/time_moe/etth1_200m_a100_pending.md`
- TimeMoE-200M 扩展数据待执行：
  - `results/time_moe/etth2_200m_a100_pending.md`
  - `results/time_moe/electricity_200m_a100_pending.md`
  - `results/time_moe/traffic_200m_a100_pending.md`
  - `results/time_moe/weather_200m_a100_pending.md`

### Sundial

- 本地 smoke 结果：`results/sundial/etth1_auto_device_smoke/`
- 本地概率预测 smoke：`results/sundial/etth1_probabilistic_smoke/`
- A100 正式实验待执行：`results/sundial/etth1_a100_pending.md`

## 4.2 对比表

| 模型 | 官方公开结果（摘要） | 本地小样验证 | A100 正式实验 |
| --- | --- | --- | --- |
| Time-MoE | 官方论文/仓库强调基于 `Time-300B` 预训练，并在长期预测 benchmark 上取得有竞争力的 zero-shot / transfer 表现 | 已完成。`ETTh1`、`target=OT`、`context=128`、`pred=24`、`windows=2`，`MAE=2.9405`，`MSE=14.3027`，`RMSE=3.7819`，`MAPE=0.0890` | 待在 A100 8GPU 上使用更大 checkpoint 或更大样本规模补齐 |
| Sundial | 官方仓库与模型卡强调 `sundial-base-128m` 在零样本评测上表现强，公开摘要中给出在 Time-Series-Library 与 GIFT-Eval 上的领先结果 | 已完成。`ETTh1`、`target=OT`、`context=128`、`pred=24`、`windows=1`、`num_samples=4`，`MAE=1.8097`，`MSE=5.2633`，`RMSE=2.2942`，`MAPE=0.0614` | 待在 A100 8GPU 上用 `pretrain_models/sundial-base-128m` 或服务器本地副本补齐对齐实验 |

## 4.3 本地已完成实验

### Time-MoE 本地 smoke

实验配置：

- 数据集：`ETTh1`
- 目标列：`OT`
- `context_length=128`
- `prediction_length=24`
- `sample_limit=2`
- 模型：`pretrain_models/TimeMoE-50M`
- 环境：`local / cpu / float32`

结果文件：

- `results/time_moe/etth1_smoke/config.json`
- `results/time_moe/etth1_smoke/metrics.json`
- `results/time_moe/etth1_smoke/predictions.csv`
- `results/time_moe/etth1_smoke/plot.png`

说明：

- 该实验是功能验证级别的 smoke test，不代表正式 benchmark。
- 指标可用于确认数据读取、滑窗切分、模型推理和结果落盘链路已经正常。

### Time-MoE 其他本地 smoke

- `ETTh2` 已完成：
  `results/time_moe/etth2_auto_device_smoke/metrics.json`
  指标：`MAE=2.2507`，`MSE=11.9392`，`RMSE=3.4553`，`MAPE=0.0480`
- `electricity` 已完成：
  `results/time_moe/electricity_auto_device_smoke/metrics.json`
  指标：`MAE=36.0702`，`MSE=1831.0802`，`RMSE=42.7911`，`MAPE=1.6020`
- `weather` 已完成：
  `results/time_moe/weather_auto_device_smoke/metrics.json`
  指标：`MAE=5.0280`，`MSE=25.4780`，`RMSE=5.0476`，`MAPE=1.4449`
- `traffic` 当前本地配置下结果为 `NaN`，建议优先在 A100 上复验，或继续调小配置排查数值稳定性
- `TimeMoE-200M` 已加入测试流程：
  - 本地结果文件：`results/time_moe/etth1_200m_auto_device_smoke/metrics.json`
  - 本地保守配置结果：`results/time_moe/etth1_200m_auto_device_smoke_small/metrics.json`
  - 当前两组本地 CPU 测试结果均为 `NaN`，说明 `200M` 在本地环境下数值不稳定，建议优先在 A100 上执行 `scripts/time_moe/a100/200m/etth1_200m_a100.sh`

### Sundial 本地 smoke

实验配置：

- 数据集：`ETTh1`
- 目标列：`OT`
- `context_length=128`
- `prediction_length=24`
- `sample_limit=1`
- `num_samples=4`
- 模型：`pretrain_models/sundial-base-128m`
- 环境：`local / auto -> cpu / float32`

结果文件：

- `results/sundial/etth1_auto_device_smoke/config.json`
- `results/sundial/etth1_auto_device_smoke/metrics.json`
- `results/sundial/etth1_auto_device_smoke/predictions.csv`
- `results/sundial/etth1_auto_device_smoke/sample_predictions.npz`
- `results/sundial/etth1_auto_device_smoke/quantiles.csv`
- `results/sundial/etth1_auto_device_smoke/probabilistic_metrics.json`

说明：

- 该实验同样属于功能验证级别的 smoke test。
- 当前本地环境未启用 `mps`，因此 `--device auto` 自动回退到 `cpu`。
- 由于 `num_samples>1`，当前 `Sundial` benchmark 已同时对概率预测功能进行测试，并保存分位数与预测区间产物。

### Sundial 其他本地 smoke

- `ETTh1` 概率预测 smoke 已完成：
  `results/sundial/etth1_probabilistic_smoke/probabilistic_metrics.json`
  指标：`pinball_loss_q10=0.3400`，`pinball_loss_q50=0.7107`，`pinball_loss_q90=0.4947`，`interval_coverage_q05_q95=0.6250`，`interval_width_q05_q95=4.0157`
- `ETTh2` 已完成：
  `results/sundial/etth2_auto_device_smoke/metrics.json`
  指标：`MAE=2.9528`，`MSE=10.1157`，`RMSE=3.1805`，`MAPE=0.0875`
- `electricity` 已完成：
  `results/sundial/electricity_auto_device_smoke/metrics.json`
  指标：`MAE=20.4728`，`MSE=549.1070`，`RMSE=23.4330`，`MAPE=0.7246`
- `traffic` 已完成：
  `results/sundial/traffic_auto_device_smoke/metrics.json`
  指标：`MAE=0.0852`，`MSE=0.0154`，`RMSE=0.1239`，`MAPE=3.8812`
- `weather` 当前本地配置下结果为 `NaN`，建议优先在 A100 上复验，或后续继续调小配置/排查数值稳定性

## 4.4 A100 正式实验建议

建议至少补齐以下两类实验：

1. 标准对比实验
   - 数据集：`ETTh1`
   - 目标：同一 `context/prediction length` 下比较 `Time-MoE` 与 `Sundial`
2. 更大设置实验
   - 数据集：`traffic` 或 `electricity`
   - 目标：观察更大样本、长上下文、较大 checkpoint 下的精度与吞吐变化

建议额外记录：

- 运行时长
- 单卡/多卡显存占用
- batch size
- 样本数与上下文长度

---

# 5.模型测试结论

基于当前仓库实现和已经完成的本地验证，可以得到以下阶段性结论：

1. 从工程接入角度看，`Time-MoE` 与统一滑窗 benchmark 的耦合更直接，适合作为本项目的第一个稳定基线。
2. `Sundial` 的公开接口更偏生成式、多样本预测，理论上更适合扩展不确定性评估，但正式对比更依赖服务器环境来保证执行效率。
3. 当前本地机已经完成 `Time-MoE` 的 ETTh1 功能性验证，说明仓库的数据读取、统一 CLI、结果产物格式已经可用。
4. 由于 foundation model 的优势往往体现在更长上下文、更大预训练模型和更广数据覆盖上，因此最终结论应以 A100 正式实验为准，而不是以本地 smoke test 为准。
5. 对本项目的落地顺序建议是：
   - 先用本地脚本验证数据与命令；
   - 再将相同命令迁移到 A100；
   - 最后把正式结果同步回 `results/`，并补完 README 的最终结论表。

当前建议的最终 benchmark 主线：

- `Time-MoE-50M/200M` vs `Sundial-base-128m`
- 统一在 `ETTh1` 上做第一轮对比
- 再在 `traffic` 或 `electricity` 上做第二轮扩展实验

这能在“可执行性、可复现性、对比公平性”之间取得较好的平衡。
