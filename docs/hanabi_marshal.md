# Hanabi MARSHAL 接入文档

本文档面向 `mindgames` 仓库，说明如何在你们现有的 `ms-swift + Hanabi gym rollout` 训练链路中使用 MARSHAL 的核心思路。

## 1. 目标和范围

本仓库的实现目标是：

- 在现有训练框架内引入 MARSHAL 的关键机制。
- 保持你们当前 Hanabi 环境和脚本可复用，不做大规模框架迁移。

本仓库当前已接入的 MARSHAL 核心机制：

- Turn-level reward signal（逐回合奖励信号）。
- Agent-specific normalization（按玩家分开的奖励归一化）。

本仓库当前未做的事情：

- 未完整迁移 MARSHAL 原仓库的 ROLL 训练栈。
- 未逐行复刻 MARSHAL 全部 pipeline 细节。

如果你要论文“严格复现”，请走第 2 节；如果你要在你们工程里稳定训练 Hanabi，请走第 3 节。

## 2. 论文严格复现路径（原仓库）

```bash
git clone https://github.com/thu-nics/MARSHAL
cd MARSHAL
# 按原仓库 README 配置 ROLL + OpenSpiel（pyspiel）
bash examples/hanabi/run_agentic_pipeline_hanabi_selfplay.sh
```

说明：

- 这条路径用于复现实验口径和论文数值。
- 与你们 `mindgames` 仓库的工程集成是两条独立路径。

## 3. 在本仓库训练（推荐）

### 3.1 一键方式

先启动 rollout server（终端 1）：

```bash
bash tools/rollout/rollout_hanabi_gym.sh
```

再启动 MARSHAL 风格训练（终端 2）：

```bash
bash tools/train/train_grpo_hanabi_marshal.sh
```

该脚本会自动做两件事：

- 将 `data/hanabi.grpo.jsonl` 转成 `data/hanabi.grpo.marshal.jsonl`。
- 以 MARSHAL 风格默认参数启动训练。

### 3.2 8 卡常用方式（4 rollout + 4 train）

rollout 侧：

```bash
mkdir -p logs
for i in 0 1 2 3; do
  port=$((8000 + i))
  CUDA_VISIBLE_DEVICES=$i \
  HOST=127.0.0.1 PORT=$port \
  CONTEXT_MANAGER=hanabi_recent_turns \
  HANABI_CTX_MAX_TURNS=1 \
  VLLM_TENSOR_PARALLEL_SIZE=1 \
  VLLM_DATA_PARALLEL_SIZE=1 \
  VLLM_MAX_MODEL_LEN=18000 \
  VLLM_MAX_NUM_SEQS=16 \
  bash tools/rollout/rollout_hanabi_gym_simple.sh \
    > "logs/rollout_${port}.log" 2>&1 &
done
```

训练侧：

```bash
CUDA_VISIBLE_DEVICES=4,5,6,7 \
NPROC_PER_NODE=4 \
VLLM_SERVER_HOST=127.0.0.1,127.0.0.1,127.0.0.1,127.0.0.1 \
VLLM_SERVER_PORT=8000,8001,8002,8003 \
VLLM_SERVER_GROUP_PORT=51216,51217,51218,51219 \
bash tools/train/train_grpo_hanabi_marshal.sh
```

## 4. 代码映射（MARSHAL 思想 -> 本仓库实现）

- 逐回合奖励信号：
  - `mindgames/envs/Hanabi/env.py`
  - 关键参数：`marshal_dense_reward`, `marshal_fuse_penalty`, `marshal_invalid_penalty`
  - 行为：每一步计算 `step_reward` 并通过 `step_info` 传给 rollout 插件

- 按玩家归一化：
  - `tools/rollout/hanabi_gym_plugin.py`
  - 关键参数：`marshal_agent_norm`, `marshal_agent_norm_method`, `marshal_agent_norm_warmup`, `marshal_agent_norm_clip`
  - 行为：维护每个玩家独立的 running mean/std，对 reward 做玩家级归一化

- rollout 上下文裁剪（防止 Hanabi scheduler 轨迹上下文叠加）：
  - `tools/rollout/hanabi_gym_plugin.py`
  - 关键参数：`CONTEXT_MANAGER=hanabi_recent_turns`, `HANABI_CTX_MAX_TURNS`
  - 行为：每轮仅保留最近 N 个 user turn（默认 1），避免 observation 历史在 scheduler 中二次累积

- 训练参数透传：
  - `tools/train/train_grpo_base.sh`
  - MARSHAL 相关高级参数通过 `EXTRA_SWIFT_ARGS` 注入到底层 `swift rlhf` 命令

- 一键训练入口：
  - `tools/train/train_grpo_hanabi_marshal.sh`

- 数据集转换工具：
  - `tools/data/prepare_hanabi_marshal_dataset.py`

## 5. MARSHAL 风格参数说明

### 5.1 env_config 参数（写入 JSONL）

由 `tools/data/prepare_hanabi_marshal_dataset.py` 注入：

- `marshal_dense_reward`：是否启用逐回合 dense reward。
- `marshal_fuse_penalty`：每次损失 fuse token 的额外惩罚系数。
- `marshal_invalid_penalty`：非法动作的额外惩罚。
- `marshal_agent_norm`：是否启用按玩家分开的归一化。
- `marshal_agent_norm_method`：`mean` 或 `mean_std`。
- `marshal_agent_norm_warmup`：每个玩家归一化 warmup 样本数。
- `marshal_agent_norm_clip`：归一化后 reward 的裁剪阈值。

### 5.2 训练参数（ms-swift rlhf）

默认由 `train_grpo_hanabi_marshal.sh` 设置：

- `ADVANTAGE_ESTIMATOR=reinforce_plus_plus`
- `SCALE_REWARDS=none`
- `WHITEN_REWARDS=false`
- `VLLM_SERVER_PASS_DATASET=true`
- `LOG_COMPLETIONS=true`

可按实验需要覆盖：

```bash
ADVANTAGE_ESTIMATOR=grpo \
SCALE_REWARDS=batch \
WHITEN_REWARDS=true \
bash tools/train/train_grpo_hanabi_marshal.sh
```

生成批次约束（由底层 `train_grpo_base.sh` 校验）：

- `GENERATION_BATCH_SIZE` 与 `STEPS_PER_GENERATION` 互斥。
- 若设置 `GENERATION_BATCH_SIZE`，必须同时被 `NUM_GENERATIONS` 和 `NPROC_PER_NODE` 整除。

## 6. 数据集准备

手动执行转换脚本示例：

```bash
python tools/data/prepare_hanabi_marshal_dataset.py \
  --input data/hanabi.grpo.jsonl \
  --output data/hanabi.grpo.marshal.jsonl \
  --marshal-dense-reward true \
  --marshal-fuse-penalty 1.0 \
  --marshal-agent-norm true \
  --marshal-agent-norm-method mean_std \
  --marshal-agent-norm-warmup 16 \
  --marshal-agent-norm-clip 4.0
```

## 7. 训练后验证清单

- 数据集检查：
  - `data/hanabi.grpo.marshal.jsonl` 每一行都应包含新增 `env_config` 字段。

- rollout 插件侧检查：
  - 当启用归一化时，`info` 中应出现 `raw_reward`, `reward_norm_method`, `reward_norm_player`。

- 环境侧检查：
  - 当启用 `marshal_dense_reward=true` 时，`step_info` 中应出现 `step_reward`。

- 训练侧检查：
  - 确认脚本实际传入了 `ADVANTAGE_ESTIMATOR/SCALE_REWARDS` 等参数。

## 8. 常见问题

- 问题：奖励几乎全是 0
  - 检查 `marshal_dense_reward` 是否开启
  - 检查 `marshal_agent_norm_warmup` 是否过大
  - 检查 `marshal_agent_norm_method` 是否与当前 reward 分布匹配

- 问题：训练波动大
  - 先关闭 clip 或调小 `marshal_fuse_penalty`
  - 增大 `marshal_agent_norm_warmup`
  - 尝试 `SCALE_REWARDS=batch`

- 问题：rollout 看不到 env_config
  - 确认 `VLLM_SERVER_PASS_DATASET=true`
  - 确认训练数据确实是 `data/hanabi.grpo.marshal.jsonl`

- 问题：gym-scheduler 叠 observation 导致爆 token（如 negative max_tokens）
  - rollout 侧启用：`CONTEXT_MANAGER=hanabi_recent_turns`
  - 先用 `HANABI_CTX_MAX_TURNS=1`，再视训练稳定性调到 2

## 9. 建议实验顺序

建议先跑以下三组，观察稳定性和最终得分：

1. 基线组：`marshal_dense_reward=false`, `marshal_agent_norm=false`
2. 单机制组：`marshal_dense_reward=true`, `marshal_agent_norm=false`
3. 完整组：`marshal_dense_reward=true`, `marshal_agent_norm=true`

这样可以直接看出 MARSHAL 两个核心机制的独立贡献和组合收益。
