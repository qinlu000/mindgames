# Hanabi 自博弈与伙伴池研究（Phase-1）

目标：在不改变环境规则的前提下，提升模型在 Hanabi 的泛化协作能力，而不只在“固定搭档”下高分。

## 为什么“单一搭档训练”会掉泛化
- Hanabi 是高协作、强约定博弈。和一个搭档长期训练后，策略容易收敛到私有约定。
- 私有约定在自对弈里看起来分高，但换搭档就会崩，表现为：
- 分数显著下降。
- 更高 fuse 耗尽率。
- `normalized_action` 异常或失配增多（特别是 think 模型）。

## 研究核心：伙伴池 + 历史策略混合
- 伙伴池（Partner Pool）：每次 rollout 时从多个搭档策略中采样，而不是固定同一个搭档。
- 历史策略（Historical Policies）：把过去若干 checkpoint 当成“旧风格搭档”加入池中，避免只适应最新策略。
- 硬搭档采样（Hard Partner Sampling）：提升与“最难配合搭档”的采样概率，专门补短板。

## 建议采样分布（可直接用）
- `40%` 当前同阶段模型（稳定性）。
- `30%` 历史 checkpoint（抗策略漂移）。
- `30%` hard partners（补弱点）。

可做两阶段课程学习：
- 早期：`60/30/10`（先稳住）。
- 中后期：`40/30/30`（强化泛化）。

## 评估指标（不要只看均分）
- `avg_score_if_coop`：总体协作分。
- Cross-play 均值：在 `N x N` 矩阵上的平均分。
- Cross-play 下分位：例如最差 `10%` 搭档上的分数（稳健性）。
- 动作质量：`invalid_norm_rate`、`unclosed_think_rate`、`invalid_move_episode_rate`。
- 结局结构：`deck_out` vs `fuse_out`。

## 立刻可执行的流程
1. 跑完当前 4 模型 100 局基线（你当前正在进行）。
2. 自动启动 `N x N` cross-play 矩阵，产出 `matrix.tsv` 和 `hard_partners.txt`。
3. 基于 hard partner 集合设计下一轮 partner-pool rollout/训练。

自动化脚本（已提供）：
- `tools/rollout/watch_then_run_partner_pool_matrix.sh`

示例（后台 tmux 启动）：

```bash
ts=$(date +%Y%m%d_%H%M%S)
sess="hanabi_research_phase1_${ts}"
OUT_ROOT="outputs/hanabi_merged_4x100_20260227_155427"
tmux new-session -d -s "$sess" \
  "cd /home/cql/projects/games/mindgames && \
   OUT_ROOT='$OUT_ROOT' EXPECT_MODELS=4 CHECK_INTERVAL=300 \
   MATRIX_CUDA_VISIBLE_DEVICES=0,1,2,3 MATRIX_EPISODES=50 MATRIX_SEED=0 \
   bash tools/rollout/watch_then_run_partner_pool_matrix.sh"
tmux ls | rg \"$sess\"
```

## Phase-1 产物
- 基线评测：`outputs/hanabi_merged_4x100_*/leaderboard.json`
- 交叉矩阵：`outputs/hanabi_partner_pool_from_*/matrix.tsv`
- 难搭档列表：`outputs/hanabi_partner_pool_from_*/hard_partners.txt`

这些结果直接作为 Phase-2 的训练输入：
- 训练时按权重采样伙伴策略。
- 每隔固定 step 冻结一个 checkpoint 入池。
- 定期重跑 cross-play 矩阵，跟踪最差搭档分数是否上升。
