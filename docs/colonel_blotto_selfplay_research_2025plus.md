# Colonel Blotto: 用 self-play 提升 LLM（2025+ 调研与落地草案）

Date: 2026-03-19

Scope:

- 本文只调研 2025 年及之后、且对 `Colonel Blotto + LLM self-play` 有直接借鉴价值的工作。
- 重点不是“Blotto 专项大而全综述”，而是筛出最值得落地到本仓库的训练与评估思路。
- 下文里“对 Blotto 的启发”属于基于论文的设计推断，不是论文直接结论。
- 本文默认面向当前仓库里的 `mindgames/envs/ColonelBlotto/env.py`：2 人、同时分配、每回合一次 allocation、默认 `3 fields / 20 units / 10 rounds`。

## 一句话结论

2025 年后，我还没有看到“直接用 Colonel Blotto self-play 把 LLM 训强”的成熟代表作；但已经有两条非常可借鉴的研究线：

1. `LLM self-play / multi-agent RL`
2. `Blotto / polyhedral game learning`

如果目标是把当前仓库里的 Colonel Blotto 做成一个能稳定训练、可评估、可扩展的 LLM self-play 任务，最值得借鉴的组合是：

1. `SPIRAL`：历史快照池 + 持续自博弈 curriculum
2. `MARSHAL`：多回合 / 多智能体 credit assignment
3. `STRATEGIST`：文本策略 + 搜索执行的双层架构
4. `SPPO / RSPO`：把 self-play 当作逼近 Nash 的过程，并用正则防止模式坍塌
5. `Polyhedral game learning`：别只看 win-rate，要看 regret / exploitability / equilibrium gap
6. `Large-scale Blotto MARL`：给未来扩到 heterogeneous / team / large-scale 版本提供路线

## 为什么当前仓库适合做这件事

结合 `mindgames/envs/ColonelBlotto/env.py`，当前环境有几个很适合 self-play 研究的特点：

- 这是一个对称的 2-player simultaneous allocation game。
- 每回合动作就是一个合法 allocation，动作格式和环境校验都已经有了。
- 默认配置很小，便于先做 toy setting 上的精确或近似 `best response / exploitability` 评估。
- 这是一个重复博弈：单回合是同时行动，整局是多回合累积分数，因此既能做单步策略分析，也能做历史对手池自博弈。

这也意味着，当前仓库里的 Blotto 与 Hanabi 不同，主要难点不是 long-context memory，而是：

- 混合策略是否塌缩
- 是否学会对不同对手分布做适应
- 是否能把“高层分配原则”转成更强的具体 allocation
- 是否能用比 win-rate 更稳健的指标判断策略质量

## 最值得借鉴的 6 篇

### 1. SPIRAL

- 论文：`SPIRAL: Self-Play on Zero-Sum Games Incentivizes Reasoning via Multi-Agent Multi-Turn Reinforcement Learning`
- 时间：arXiv 首版 2025-06-30；arXiv 当前页显示 Accepted at ICLR 2026
- 链接：<https://arxiv.org/abs/2506.24119>
- 论文直接信息：
  - 把 LLM 放进 zero-sum、multi-turn 游戏里做 fully online self-play
  - 核心机制之一是持续自博弈产生的 opponent curriculum
  - 提出了 `role-conditioned advantage estimation (RAE)` 来稳定多智能体训练
- 对 Blotto 的启发（设计推断）：
  - 不要只让当前 policy 和一个固定 baseline 对打
  - 应该维护一个 `historical checkpoint pool`，让当前模型和“自己过去的版本 + 当前版本”混合对打
  - 这样更不容易过拟合到少数 allocation 模式，也更接近 Blotto 里需要的 mixed-strategy 学习
- 对当前仓库的直接落地：
  - 训练时让 `Commander Alpha / Commander Beta` 都从策略池中采样
  - 每隔固定 step 冻结一个 snapshot，加入 opponent pool
  - 评估时单独报告 `vs latest`, `vs pool`, `cross-play matrix`

### 2. MARSHAL

- 论文：`MARSHAL: Incentivizing Multi-Agent Reasoning via Self-Play with Strategic LLMs`
- 时间：arXiv 首版 2025-10-17
- 链接：<https://arxiv.org/abs/2510.15414>
- 论文直接信息：
  - 面向 cooperative + competitive 的 multi-agent strategic games
  - 核心是 `turn-level advantage estimator`
  - 以及 `agent-specific advantage normalization`
- 对 Blotto 的启发（设计推断）：
  - 对当前最小版 simultaneous Blotto，这套 credit assignment 不是第一优先级
  - 但如果后续扩到 `team Blotto`、`heterogeneous Blotto`、`sequential allocation Blotto`，或者允许模型先写计划再逐步落子，这种 turn-level / agent-level 归因会非常重要
  - 即使保留当前 simultaneous 设定，也可以借鉴“按 player / role 单独归一化 advantage”的思路，减少训练抖动
- 对当前仓库的直接落地：
  - 第一阶段可只借鉴 `per-role normalization`
  - 第二阶段若做“分多步生成 allocation”或“team version”，再引入更细的 turn-level credit assignment

### 3. STRATEGIST

- 论文：`Strategist: Self-improvement of LLM Decision Making via Bi-Level Tree Search`
- 时间：OpenReview 显示 Published 2025-01-22，ICLR 2025 Poster
- 链接：<https://openreview.net/forum?id=gfI9v7AbFg>
- 论文直接信息：
  - 不是直接训练一个 end-to-end policy
  - 让 LLM 生成和更新高层策略文本
  - 用 `MCTS` 做执行和 refinement
  - 通过 self-play 模拟不断更新策略
- 对 Blotto 的启发（设计推断）：
  - 这和 Blotto 很搭，因为 Blotto 的关键瓶颈本来就不是自然语言，而是“组合搜索 + mixed strategy”
  - 一个更合理的结构不是“LLM 直接吐一个 allocation 然后结束”，而是：
    1. 先输出高层分配原则
    2. 再用局部搜索 / sampling / tree search 生成候选 allocation
    3. 用 self-play 或 best-response-style evaluator 挑出更强动作
- 对当前仓库的直接落地：
  - 可以把当前 action space 保持不变，但在 agent 侧增加一个 `strategy -> executor` 两层接口
  - `strategy` 负责产出“应该均衡分配 / 应该在关键 battlefield 做尖峰 / 应该避免被镜像 exploit”等文本计划
  - `executor` 负责把文本策略映射成若干合法 allocation 候选，再筛选一个提交给环境

### 4. SPPO / RSPO

- 论文 1：`Self-Play Preference Optimization for Language Model Alignment`
- 时间：ICLR 2025
- 链接：<https://proceedings.iclr.cc/paper_files/paper/2025/hash/e48fa1c4f08fd1ae35d5df8352c3106d-Abstract-Conference.html>
- 论文 2：`RSPO: Regularized Self-Play Alignment of Large Language Models`
- 时间：arXiv 首版 2025-02-24
- 链接：<https://arxiv.org/abs/2503.00030>
- 论文直接信息：
  - SPPO 把 self-play alignment 视为一个两人 constant-sum game 上逼近 Nash equilibrium 的过程
  - RSPO 进一步系统研究 reference regularization，并给出 forward KL / reverse KL 及其组合的效果
- 对 Blotto 的启发（设计推断）：
  - 这是当前最应该借到 Blotto 的“训练稳定性”思路
  - Blotto 天然依赖 mixed strategy；如果没有足够的 entropy / KL / population regularization，policy 很容易塌成少数几个 allocation template
  - 一旦塌缩，短期 self-play win-rate 可能还不错，但通常会被专门的 exploiter 打穿
- 对当前仓库的直接落地：
  - 训练目标里保留对 reference policy 的 KL 正则
  - 同时额外监控 action entropy、支持集大小、历史池 cross-play 表现
  - 如果后续把 allocation 生成改成多样采样，还可以直接把“response diversity”当成辅助健康指标

### 5. Efficient Kernelized Learning in Polyhedral Games beyond Full Information

- 论文：`Efficient Kernelized Learning in Polyhedral Games beyond Full Information: From Colonel Blotto to Congestion Games`
- 时间：OpenReview 显示 Published 2025-09-18，NeurIPS 2025 Poster
- 链接：<https://openreview.net/forum?id=FUBaZDMOFj>
- 论文直接信息：
  - 研究 polyhedral games 在 payoff-only / partial-information 场景下高效学习 `CCE`
  - 明确把 Colonel Blotto 作为代表问题之一
  - 重点不是 LLM，而是在巨大组合动作空间里如何做高效 no-regret learning
- 对 Blotto 的启发（设计推断）：
  - 这篇最重要的价值是提醒评估标准不要只盯着 win-rate
  - 如果真想说“学到了更好的 Blotto 策略”，更有说服力的指标是：
    - `best-response exploitability`
    - `external regret`
    - `CCE / equilibrium gap`
  - 在小规模 toy Blotto 上，甚至应该优先做能精确算或近似算这些指标的实验
- 对当前仓库的直接落地：
  - 新增 `toy Blotto eval`：把 `num_total_units` 调小到可以枚举所有 allocation
  - 用枚举或近似 best response 评估每个 checkpoint 的 exploitability
  - 把“self-play Elo 上升但 exploitability 变差”的情况显式抓出来

### 6. Multi-Agent Reinforcement Learning for Heterogeneous Large-Scale Blotto Games

- 论文：`Multi-Agent Reinforcement Learning for Heterogeneous Large-Scale Blotto Games`
- 时间：OpenReview 显示 2025-09-19 提交，Submitted to ICLR 2026
- 链接：<https://openreview.net/forum?id=QBGVlffCzf>
- 论文直接信息：
  - 直接研究 large-scale heterogeneous Blotto
  - 目标场景扩展到 thousands of agents / dozens of battlefields
  - 提出 `Group-Mix` 做 type-aware value decomposition
  - 提出 `H-PPO` 做 hierarchical curriculum learning
- 对 Blotto 的启发（设计推断）：
  - 这篇和当前仓库的 2-player toy Blotto 不是一个规模，但它提供了明确的扩展路线
  - 如果后续要从“单个 LLM 决策一个 allocation”扩到“多 unit type / 多指挥官 / team Blotto / heterogeneous battlefield value”，可以直接借它的参数共享和层级 curriculum 思路
- 对当前仓库的直接落地：
  - 不是现在第一优先级
  - 更适合作为 `v2 / v3` 规划依据：当你把当前环境扩到 multi-colonel 或 heterogeneous version 时，再引入 value factorization / curriculum 设计

## 把这些研究翻译成当前仓库的训练路线

### Priority 1: 先把评估做对

在当前仓库里，最先该补的不是花哨 agent 结构，而是评估。

建议至少固定输出下面几类指标：

- `self-play win-rate`
- `cross-play vs historical checkpoints`
- `best-response exploitability`
- `policy entropy / allocation diversity`
- `seed variance`

原因很简单：Blotto 是一个混合策略游戏，只看 win-rate 很容易误判。

### Priority 2: 历史快照池 self-play

先按 SPIRAL 的思路做最小闭环：

1. 当前 policy 与历史快照池混合对战
2. 周期性冻结 checkpoint
3. 训练集持续由在线 rollout 生成
4. 评估时跑 cross-play matrix

这是我认为当前仓库里最值得先实现的部分，因为它对环境改动最小，但能显著降低“只会打当前自己”的风险。

### Priority 3: 加上正则，避免策略塌缩

按 SPPO / RSPO 的启发，至少保留两类约束：

- `reference KL`
- `entropy / diversity monitoring`

如果后续观察到策略塌成固定模板，可以进一步考虑：

- population-level regularization
- 对 allocation 分布的支持集做约束
- 用近似 exploiter 专门攻击低熵策略

### Priority 4: 把“直接出动作”升级成“策略 + 执行器”

这是 STRATEGIST 对 Blotto 最值钱的启发。

建议不要把最终形态限制为“模型一次性吐出 `[A4 B2 C14]`”。更强的结构通常会是：

1. `planner`：输出高层分配原则
2. `executor`：根据原则生成若干合法 allocation
3. `selector`：通过局部搜索、对手模型或 rollout value 选择最终动作

这样做的好处是：

- 更适合组合动作空间
- 更容易注入搜索
- 更容易做 preference pair 或 refinement 数据
- 更容易分析“模型到底学到了什么策略原则”

### Priority 5: 只在需要时引入更复杂的 credit assignment

MARSHAL 的 turn-level credit assignment 很强，但对当前 simultaneous 版本不是第一优先级。

我的建议是：

- 当前 2-player simultaneous 版本：先做 `SPIRAL + RSPO + STRATEGIST-style executor`
- 如果扩到 team / sequential / hierarchical allocation：再系统引入 `MARSHAL-style` credit assignment

### Priority 6: 把 large-scale heterogeneous Blotto 作为中长期扩展

如果未来你想研究：

- 多个 colonel 协同分配
- 不同 unit type
- 不同 battlefield value
- 大规模 resource allocation

那就可以把 heterogeneous large-scale Blotto 这篇当作架构参考，而不是现在就把问题复杂化。

## 我建议的最小可跑方案

如果目标是“先在这个仓库里做出一个靠谱的 Colonel Blotto self-play baseline”，我建议按下面顺序推进：

1. `toy setting`
   - 先把 `num_fields` 和 `num_total_units` 调到可枚举范围
   - 建立 exact / approximate best-response evaluator
2. `regularized self-play`
   - 当前策略 vs 历史池
   - 加 KL / entropy 监控
3. `strategy-text executor`
   - 先输出高层策略，再做候选 allocation 搜索
4. `cross-play + exploitability report`
   - 每个 checkpoint 固定输出同一套报告

这条路线的优点是：

- 每一步都能单独验证
- 不需要一开始就上复杂 multi-agent credit assignment
- 能较早发现“win-rate 上升但策略更脆弱”的假进步

## 对本仓库的直接实现建议

如果后面真的要开始做代码，我会优先改下面几层：

- `mindgames/envs/ColonelBlotto/env.py`
  - 保持环境最小化，不急着把搜索逻辑塞进 env
- `mindgames/agents/`
  - 新增一个 Blotto 专用 agent，支持 `direct-allocation` 和 `strategy-plus-executor` 两种模式
- `tools/run_rollouts.py`
  - 增加 historical checkpoint pool / cross-play matrix 的 rollout 入口
- `experiments/`
  - 单独建 Colonel Blotto 的 self-play 配置，而不是复用 Hanabi 实验名
- `docs/`
  - 后续再补一份 `colonel_blotto_selfplay_plan.md`，把训练流、评估流和实验矩阵固定下来

## 参考链接

- SPIRAL: <https://arxiv.org/abs/2506.24119>
- MARSHAL: <https://arxiv.org/abs/2510.15414>
- STRATEGIST: <https://openreview.net/forum?id=gfI9v7AbFg>
- SPPO: <https://proceedings.iclr.cc/paper_files/paper/2025/hash/e48fa1c4f08fd1ae35d5df8352c3106d-Abstract-Conference.html>
- RSPO: <https://arxiv.org/abs/2503.00030>
- Efficient Kernelized Learning in Polyhedral Games beyond Full Information: <https://openreview.net/forum?id=FUBaZDMOFj>
- Multi-Agent Reinforcement Learning for Heterogeneous Large-Scale Blotto Games: <https://openreview.net/forum?id=QBGVlffCzf>
