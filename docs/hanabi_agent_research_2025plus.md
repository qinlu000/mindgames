# Hanabi: 从 LLM policy 到 agent system（2025+ 调研与设计路线）

Date: 2026-03-18

Scope:

- 本文只调研 2025 年及之后的 agent 相关工作为主。
- 不调研 Hanabi 专项论文；这里只看通用 agent 研究，再映射到 Hanabi。
- 下文里“对 Hanabi 的启发”是基于这些论文做的设计推断，不是论文直接结论。

## 一句话结论

如果你想研究“从单次 LLM policy 变成 agent system”能不能提升 Hanabi 表现，重点不该只是加一个长文本 memory。

2025 年之后比较值得吸收的方向是：

1. typed / hierarchical memory，而不是自由文本日记
2. belief state，而不只是 goal state
3. uncertainty-aware planning，而不只是多想几步
4. verifier / tool layer，而不只是把更多状态塞进 prompt
5. episodic reflection + RL credit assignment，而不只是靠 prompt engineering

对 Hanabi 来说，我最看好的顺序是：

1. `goal memory`（你当前分支已经在做）
2. `belief memory`
3. `uncertainty-aware hint planner`
4. `action verifier`
5. `episode summary / reflection`
6. `partner / convention memory`

如果只在 `memory` 和 “别的东西”之间做优先级排序，那么我会说：

- 比起 generic long-term memory，更应该优先引入 `belief + uncertainty + verifier`
- 比起多 agent 对话式架构，更应该优先引入 `typed state + structured memory ops`

## 为什么这件事在 Hanabi 上特别重要

Hanabi 不是一个“多看上下文就会更强”的任务，而是一个：

- 部分可观测（你看不到自己的牌）
- 强协作
- 强约定
- 动作代价不对称（错误出牌直接掉 fuse）
- 信息行动很重要（Reveal 的价值在于减少关键不确定性）

所以从 LLM 到 agent 的核心，不是让模型记住更多原始文本，而是让它显式维护：

- 目前想做什么：`goal`
- 自己手牌可能是什么：`belief`
- 队友可能怎么解释提示：`partner model / convention`
- 哪些不确定性最危险：`uncertainty`
- 在当前风险下，哪类动作最优：`plan + verify`

## 先看你现在这个 repo 更像什么

结合当前仓库，我会把现状判断为“LLM policy + 短上下文”，还没有真正进入 agent system：

- `mindgames/agents/openai_agent.py`
  - 当前主路径仍是“给模型一个 prompt，要求输出 EXACTLY ONE action”
  - 这更像 action-only policy，不是显式 memory/planning agent
- `mindgames/wrappers/ObservationWrappers/llm_observation_wrapper.py`
  - 目前主要是在拼 prompt、状态快照、recent events、board state
  - 这还是“把更多文本给模型”的范式
- `tools/rollout/hanabi_gym_plugin.py`
  - 当前 `HanabiRecentTurnsContextManager` 的核心是裁剪最近若干 user turns
  - 这已经有 agent 化入口，但还没有 summary state、belief state、typed memory
- `docs/hanabi_goal_memory_v1.md`
  - 你已经开始把“goal working memory”单独拿出来做
  - 这是对的，但它更像 agentization 的第一层，不是全部

所以更准确地说，你现在不是“要不要做 agent”，而是：

- 已经在做 agentization
- 现在要决定下一步加哪一层最值钱

## 2025+ 研究里最值得吸收的几条主线

### 1. Memory 不再只是检索，而是 agent-managed memory system

2025 年之后，memory 研究的重点明显从“把过去的文本拿回来”转向“memory 是一个被管理、被演化、被分层调度的系统”。

#### A-MEM（2025）

- 链接：<https://arxiv.org/abs/2502.12110>
- 关键词：agentic memory, adaptive organization, memory evolution
- 对我最重要的启发：
  - memory 不是 append-only diary
  - memory 会改写、合并、链接、重组织
  - memory 的结构要跟 agent 的任务和上下文动态匹配

对 Hanabi 的映射：

- 你不能只存“上回合发生了什么”
- 你需要存“这个 Reveal 让哪张牌的身份分布发生了变化”
- 你还需要允许旧 belief 失效、被覆盖、被合并

也就是说，Hanabi memory 更像“可更新的 belief graph”，不是聊天记录。

#### MemOS（2025）

- 链接：<https://arxiv.org/abs/2507.03724>
- 关键词：memory operating system, multi-scale memory, lifecycle management
- 对我最重要的启发：
  - memory 要按时间尺度和表示形式分层
  - 不同 memory 的写入、读取、压缩、淘汰策略应该不同

对 Hanabi 的映射：

- working memory：当前 1-3 回合内的 active goals / urgent clues
- belief memory：对自己手牌和队友手牌意图的当前假设
- episode memory：一局结束后的成功/失败模式
- partner memory：某类搭档或某个 checkpoint 的提示风格
- procedure memory：固定可复用的“保 5 / 安全弃牌 / 给立即可打提示”等策略模板

这意味着：

- 不要只做一个总的 `memory_text`
- 应该做多类 memory，每类有自己的 budget、TTL、更新规则

#### MIRIX（2025）

- 链接：<https://arxiv.org/abs/2507.07957>
- 关键词：six memory types, multi-agent memory system, lifelong personal assistant
- 对我最重要的启发：
  - memory types 应该显式区分
  - 一个 controller 可以根据任务决定读什么、写什么、压缩什么

对 Hanabi 的映射：

- 最值得拆开的 memory types 至少有：
  - `goal memory`
  - `belief memory`
  - `partner/convention memory`
  - `episode reflection memory`
  - `public summary memory`
  - `procedure memory`

这比“一个大上下文 + 一点 RAG”更像真正可研究的 agent system。

### 2. 长程 agent 的关键不是更长上下文，而是 summary state

#### ReSum（2025）

- 链接：<https://arxiv.org/abs/2509.13313>
- 关键词：context summarization, long-horizon search, ReSum-GRPO
- 对我最重要的启发：
  - 长程 agent 在上下文不够时，不是简单截断，也不是把所有历史原样塞回去
  - 更有效的方式是周期性压缩为“reasoning state”
  - 而且这个 summary state 甚至可以跟 RL/GRPO 训练耦合

对 Hanabi 的映射非常直接：

- 你现在的 `HanabiRecentTurnsContextManager` 只是保最近 N turns
- 下一步更强的做法，不是把 N 调大
- 而是做 `HanabiSummaryStateContextManager`

它每隔 1-2 回合，把历史压成结构化摘要，例如：

- 当前最可能可打的自手槽位
- 当前最危险的误弃风险
- 最近一次 reveal 的预期意图
- 当前 goal stack
- 当前 unresolved uncertainty

这类 summary state 比纯 recent turns 更适合 Hanabi，因为：

- Hanabi 的关键状态不是语言表面，而是隐藏状态推断结果
- 你真正想保留的是“推断后的状态”，不是原文

### 3. 强 agent 会显式处理 uncertainty，而不是只做更长的推理

#### WebSailor（2025）

- 链接：<https://arxiv.org/abs/2507.02592>
- 关键词：uncertainty reduction, long-horizon web agent, high uncertainty tasks
- 对我最重要的启发：
  - 强 agent 的能力不只是“规划”
  - 还包括在高不确定性环境里，主动识别并降低关键 uncertainty
  - 训练数据和策略都可以围绕 uncertainty reduction 设计

对 Hanabi 的映射是最有价值的一点：

- Reveal 的价值，本质上就是减少关键 uncertainty
- 什么时候该打、该弃、该提示，本质上是一个 uncertainty-sensitive decision

因此 Hanabi agent 不应只做：

- “我现在最像哪个动作”

还应做：

- “当前哪个 uncertainty 最危险”
- “哪个 reveal 能最大幅度减少这类 uncertainty”
- “在 fuse 风险高时，我是否该优先 information gathering”

这会自然导向一个 `uncertainty-aware hint planner`。

### 4. 2025 年开始，真正的 agent 训练越来越强调 end-to-end RL

#### DeepResearcher（2025）

- 链接：<https://arxiv.org/abs/2504.03160>
- 关键词：end-to-end RL, real-world environment, planning, self-reflection, multi-agent architecture
- 对我最重要的启发：
  - agent 能力可以在真实交互环境里通过 RL 学出来，而不只是 prompt 设计出来
  - 规划、交叉验证、自我反思这类行为，不一定要显式手写成固定模板
  - 在合适训练信号下，它们会部分涌现

对 Hanabi 的启发：

- 你这个项目本来就在真环境里 rollout
- 所以“agent 化”不应该只停留在推理时结构变化
- 还应该考虑把 memory op、summary、verifier decision 一起纳入训练轨迹

换句话说：

- agent 不只是 inference-time architecture
- agent 也应该是 trainable workflow

#### Agent Lightning（2025）

- 链接：<https://arxiv.org/abs/2508.03680>
- 关键词：train any agent with RL, execution-training disaggregation, hierarchical credit assignment
- 对我最重要的启发：
  - 可以把 agent runtime 和 RL training 解耦
  - 复杂 agent workflow 也可以用统一 RL 接口训练
  - hierarchical credit assignment 对多步、动态分支 agent 很关键

这对 Hanabi 非常重要，因为 Hanabi 的 reward 本来就延迟而且多步归因困难：

- 一个 Reveal 的价值，要隔 1-3 回合才看出来
- 一个错误 discard 可能要很多步后才体现
- 一个错误 belief update 会连锁影响多个回合

所以如果你要认真研究“LLM 到 agent 是否提升 Hanabi”，一定要把 credit assignment 放到研究问题里，而不是只看最后 score。

## 把这些研究翻译成 Hanabi，该引入什么

下面按我建议的优先级写。

### Priority 1: Belief memory

这是我认为在 `goal memory` 之后最该加的模块。

原因：

- Hanabi 的核心困难不是“忘了目标”
- 而是“看不到自己的牌，因此需要维持关于自己手牌的假设”

如果只有 goal memory，没有 belief memory，常见问题是：

- goal 对，但目标牌身份判断错
- 提示理解错，但系统里没有地方显式记录“不确定”
- 旧提示已经因为抽牌/移位失效，但模型还沿用旧判断

建议最小 schema：

```json
{
  "belief_id": "self_slot_2",
  "entity": {"player": 0, "slot": 2},
  "candidate_cards": [
    {"color": "green", "rank": 1, "p": 0.72},
    {"color": "yellow", "rank": 2, "p": 0.18}
  ],
  "playable_prob": 0.72,
  "discard_safe_prob": 0.06,
  "evidence": [
    "turn_5: partner revealed rank 1 to slot 2",
    "turn_6: no follow-up save hint"
  ],
  "confidence": 0.72,
  "ttl": 2
}
```

关键点：

- belief 是 typed object，不是自然语言段落
- belief 要有 `confidence`
- belief 要有 `evidence`
- belief 要能因为 slot shift 或新 reveal 被改写

### Priority 2: Uncertainty-aware hint planner

这是第二个最该加的。

不是所有 reveal 都应该被建模成“给队友一个动作建议”。
很多 reveal 的真正价值是：

- 降低误打风险
- 降低误弃风险
- 让下一回合更稳
- 清除 belief 冲突

建议显式为每个候选动作计算：

- expected score delta
- fuse risk
- information gain
- future action enablement

一个简单但很有研究价值的启发式是：

- 当 `playable_prob` 高且 `fuse risk` 低时，鼓励 Play
- 当 `discard_safe_prob` 高且信息 token 紧缺时，鼓励 Discard
- 当关键槽位 uncertainty 高且错误代价高时，优先 Reveal

这会让 Hanabi 更像“risk-aware agent”，而不是“文本分类器”。

### Priority 3: Action verifier / tool layer

我强烈建议把一部分状态推断交给 deterministic tool，而不是要求 LLM 全记住。

Hanabi 里可以考虑的 tool 层包括：

- legal action checker
- slot shift tracker
- public card-count tracker
- discard criticality estimator
- immediate playability checker（对公开可见牌）
- hint truthfulness checker

注意这里的原则：

- system 管 deterministic public state
- LLM 管 latent belief / partner intent / action tradeoff

如果不做这个分层，memory 很容易退化成“帮模型记 deterministic facts”，研究价值也会降低。

### Priority 4: Episode summary / reflection

这部分也是我很看好的。

每局结束后，不只记录总分，还应该生成一个短反思对象：

```json
{
  "episode_id": "seed_42_ep_19",
  "what_worked": [
    "single-card rank clue to newest slot reliably led to immediate play"
  ],
  "what_failed": [
    "belief on self_slot_1 remained stale after discard-induced shift"
  ],
  "update_candidates": [
    "increase shift-sensitivity for self-belief updates",
    "downweight risky play when playable_prob is between 0.45 and 0.65"
  ]
}
```

reflection 的用途有两类：

- inference-time：把最近失败模式变成下一局的 caution rule
- training-time：把它变成更细粒度 credit assignment 信号

### Priority 5: Partner / convention memory

这部分很重要，但我不建议早于 belief memory 和 verifier。

为什么重要：

- Hanabi 很多 reveal 的意义依赖搭档约定
- 同一句提示，不同搭档可能含义不同
- 你已有 `partner pool` 研究方向，这个模块和它天然耦合

建议先做两层：

- intra-episode partner model
  - 本局里，这个搭档最近提示风格像什么
- cross-episode partner profile
  - 对某个固定 checkpoint / policy family 的统计记忆

一个最小 partner profile 可以长这样：

```json
{
  "partner_id": "policy_family/qwen3_8b_ckpt_420",
  "conventions": [
    {
      "pattern": "single-card clue to newest slot",
      "likely_intent": "play_now",
      "confidence": 0.64,
      "support": 17
    }
  ]
}
```

### Priority 6: Procedure / skill memory

如果前面几个基础层有了，再加这个会更有用。

它不是记某张牌，而是记“怎么做”：

- save critical 5
- set up partner immediate play
- safe discard fallback
- emergency info recovery

这类 procedure memory 能把 agent 从“每次都从头想”推进到“可复用策略模板 + 当前 belief 填槽”。

## 我不建议一开始就做的东西

### 1. 自由文本长日记 memory

问题：

- 难验证
- 难压缩
- 容易漂移
- 很容易把 deterministic facts、belief、goal、reflection 混成一团

### 2. 每回合多 agent debate

研究上很酷，但当前阶段我不建议优先做。

原因：

- latency 高
- trace 很难稳定
- credit assignment 更复杂
- 你还没有 typed state，就会先被 orchestration 噪声淹没

更好的顺序是：

- 先 single-model + typed memory + verifier
- 再考虑 planner / actor / critic 的角色拆分

### 3. 直接做跨很多局的 unconstrained lifelong memory

没有 clean belief update、summary 和 partner identity 之前，这类 memory 很容易变成噪声池。

## 一个更像 agent 的 Hanabi 最小架构

我建议把状态分成 system-owned 和 model-owned 两层。

### System-owned

- immutable public event log
- structured public state
- legal action set
- slot shift map
- public count / discard statistics
- summary compression trigger

### Model-owned

- goal memory
- belief memory
- partner/convention memory
- episodic reflection memory
- action choice

### 推荐单回合输出 schema

这比现在的 action-only 输出更适合研究：

```json
{
  "belief_ops": [
    {
      "op": "upsert_belief",
      "belief_id": "self_slot_2",
      "playable_prob": 0.72,
      "discard_safe_prob": 0.06,
      "confidence": 0.72
    }
  ],
  "goal_ops": [
    {
      "op": "set",
      "goal_id": "play_slot_2",
      "goal": "play the likely good card soon",
      "target": "self_slot2",
      "priority": "high",
      "ttl": 1
    }
  ],
  "action_candidates": [
    {
      "action": "[Play] 2",
      "score_estimate": 0.81,
      "fuse_risk": 0.28
    },
    {
      "action": "[Reveal] player 1 card 4 rank 1",
      "score_estimate": 0.59,
      "information_gain": 0.77
    }
  ],
  "action": "[Play] 2"
}
```

这里面最关键的不是字段名字，而是三件事：

- memory update 跟 action 同一回合产出
- candidate action 里保留 uncertainty / risk 信息
- 最终 action 可以被 verifier 二次检查

## 对当前分支最自然的演进路线

当前 `docs/hanabi_goal_memory_v1.md` 的方向我认为是对的，但它最好被放在更大路线里看：

### Phase 1: Goal memory V1

对应当前分支：

- 小型 typed goal memory
- 单次 completion 里同时产出 `memory_ops + action`
- 不引入 belief memory

这是一个很好的“agent 化最小闭环”。

### Phase 2: Belief memory V1

建议新加：

- `belief_memory_enabled`
- `belief_memory_max_items`
- `belief_render_topk`
- `belief_memory_default_ttl`

研究问题：

- goal memory 之外，加 belief memory 是否显著提分
- belief memory 是否降低 fuse loss 和 invalid risky play

### Phase 3: Summary state + verifier

建议新加：

- `summary_state_enabled`
- `summary_every_n_turns`
- `action_verifier_enabled`
- `uncertainty_policy=heuristic|llm|hybrid`

研究问题：

- summary state 能否比单纯 recent-turn truncation 更稳
- verifier 是否能在不明显降低速度的前提下降低高风险错误

### Phase 4: Partner / convention memory

建议新加：

- `partner_memory_enabled`
- `partner_memory_scope=episode|cross_episode`
- `partner_profile_key=model|checkpoint|policy_family`

研究问题：

- partner memory 是否提升 cross-play
- 是否会损伤对未知搭档的泛化

### Phase 5: RL-trained agent workflow

这里才是“从 LLM agent 原型到可训练 agent system”的关键一步。

建议把下面这些都当成可训练轨迹的一部分：

- memory ops
- summary ops
- candidate scoring
- verifier accept/reject
- final action

## 对 repo 的具体落点

如果只做文档规划，我建议把后续改动主要落在这几个位置：

### `mindgames/agents/openai_agent.py`

当前更像 action-only policy。
后续可改成：

- 支持结构化 JSON 输出
- 支持 `belief_ops + goal_ops + action`
- 支持 candidate actions 和 verifier 反馈

### `mindgames/wrappers/ObservationWrappers/llm_observation_wrapper.py`

当前主要是把 prompt、state snapshot、recent events、board state 拼接起来。
后续可改成分段渲染：

- structured public state
- rendered belief summary
- rendered goal summary
- unresolved uncertainties
- recent events

### `tools/rollout/hanabi_gym_plugin.py`

这里是最自然的 agent 化入口。

当前已有：

- `HanabiRecentTurnsContextManager`

后续可以新增：

- `HanabiSummaryStateContextManager`
- `HanabiBeliefMemoryContextManager`
- `HanabiAgentStateContextManager`

也就是说，不只是裁 recent turns，而是维护可训练的 summary / memory state。

### `mindgames/envs/Hanabi/env.py`

这里适合继续保持“环境权威源”的角色。

建议它继续 system-owned 地提供：

- public state snapshot
- legal action constraints
- hint validity / truthfulness checks
- step reward / step info

不要把这些 deterministic 逻辑挪给模型去记。

## 实验设计建议

如果你要把“LLM -> agent”做成清楚的研究问题，我建议至少做下面这组 ablation。

### 核心 ablation

1. baseline action-only LLM
2. `+ goal memory`
3. `+ belief memory`
4. `+ belief memory + goal memory`
5. `+ verifier`
6. `+ summary state`
7. `+ partner memory`
8. `+ RL-trained workflow`

### 指标不要只看平均分

至少还要看：

- average score
- perfect game rate
- fuse-out rate
- invalid move rate
- risky play rate
- bad discard rate
- reveal-to-useful-action conversion rate
- cross-play mean
- cross-play bottom decile

如果你做了 belief memory，我强烈建议再加两类指标：

- calibration
  - 例如 `playable_prob` 的 Brier score
- memory quality
  - stale belief rate
  - contradicted belief rate
  - summary omission rate

## 我对这个研究问题的判断

如果问题是：

“LLM 变成 agent，能不能提升 Hanabi 表现？”

我的判断是：

- 大概率能
- 但真正带来提升的，不会是单一的 generic memory
- 最关键的是把 Hanabi 中真正困难的隐变量显式化

更具体地说，我预期最可能带来增益的是：

1. `belief memory`
2. `uncertainty-aware hint planner`
3. `action verifier`
4. `summary state`

而不是：

1. 更长 prompt
2. 更自由的 diary memory
3. 每回合多 agent 辩论

## 推荐阅读（2025+）

- A-MEM: Agentic Memory for LLM Agents
  - <https://arxiv.org/abs/2502.12110>
  - 重点看 memory 如何动态组织、链接和演化
- DeepResearcher: Scaling Deep Research via Reinforcement Learning in Real-world Environments
  - <https://arxiv.org/abs/2504.03160>
  - 重点看 end-to-end RL + self-reflection + specialized multi-agent architecture
- WebSailor: Navigating Super-human Reasoning for Web Agent
  - <https://arxiv.org/abs/2507.02592>
  - 重点看 uncertainty reduction 的 framing
- MemOS: A Memory OS for AI System
  - <https://arxiv.org/abs/2507.03724>
  - 重点看 multi-scale memory 与 lifecycle management
- MIRIX: Multi-Agent Memory System for LLM-Based Agents
  - <https://arxiv.org/abs/2507.07957>
  - 重点看 memory type 拆分和 controller 视角
- Agent Lightning: Train ANY AI Agents with Reinforcement Learning
  - <https://arxiv.org/abs/2508.03680>
  - 重点看 execution-training disaggregation 和 hierarchical credit assignment
- ReSum: Unlocking Long-Horizon Search Intelligence via Context Summarization
  - <https://arxiv.org/abs/2509.13313>
  - 重点看 summary state 与 GRPO 的结合方式

## 和当前 `goal_memory_v1` 的关系

`docs/hanabi_goal_memory_v1.md` 里的方向我建议继续保留，因为它已经抓住了一个对的切入口：

- goal memory 要 typed
- memory op 和 action 最好同回合输出
- system 负责 validation 和 hygiene

但它应该被理解为：

- 不是“完整 agent 设计”
- 而是“agentization 的第一层”

如果下一步只能加一层，我建议加：

- `belief memory`

如果下一步能加两层，我建议加：

- `belief memory`
- `action verifier`

如果你要把这件事做成一条更完整研究线，那就是：

- goal -> belief -> uncertainty -> verifier -> summary -> partner memory -> RL workflow

