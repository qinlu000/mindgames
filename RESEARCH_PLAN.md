# Mindgames Research Plan (Truth&Deception + Hanabi)

## 0) 目标与边界
- **目标**：提升模型在 text-based mindgames 中的**可测游戏表现**（胜率/得分/稳定性）。
- **当前载体环境**：`TruthAndDeception`（对抗、欺骗/识别） + `Hanabi`（合作、信息受限沟通）。
- **优先级**：游戏表现 > ToM 解释；ToM 仅作为“这些游戏为什么难”的背景。
- **原则**：低噪声、可复现、可消融、先跑通闭环再上复杂 RL。

## 1) 一句话方法论
固定评测协议（env/对手/采样/seed）→ 跑 baseline → 采集轨迹 → SFT(LoRA) 做“合法+稳定策略” → 再用 self-play/对手池/偏好或 RL 提升 → 全程做回归评测与消融定位增益来源。

## 2) 固定评测协议（必须先定死）
### 2.1 环境与版本
- 统一用 `*-train` 变体（更适合训练的 wrappers）。
- Truth&Deception：`TruthAndDeception-v0-train`（必要时加 `TruthAndDeception-v0-long-train`）。
- Hanabi：`Hanabi-v0-train`（默认 2 人局，后续可扩到 3 人）。

### 2.2 固定项（对比实验必须一致）
- `env_id`、`num_players`
- opponent/teammate（固定对手或固定队友；或 self-play 也要固定版本）
- 采样：`temperature/top_p/max_tokens`（以及 stop tokens/截断策略）
- `episodes` 与 `seed 列表`
- 代码版本：`git_sha`（本 repo）、环境版本（本子项目 `mindgames` 的 git_sha）
- prompt 版本：`prompt_hash`（如使用 prompt-tooling）

### 2.3 指标（metric）
**通用：**
- `invalid_rate`：无效动作率（规则/格式错误导致的失败，必须单独监控）
- `avg_turn_count`

**Truth&Deception（对抗）：**
- `win_rate`（核心）
- **按角色拆分**：模型当 `Deceiver`（P0）/ 当 `Guesser`（P1）的 win_rate（两种能力不同）
- 终局格式成功率：最后猜测是否按 `[Fact 1]`/`[Fact 2]`（否则会被判 invalid）

**Hanabi（合作）：**
- `avg_score`（0–25，核心）
- `perfect_score_rate`（25 分占比）
- 失败模式统计：从 `reason` 聚合（例如 fuse 归零/牌堆耗尽）

> 实操建议：每个 setting 至少 50–200 局；先 10 局 smoke test，再扩大；对比用同一 seed 列表。

## 3) 代码与数据落盘约定（最小可复现链）
建议你长期固定以下结构（可逐步完善）：
```
mindgames/
  data/
    rollouts/      # 原始对局 JSONL（可复放）
    sft/           # SFT 数据 JSONL（messages 格式）
    eval/          # 评测 rollouts + 汇总
  runs/            # 每次实验的 config + 产物索引（可配 W&B）
  prompts/         # prompt registry（可选，配 prompt-tooling）
```

本子项目已提供的工具：
- 采集 rollouts：`mindgames/tools/run_rollouts.py`
- 汇总评测：`mindgames/tools/summarize_rollouts.py`
- 转 SFT 数据：`mindgames/tools/rollouts_to_sft_jsonl.py`

## 4) 阶段性路线图（从低风险到高收益）
### Phase A：Baseline（1–2 天）
目标：在固定协议下跑通“可重复评测”。
1) 选定 baseline 模型（训练前 checkpoint）。
2) 跑评测并落盘：
   - 生成 `data/eval/<env>/<model>/before.rollouts.jsonl`
   - 用 `summarize_rollouts.py` 产出汇总 JSON（或写入 W&B）
3) 建立回归门禁：每次改 prompt/代码/模型都跑一个小评测集（比如每个 env 10 局）。

### Phase B：SFT（LoRA）把“合法+稳定”先做出来（2–7 天）
目标：显著降低 `invalid_rate`，并在固定对手/队友下提升基础指标。
1) 轨迹采集：
   - 用强老师（更强模型）或你的 baseline 自采样，生成 `data/rollouts/*.jsonl`
2) 转 SFT 数据：
   - `rollouts_to_sft_jsonl.py --skip-invalid`
3) 训练：
   - 先只做 SFT，确保模型稳定输出合法动作/格式（Truth&Deception 的 `[Fact 1]/[Fact 2]` 等）
4) 评测对比：
   - before/after 在同一评测协议下对比（看 win_rate/avg_score/invalid_rate）

### Phase C：提升策略（自博弈/对手池/偏好/RL）（1–3 周）
目标：在“规则遵守”稳定后，提升策略质量与泛化。
优先顺序建议：
1) **对手池/队友池**：用多个固定快照避免过拟合单一 opponent/teammate。
2) **self-play 数据闭环**：最新模型与池中对手对弈→筛选失败/关键局→再训练。
3) **偏好学习/GRPO**：当你有“好/坏行为”的判别标准或胜负回报可用时再上。

## 4.5) 借鉴 papers 的可落地要点（SPIRAL / MARSHAL / ToM_Broken）
### (A) SPIRAL：把“游戏”当作可无限生成的在线 RL 训练场
- **Turn-level MDP**：把“一个回合的整段回复”当作 action（而不是 token-level），训练目标与评测都围绕 turn 的胜负/得分与合法性。
- **自博弈作为课程学习**：固定对手容易被 exploit；self-play 的优势在于对手随训练同步变强，天然形成 curriculum（你需要做的是“对手池快照 + 采样策略”，而不是只打最新自我）。
- **多游戏=互补技能**：SPIRAL 强调不同游戏会诱导不同能力；对应到本项目就是：`Truth&Deception` 偏“交流/欺骗/对抗”，`Hanabi` 偏“协作/意图理解/信息受限规划”。后续可以用 multi-task 或交替训练来减少过拟合。

### (B) MARSHAL：稳定训练靠“辅助约束”，并用“简化→复杂”做 OOD 泛化
- **格式正则化/强惩罚**：论文里用小正奖励鼓励格式正确、用大负奖励终止来处理无效输出；对我们来说等价于把 `invalid_rate` 当成一等公民，并在训练目标里显式压它（不然提升常常被格式噪声吞掉）。
- **长度惩罚（verbosity）**：避免模型用冗长解释污染动作；对 `Hanabi` 这类多轮协作尤其重要（冗长会增加噪声与格式错误概率）。
- **训练简化版→评测复杂版**：MARSHAL 用 Mini Hanabi 训练、Simple Hanabi 评测来验证泛化；对应到我们可以：
  - 先做 **Hanabi 简化变体**（更少颜色/点数/手牌/ token）训练到稳定策略；
  - 再评测到更复杂设定（你当前的 `Hanabi-v0-train`）。
  这样比直接在复杂 Hanabi 上从零训练更稳定，也更容易做 ablation（“泛化是否真的发生”）。
- **回报尺度归一**：对抗游戏多是 ±1，Hanabi 是 0–25（或更复杂）；多任务训练时要把 reward scale 统一，否则优化会偏向某一游戏。

### (C) ToM_Broken：不要用“表面 ToM 指标”替代“功能性表现”，并控制知识泄漏
- **指标警告**：论文强调“看起来像 ToM 的回答”不等于“在交互中做对了”；对我们来说就是：不要用中间推理/解释质量当主指标，主指标仍应是 win/score/invalid。
- **数据构造的防泄漏思路**：用“虚构情境 + 私有证据”的方式能降低模型靠预训练常识/记忆作弊的概率。落到 `Truth&Deception` 上：
  - 维护 train/test 两套 fact pool（严格不重叠）；
  - 更进一步可用程序生成“自洽但虚构”的事实对（并把正确答案作为私有信息只给 Deceiver）。

### (D) 从 papers 学到的“分析工作流”
- **失败模式分层**：把失败拆成：格式/规则、策略错误、信息利用错误、沟通失败；分别统计与定向改进。
- **checkpoint 演化分析**：像 SPIRAL 那样定期抽样若干局，对比不同 checkpoint 的行为变化（可以先做“最差/最好局”的对比）。

## 5) 关键消融（最小集合）
每次只改一项，跑同一 eval matrix：
- Prompt：system prompt 是否含“动作格式示例/硬约束/简短策略提示”
- Data：只用合法局 vs 混入少量非法局（看鲁棒性）
- Opponent：固定对手 vs opponent pool
- Sampling：temperature 0.0/0.2/0.7（看稳定性）

## 6) 结果“可信”的最低标准
你可以把以下作为内部 gate：
- Truth&Deception：`invalid_rate` 明显下降，且角色拆分 win_rate 至少一侧提升并可重复
- Hanabi：`avg_score` 提升且跨 seed 稳定；`perfect_score_rate` 有增长更好
- 所有结论都能被复跑：同 config + 同 seeds + 同版本依赖能重现

## 7) 与你已有工具的对接建议（可选但高收益）
- **prompt-tooling**：每次实验记录 `prompt_hash`，避免“提示词变了但忘记记录”的不可复现。
- **wandb-tracker**：Runs+Artifacts 统一存：config / rollouts / eval summary / checkpoints。

## 8) 下一步（你现在就能做的 3 件事）
1) 定一个评测 matrix（每 env 50 局、固定 seeds、固定对手/队友）。
2) 跑 baseline 并存档（before.rollouts + summary）。
3) 用老师模型采集 200–1k 局轨迹，做第一版 SFT（目标先把 invalid_rate 压到很低）。
