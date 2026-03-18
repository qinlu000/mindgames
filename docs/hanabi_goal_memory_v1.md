# Hanabi Goal Memory V1

Date: 2026-03-17

Related note:

- For the broader 2025+ agent literature review and the next-step roadmap
  beyond goal memory, see `docs/hanabi_agent_research_2025plus.md`.

This branch focuses on single-episode goal memory for the Hanabi agent.
It does not attempt cross-game memory, partner-profile persistence, or
long-term belief storage yet.

After the initial branch setup, the preferred V1 direction is now:

- let the LLM control goal working memory directly
- let the same model decide memory updates and the final action
- constrain that control through a typed memory interface instead of a
  free-form diary

## Branch

- Branch name: `exp/hanabi-goal-memory-v1`

## Why use a dedicated branch

- The change will touch the main Hanabi rollout path, not just a small prompt
  tweak.
- We will likely need A/B comparisons across:
  - no goal memory
  - heuristic goal memory
  - LLM goal writer
  - fully LLM-controlled goal working memory
- A separate branch keeps the main training and evaluation flow clean while
  the experiment is still unstable.
- The actual experiment comparison should rely on feature flags inside the
  same branch, not on comparing different branches with different code.

## V1 hypothesis

The main experiment hypothesis is:

- Hanabi benefits from a short-lived goal working memory that tracks only the
  most relevant active goals for the current episode.
- The LLM can manage that goal memory itself if memory edits are forced through
  a structured operation interface.
- The system should not decide which goals to keep; it should only enforce
  schema validation, budget limits, TTL expiry, and immutable event logging.

In short:

- the LLM owns goal selection
- the LLM owns goal insertion/deletion/reprioritization
- the system owns validation and memory hygiene

## Design principles

- Keep V1 goal-only.
  - Do not mix belief memory into the first experiment.
  - Goal memory is the planning layer, not the factual card-state ledger.
- Use typed memory ops.
  - Do not let the model rewrite a long natural-language diary each turn.
  - Memory should be edited through a small set of explicit operations.
- Keep memory short.
  - Only store a few active goals.
  - Render only the top goals back into the next prompt.
- Keep raw events immutable.
  - Public observations, prior actions, and slot-shift consequences must stay
    in a system-owned log.
  - The LLM may reason over that log but must not edit it.
- Preserve a clean baseline.
  - Default behavior stays unchanged until goal memory is enabled.

## Experiment matrix

The branch should support at least these variants:

1. Baseline
   - `goal_memory_enabled=false`
2. Heuristic goals
   - `goal_memory_enabled=true`
   - `goal_writer_mode=heuristic`
   - `goal_memory_control_mode=advisory`
3. LLM goal writer, separate memory pass
   - `goal_memory_enabled=true`
   - `goal_writer_mode=llm`
   - `goal_memory_control_mode=advisory`
4. Fully LLM-controlled goal working memory
   - `goal_memory_enabled=true`
   - `goal_writer_mode=llm`
   - `goal_memory_control_mode=full`

Variant 4 is the primary research direction for this branch.
Variants 1-3 remain important as ablations and debugging baselines.

## Initial feature flags

- `goal_memory_enabled=false`
- `goal_writer_mode=heuristic|llm`
- `goal_memory_control_mode=advisory|full`
- `goal_memory_output_format=json_ops`
- `goal_render_topk=2`
- `goal_memory_max_goals=3`
- `goal_memory_max_ops_per_turn=2`
- `goal_memory_default_ttl=2`

Notes:

- `advisory` means goal memory is generated and rendered, but the system still
  treats it as a planning aid.
- `full` means the LLM emits both memory ops and the final action in one typed
  output object.
- `json_ops` means the model edits memory through explicit operations rather
  than free-form text replacement.

## Preferred V1 control model

The preferred V1 control model is:

- one base model
- one completion per turn
- one structured output containing both:
  - `memory_ops`
  - `action`

This preserves the idea that the LLM has full control over working memory while
avoiding prompt-format drift between a planner model and an actor model.

The recommended output contract is:

```json
{
  "memory_ops": [
    {
      "op": "upsert_goal",
      "goal_id": "play_slot_2",
      "goal_type": "play_slot",
      "target": {"player": 0, "slot": 2},
      "priority": 0.88,
      "ttl": 1,
      "reason": "single-card reveal last turn; likely immediately playable"
    },
    {
      "op": "delete_goal",
      "goal_id": "recover_info"
    }
  ],
  "action": "[Play] 2"
}
```

This is preferred over a free-form response because it is easier to validate,
log, replay, and compare across ablations.

## Goal memory state

Committed working memory should remain small and typed. A reasonable first-pass
state looks like this:

```json
{
  "turn_id": 7,
  "goals": [
    {
      "goal_id": "play_slot_2",
      "goal_type": "play_slot",
      "target": {"player": 0, "slot": 2},
      "priority": 0.88,
      "ttl": 1,
      "source": "llm",
      "reason": "single-card reveal last turn; likely immediately playable",
      "status": "active"
    },
    {
      "goal_id": "safe_discard",
      "goal_type": "safe_discard_fallback",
      "target": {"player": 0, "slot": 4},
      "priority": 0.31,
      "ttl": 2,
      "source": "llm",
      "reason": "fallback if play confidence drops",
      "status": "active"
    }
  ]
}
```

Recommended goal types for V1:

- `play_slot`
- `save_partner_card`
- `set_up_partner_play`
- `recover_info_token`
- `safe_discard_fallback`

Do not add belief-like objects in V1.

## Allowed memory operations

The LLM should edit memory only through a small operation set:

- `upsert_goal`
- `delete_goal`
- `reprioritize_goal`
- `compress_goals`

Suggested meanings:

- `upsert_goal`
  - add a new goal or replace an existing goal with the same `goal_id`
- `delete_goal`
  - remove a stale or conflicting goal
- `reprioritize_goal`
  - change the ranking of an existing goal without changing its content
- `compress_goals`
  - optional future op for merging similar goals under tight budget

For V1, `upsert_goal` and `delete_goal` are enough to start.

## System-owned invariants

Even in full-control mode, the system should still own these invariants:

- schema validation
  - reject malformed JSON
  - reject unknown ops
  - reject invalid targets
- budget limits
  - cap active goals
  - cap ops per turn
  - cap reason length
- TTL expiry
  - decrement goal TTL each turn
  - remove goals whose TTL reaches zero
- slot rebasing
  - if a play/discard shifts hand indices, update slot-indexed goals before the
    next turn
- immutable public event log
  - keep a system-owned record of observations and actions
  - never let the model edit that log

This keeps the memory channel auditable without taking away goal ownership from
the LLM.

## Turn lifecycle

The intended per-turn control flow is:

1. Load committed goal memory for the trajectory.
2. Apply system maintenance:
   - expire TTLs
   - rebase slot-indexed goals after hand shifts
3. Build the next prompt from:
   - current Hanabi observation
   - immutable recent public events
   - rendered top-k committed goals
4. Call the model once.
5. Parse the structured output:
   - `memory_ops`
   - `action`
6. Validate and apply memory ops.
7. Execute the action in the env.
8. Log telemetry:
   - emitted memory ops
   - final committed goals
   - whether the action matched the top goal

## Prompting contract

The prompt for full-control mode should make these constraints explicit:

- You may update only goal working memory.
- You may not invent or edit public events.
- Keep memory short and useful for the next few turns only.
- Delete stale goals rather than accumulating them.
- Output valid JSON only.
- The action field must contain exactly one legal Hanabi action string.

The prompt should also tell the model that goal memory is not a diary. It is a
short-lived action-guidance layer for the current episode.

## Rendering strategy

Only committed goals should be rendered back into the next turn prompt.
Render no more than `goal_render_topk` goals.

Example rendering:

```text
Goal memory:
- Primary goal: play slot 2 now.
- Backup goal: if slot 2 is unsafe, recover an info token safely.
```

Do not render the full op history back into context.
Do not render stale or low-priority goals.

## Integration points

Expected code touch points:

- `tools/rollout/hanabi_gym_plugin.py`
  - before inference: inject rendered goal memory
  - after model output: parse and validate `memory_ops`
  - after env step: update TTL, rebase indices, and log telemetry
- rollout launchers
  - pass the goal-memory flags through env or ctx config
- training/evaluation
  - compare variants by flags, not by maintaining separate logic branches

## Telemetry and evaluation

Primary comparisons should include:

- average cooperative score
- invalid action / normalized action failure rate
- fuse-out vs. deck-out ending pattern
- `goal_hit_rate`
  - whether the final action matches the top committed goal
- `goal_churn`
  - how often the active goal set is rewritten
- `stale_goal_rate`
  - how often rendered goals are immediately invalidated
- `avg_active_goals`
  - average number of committed active goals per turn

## Expected failure modes

Watch for these failure patterns:

- self-confirming hallucination
  - the model writes a bad goal, then keeps acting as if that goal is evidence
- slot drift
  - goals point to stale hand indices after play/discard shifts
- goal churn
  - the model rewrites all goals every turn and loses continuity
- memory bloat
  - too many goals or too much reason text consume the context budget
- action-format pollution
  - invalid JSON or invalid Hanabi action strings

These are part of the reason V1 should keep memory typed, budgeted, and goal-
only.

## Non-goals for V1

- cross-game memory
- partner-style memory
- belief-library implementation
- convention learning beyond the active-goal layer
- unconstrained natural-language diary memory
