# Hanabi Goal Memory V1

Date: 2026-03-18

Related note:

- For the broader 2025+ agent literature review and roadmap beyond goal memory,
  see `docs/hanabi_agent_research_2025plus.md`.

This branch focuses on a minimal cross-turn goal memory for the Hanabi agent.
It does not attempt cross-game memory, long-form reflection, belief storage, or
partner-profile persistence.

The current direction is intentionally strict:

- keep only the smallest reusable goal state
- let the same model emit both `goal_ops` and `action`
- keep task-specific cleanup logic in the system, not in the goal schema

## Branch

- Branch name: `exp/hanabi-goal-memory-v1`

## V1 hypothesis

The main experiment hypothesis is:

- Hanabi benefits from a tiny committed goal list that survives across turns.
- The useful part of goal memory is commitment persistence, not storing detailed
  reasoning traces.
- A minimal goal schema should generalize better to other tasks than a richer
  planner-style object.

In short:

- the LLM owns which goals to keep
- the LLM owns setting and removing goals
- the system owns validation, TTL expiry, slot rebasing, and event logging

## Task adapter seam

To make this portable beyond Hanabi, the wrapper now has an explicit task
adapter seam:

- `GoalMemoryTaskAdapter`
  - minimal interface for prompt wording, target normalization, player-id
    inference, and post-action maintenance
- `GenericGoalMemoryTaskAdapter`
  - no domain-specific action semantics; useful as a starting point for other
    tasks
- `HanabiGoalMemoryTaskAdapter`
  - current Hanabi-specific implementation for action matching and self-slot
    rebasing

The wrapper can be constructed as:

```python
wrapper = GoalMemoryAgentWrapper(agent, adapter=MyTaskAdapter())
```

So the core goal schema stays task-light, while target parsing and action-result
logic live in a replaceable adapter.

## Design principles

- Keep goal memory minimal.
  - Do not store confidence, reasons, belief refs, or condition lists in V1.
  - The goal list should represent commitments, not explanations.
- Keep operations minimal.
  - Use only `set` and `remove`.
  - Avoid a growing planner-op vocabulary.
- Keep prompt burden low.
  - Render only a few active goals.
  - Show them as carry-over commitments from previous turns.
- Keep Hanabi logic outside the schema.
  - Slot rebasing and action matching belong to the system-side adapter.
  - Do not encode Hanabi-specific semantics into the generic goal object.

## Minimal goal schema

The committed goal object is now:

```json
{
  "goal_id": "g1",
  "goal": "save partner critical card",
  "target": "player1_slot4",
  "priority": "high",
  "ttl": 2,
  "status": "active",
  "created_turn": 12,
  "last_updated_turn": 12
}
```

Fields owned by the model:

- `goal_id`
- `goal`
- `target`
- `priority`
- `ttl`

Fields owned by the system:

- `status`
- `created_turn`
- `last_updated_turn`
- `task_adapter` in the snapshot/logging layer

Priority is intentionally coarse:

- `high`
- `medium`
- `low`

## Turn output contract

The model returns one JSON object containing both memory updates and the final
Hanabi action:

```json
{
  "selected_goal_id": "g1",
  "goal_ops": [
    {
      "op": "set",
      "goal_id": "g1",
      "goal": "save partner critical card",
      "target": "player1_slot4",
      "priority": "high",
      "ttl": 2
    },
    {
      "op": "remove",
      "goal_id": "g2"
    }
  ],
  "action": "[Reveal] player 1 card 4 rank 5"
}
```

Semantics:

- `set`
  - create a new goal or overwrite an existing goal with the same `goal_id`
- `remove`
  - remove a stale or conflicting goal
- `selected_goal_id`
  - optional pointer to the goal the chosen action is mainly following

## Prompt rendering

The prompt should present active goals as prior-turn commitments owned by the
agent itself, for example:

```text
These are goals you set in previous turns and are still active.
Treat them as your current working commitments unless you explicitly update or remove them.

Active goals:
- g1 | save partner critical card | target=player1_slot4 | priority=high | ttl_left=2
- g2 | safe discard fallback | target=self_slot4 | priority=low | ttl_left=1
```

The key framing is:

- these are goals the agent set previously
- the system carried them forward after validation/state updates
- they persist unless the agent explicitly changes them through `goal_ops`

## System-owned invariants

Even in full-control mode, the system should still own these invariants:

- schema validation
  - reject malformed JSON
  - reject unknown ops
- budget limits
  - cap active goals
  - cap ops per turn
- TTL expiry
  - expire goals that outlive their turn budget
- Hanabi slot maintenance
  - if a play/discard shifts self-hand indices, rebase remaining self-slot
    targets
  - if a self-slot target is consumed, invalidate it unless it was the goal just
    completed
- immutable logging
  - keep goal events separate from model-edited memory

## Why this is more reusable

This version is intentionally easier to port beyond Hanabi because it avoids
binding the schema to one domain.

What is not in the core goal object anymore:

- `goal_type`
- `confidence`
- `reason`
- `belief_refs`
- `preconditions`
- `success_conditions`
- `abort_conditions`

Those fields make the object look more precise, but they also make it more task-
specific and more expensive to maintain. For generalization, the main value of
goal memory is persistent intent, not a serialized chain of thought.

## Current implementation status

The current branch implements this minimal design in `mindgames/agents/goal_memory.py`.
For the Hanabi rollout path, `tools/rollout/run_rollouts.py` now wires it
through `HanabiGoalMemoryTaskAdapter()` behind feature flags:

- `--goal-memory-enabled`
- `--goal-memory-max-active`
- `--goal-memory-render-topk`
- `--goal-memory-default-ttl`
- `--goal-memory-max-ops-per-turn`

Default Hanabi behavior remains unchanged unless goal memory is enabled.
