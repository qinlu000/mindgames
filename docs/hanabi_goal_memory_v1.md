# Hanabi Goal Memory V1

Date: 2026-03-17

This branch focuses on single-episode goal memory for the Hanabi agent.
It does not attempt cross-game memory, partner-profile persistence, or
long-term belief storage yet.

## Branch

- Branch name: `exp/hanabi-goal-memory-v1`

## Why use a dedicated branch

- The change will touch the main Hanabi rollout path, not just a small prompt tweak.
- We will likely need A/B comparisons across:
  - no goal memory
  - heuristic goal memory
  - LLM goal writer
- A separate branch keeps the main training and evaluation flow clean while
  the experiment is still unstable.
- The actual experiment comparison should rely on feature flags inside the
  same branch, not on comparing different branches with different code.

## Ground rules

- Keep default behavior unchanged until the new path is explicitly enabled.
- Start with single-episode goal memory only.
- Treat goal memory as an experimental layer around the existing rollout
  loop, not as a rewrite of the Hanabi environment.
- Prefer structured goal state over free-form diary text.

## Initial feature flags

- `goal_memory_enabled=false`
- `goal_writer_mode=heuristic|llm`
- `goal_render_topk=2`

These flags should be enough for the first round of ablations:

1. Baseline: `goal_memory_enabled=false`
2. Heuristic goals: `goal_memory_enabled=true`, `goal_writer_mode=heuristic`
3. LLM goals: `goal_memory_enabled=true`, `goal_writer_mode=llm`

## Expected integration points

- `tools/rollout/hanabi_gym_plugin.py`
  - before inference: render goal memory into the current prompt context
  - after env step: update goal memory from the latest public events
  - rollout info: log the active goals and whether a goal update was triggered
- rollout launchers
  - pass the goal-memory feature flags through env or ctx config
- training/evaluation
  - compare variants by flags, not by maintaining separate logic branches

## First-pass scope

- Maintain a small active-goal set for the current episode
- Render only the top goals into the prompt
- Keep the output action-only; do not mix action generation with free-form
  memory writing in the same completion
- Record enough rollout metadata to measure whether the active goal matched
  the chosen action

## Evaluation notes

Primary comparisons should include:

- average cooperative score
- invalid action / normalized action failure rate
- fuse-out vs. deck-out ending pattern
- optional goal-hit rate:
  - whether the chosen action matches the top rendered goal

## Non-goals for V1

- cross-game memory
- partner-style memory
- belief-library implementation
- convention learning beyond the active-goal layer
