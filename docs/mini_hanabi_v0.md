# MiniHanabi-v0

`MiniHanabi-v0` is a short-context, coordination-heavy Hanabi variant intended as a social-intelligence testbed. The design goal is to preserve hidden information, audience design, and memory of public actions without requiring the long horizons and large belief states of full Hanabi.

## Design goals

- Keep the full per-turn observation small enough for routine LLM rollouts.
- Preserve nontrivial cooperative inference: hints are public, partial, and action-constrained.
- Avoid making the game too easy by exposing solved belief sets or direct playability labels.
- Remove bookkeeping that adds context length without adding much strategic value, such as shifting hand indices and deck-exhaustion final rounds.

## Core rules

- Players: exactly `2`
- Colors: `Red`, `Blue`, `Green`
- Ranks: `1`, `2`, `3`
- Per-color deck: `[1, 1, 2, 3]`
- Total deck size: `12`
- Hand size: `2`
- Hand slots: fixed `A`, `B`
- Max info tokens: `2`
- Start info tokens: `2`
- Max fuse tokens: `2`
- Start fuse tokens: `2`
- Turn cap: `12` consumed turns
- Perfect score: `9`

## Fixed-slot hand model

Each player has two persistent hand slots, `A` and `B`.

- Slots do not shift after a play or discard.
- If a card leaves slot `A`, the replacement card fills slot `A`.
- If the deck is empty, the slot becomes empty.
- Hints and actions always refer to slots, never to moving indices.

This keeps the interface stable across turns and reduces unnecessary bookkeeping, while still requiring players to remember what each slot likely contains.

## Actions

Players must output exactly one action.

Valid action families:

- `[Play A]`, `[Play B]`
- `[Discard A]`, `[Discard B]`
- `[Hint Color Red]`, `[Hint Color Blue]`, `[Hint Color Green]`
- `[Hint Rank 1]`, `[Hint Rank 2]`, `[Hint Rank 3]`

Robust parsing should also accept equivalent slot aliases `0/1` for `A/B`, and wrapper-normalized formats such as:

- `[Play] A`
- `[Discard] B`
- `[Hint] Color Red`
- `[Hint] Rank 2`

No free-form chat is allowed.

## Hint rules

Hints follow standard-style Hanabi semantics within the 2-player setting.

- A hint always targets the partner.
- A hint must be truthful.
- A hint must touch at least one occupied partner slot.
- Color hints touch all occupied partner slots of that color.
- Rank hints touch all occupied partner slots of that rank.
- Giving a hint spends `1` info token.
- If info tokens are `0`, hinting is invalid.

Knowledge updates:

- Touched slots receive positive information for the hinted attribute.
  - Example: after a color hint touching slot `A`, slot `A` now knows its color.
- Untouched occupied slots receive negative information for that attribute.
  - Example: if a color hint touches only slot `A`, then occupied slot `B` learns it is not that color.
- Empty slots are ignored by hints.

## Play and discard rules

For each color, fireworks must be built in ascending order `1 -> 2 -> 3`.

### Play

- A play succeeds if the card is the next required rank for that color.
- On success, that firework advances by `1`.
- On failure, the card is discarded and the team loses `1` fuse token.
- If a successful play completes a color by playing rank `3`, regain `1` info token if below the cap.

### Discard

- The discarded card is moved to the discard pile.
- Discarding regains `1` info token if below the cap.

After any play or discard:

- Draw one replacement card into the same slot if the deck is non-empty.
- Otherwise that slot becomes empty.
- Any slot-specific private knowledge for the replaced card is reset.

## Episode end conditions

The game ends immediately when any of the following occurs:

- All three fireworks reach rank `3` (`score = 9`)
- Fuse tokens reach `0`
- The `12`-turn cap is reached

Unlike full Hanabi, there is no extra final round after deck exhaustion.

## Reward

This is a cooperative game. Both players receive the same terminal reward:

- `reward[player] = final_score`

where `final_score` is the sum of all firework heights, in `[0, 9]`.

## Observation design

The environment should expose compact public state plus raw hint constraints, not solved beliefs.

Each player observation should include:

- Current fireworks by color
- Info tokens, fuse tokens, deck size, turn count
- Public discard pile
- Partner's visible cards
- The acting player's own slot knowledge, expressed as constraints only
  - known color / unknown color
  - known rank / unknown rank
  - excluded colors
  - excluded ranks
  - optional recency marker such as last touched turn

The observation should not expose:

- exact posterior `possible_cards={...}`
- direct `playable` or `safe_discard` labels
- hidden-card ground truth for the acting player

## Why this version is harder than the easiest toy variants

Compared with ultra-small cooperative hidden-information games, this version is intentionally a little harder:

- `3` colors instead of `2`
- standard all-matching hints instead of single-card hints
- public negative information from untouched slots
- no slot shifting, so long-lived slot identity matters
- observations expose only constraints, so models must do their own belief tracking

The result is still much shorter-context than full Hanabi, but it should produce clearer separation between agents that truly track partner knowledge and agents that only react locally.
