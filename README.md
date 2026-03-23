# mindgames

This repository contains only the three game environments:
- `MiniHanabi`
- `ColonelBlotto`
- `Negotiation`

## Install

```bash
uv sync
```

If you also want the model-backed agent helpers, install the optional extra:

```bash
uv sync --extra agents
```

## Quick Start

```python
import mindgames as mg

env = mg.make("MiniHanabi-v0")
env.reset(num_players=2, seed=0)

player_id, observation = env.get_observation()
print(player_id)
print(observation)
```

Available environment ids include:
- `MiniHanabi-v0`
- `MiniHanabi-v0-train`
- `MiniHanabi-v0-raw`
- `ColonelBlotto-v0`
- `ColonelBlotto-v0-train`
- `ColonelBlotto-v0-raw`
- `Negotiation-v0`
- `Negotiation-v0-train`
- `Negotiation-v0-raw`
- `Negotiation-v0-short`
- `Negotiation-v0-short-train`
- `Negotiation-v0-short-raw`
- `Negotiation-v0-long`
- `Negotiation-v0-long-train`
- `Negotiation-v0-long-raw`

The `-train` variants keep the environment logic unchanged but add observation and action wrappers that are convenient for LLM-style interaction. The `-raw` variants expose the bare environment with no default wrappers.
