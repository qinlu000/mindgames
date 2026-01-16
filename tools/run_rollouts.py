#!/usr/bin/env python3
"""
Run mindgames envs locally (offline) and write rollouts to JSONL.

Designed as a minimal, generic runner for *any* TextArena env id.
Works even if TextArena isn't installed, as long as this repo contains ./textarena.

Example:
  python tools/run_textarena_rollouts.py \\
    --env-id TruthAndDeception-v0-train --num-players 2 --episodes 20 \\
    --agent openrouter:moonshotai/kimi-k2:free --agent openrouter:moonshotai/kimi-k2:free \\
    --out data/tad.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from rollout_utils import _compact_step_rec


def _ensure_pkg_importable() -> None:
    # Ensure both:
    # - `mindgames/` (project root) is importable, and
    # - sibling `textarena/` is importable when running from the mindgames folder.
    project_root = Path(__file__).resolve().parents[1]  # .../mindgames
    repo_root = project_root.parent  # .../ (contains mindgames/ and textarena/)
    sys.path.insert(0, str(project_root))
    sys.path.insert(0, str(repo_root))


_ensure_pkg_importable()

import mindgames as mg  # noqa: E402


@dataclass
class AgentSpec:
    kind: str
    model: Optional[str] = None


def _parse_agent_spec(s: str) -> AgentSpec:
    s = s.strip()
    if ":" in s:
        kind, model = s.split(":", 1)
        kind = kind.strip().lower()
        model = model.strip()
        if not model:
            raise ValueError(f"Invalid --agent spec (missing model): {s}")
        return AgentSpec(kind=kind, model=model)
    return AgentSpec(kind=s.lower(), model=None)


def _build_agent(
    spec: AgentSpec,
    system_prompt: str,
    gen_kwargs: Dict[str, Any],
    *,
    openai_api_key: Optional[str],
    openai_base_url: Optional[str],
    request_timeout_s: Optional[float],
    retry_kwargs: Dict[str, Any],
) -> mg.Agent:
    class _ScriptedAgent(mg.Agent):
        def __init__(self, action: str):
            super().__init__()
            self._action = action

        def __call__(self, observation: str) -> str:  # noqa: ARG002
            return self._action

    def _openai_like_kwargs() -> Dict[str, Any]:
        extra_body: Dict[str, Any] = dict(gen_kwargs.get("extra_body") or {})
        if gen_kwargs.get("chat_template_kwargs") is not None:
            extra_body["chat_template_kwargs"] = dict(gen_kwargs["chat_template_kwargs"])
        if gen_kwargs.get("top_k") is not None:
            extra_body["top_k"] = int(gen_kwargs["top_k"])
        if gen_kwargs.get("repetition_penalty") is not None:
            extra_body["repetition_penalty"] = float(gen_kwargs["repetition_penalty"])
        if gen_kwargs.get("gen_seed") is not None:
            extra_body["seed"] = int(gen_kwargs["gen_seed"])

        kwargs: Dict[str, Any] = {}
        if gen_kwargs.get("temperature") is not None:
            kwargs["temperature"] = float(gen_kwargs["temperature"])
        if gen_kwargs.get("top_p") is not None:
            kwargs["top_p"] = float(gen_kwargs["top_p"])
        if gen_kwargs.get("max_tokens") is not None:
            kwargs["max_tokens"] = int(gen_kwargs["max_tokens"])
        if gen_kwargs.get("presence_penalty") is not None:
            kwargs["presence_penalty"] = float(gen_kwargs["presence_penalty"])
        if gen_kwargs.get("frequency_penalty") is not None:
            kwargs["frequency_penalty"] = float(gen_kwargs["frequency_penalty"])
        if extra_body:
            kwargs["extra_body"] = extra_body
        if request_timeout_s is not None:
            kwargs["timeout"] = float(request_timeout_s)
        return kwargs

    if spec.kind == "human":
        return mg.agents.HumanAgent()

    if spec.kind == "scripted":
        if not spec.model:
            raise ValueError("scripted agent requires a spec like 'scripted:<name>'")
        name = spec.model.strip().lower()
        if name in {"hanabi_discard0", "hanabi_discard", "discard0", "discard"}:
            return _ScriptedAgent("[Discard] 0")
        if name.startswith("const="):
            return _ScriptedAgent(spec.model[len("const=") :])
        raise ValueError(
            f"Unknown scripted agent: {spec.model!r}. Supported: "
            "hanabi_discard0 | const=<action>"
        )

    if spec.kind == "hf":
        agent = mg.agents.HFLocalAgent(model_name=spec.model)  # type: ignore[arg-type]
        agent.system_prompt = system_prompt
        return agent

    if spec.kind == "openai":
        agent = mg.agents.OpenAIAgent(
            model_name=spec.model,  # type: ignore[arg-type]
            system_prompt=system_prompt,
            api_key=(openai_api_key or os.getenv("OPENAI_API_KEY")),
            base_url=(openai_base_url or os.getenv("OPENAI_BASE_URL")),
            **retry_kwargs,
            **_openai_like_kwargs(),
        )
        return agent

    if spec.kind == "qwen":
        agent = mg.agents.QwenAgent(
            model_name=spec.model,  # type: ignore[arg-type]
            system_prompt=system_prompt,
            api_key=(openai_api_key or os.getenv("OPENAI_API_KEY")),
            base_url=(openai_base_url or os.getenv("OPENAI_BASE_URL")),
            **retry_kwargs,
            **_openai_like_kwargs(),
        )
        return agent

    if spec.kind == "gemini":
        max_tokens = gen_kwargs.get("max_tokens", None)
        generation_config: Dict[str, Any] = {}
        if gen_kwargs.get("temperature") is not None:
            generation_config["temperature"] = float(gen_kwargs["temperature"])
        if gen_kwargs.get("top_p") is not None:
            generation_config["top_p"] = float(gen_kwargs["top_p"])
        if gen_kwargs.get("top_k") is not None:
            generation_config["top_k"] = int(gen_kwargs["top_k"])
        if max_tokens is not None:
            generation_config["max_output_tokens"] = int(max_tokens)

        agent = mg.agents.GeminiAgent(
            model_name=spec.model,  # type: ignore[arg-type]
            system_prompt=system_prompt,
            generation_config=generation_config,
        )
        return agent

    if spec.kind == "openrouter":
        kwargs: Dict[str, Any] = {"model_name": spec.model, "system_prompt": system_prompt}  # type: ignore[arg-type]
        kwargs.update(retry_kwargs)
        if gen_kwargs.get("temperature") is not None:
            kwargs["temperature"] = float(gen_kwargs["temperature"])
        if gen_kwargs.get("top_p") is not None:
            kwargs["top_p"] = float(gen_kwargs["top_p"])
        if gen_kwargs.get("max_tokens") is not None:
            kwargs["max_tokens"] = int(gen_kwargs["max_tokens"])
        agent = mg.agents.OpenRouterAgent(**kwargs)
        return agent

    if spec.kind == "ollama":
        kwargs: Dict[str, Any] = {"model_name": spec.model, "system_prompt": system_prompt}  # type: ignore[arg-type]
        kwargs.update(retry_kwargs)
        if gen_kwargs.get("temperature") is not None:
            kwargs["temperature"] = float(gen_kwargs["temperature"])
        if gen_kwargs.get("top_p") is not None:
            kwargs["top_p"] = float(gen_kwargs["top_p"])
        if gen_kwargs.get("max_tokens") is not None:
            kwargs["max_tokens"] = int(gen_kwargs["max_tokens"])
        agent = mg.agents.OllamaAgent(**kwargs)
        return agent

    raise ValueError(f"Unknown agent kind: {spec.kind} (supported: human, hf, openai, qwen, gemini, openrouter, ollama)")


def _jsonl_write(fp, obj: Dict[str, Any]) -> None:
    fp.write(json.dumps(obj, ensure_ascii=False) + "\n")
    # Make progress observable when tailing the file during long-running runs.
    fp.flush()


def _merge_gen_kwargs(base: Dict[str, Any], override: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not override:
        return dict(base)
    merged = dict(base)
    for k, v in override.items():
        if k in {"extra_body", "chat_template_kwargs"} and isinstance(v, dict) and isinstance(merged.get(k), dict):
            vv = dict(merged.get(k) or {})
            vv.update(v)
            merged[k] = vv
        else:
            merged[k] = v
    return merged


def _coerce_episode_id(val: Any) -> Optional[int]:
    if isinstance(val, int):
        return val
    if isinstance(val, str) and val.isdigit():
        return int(val)
    return None


def _normalize_action(env: mg.Env, action: str) -> str:
    normalized = action
    current = env
    while isinstance(current, mg.Wrapper):
        if isinstance(current, mg.ActionWrapper):
            normalized = current.action(normalized)
        current = current.env
    return normalized


def _load_rollout_progress(path: Path) -> Tuple[set[int], set[int]]:
    completed: set[int] = set()
    seen_steps: set[int] = set()
    if not path.exists():
        return completed, seen_steps

    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception as e:
                print(f"WARN: Invalid JSON on line {line_no} of {path}: {e}", file=sys.stderr)
                break
            if not isinstance(rec, dict):
                continue
            ep_id = _coerce_episode_id(rec.get("episode_id"))
            if ep_id is None:
                continue
            rec_type = rec.get("type")
            if rec_type == "episode_end":
                completed.add(ep_id)
            elif rec_type == "step":
                seen_steps.add(ep_id)
    return completed, seen_steps


def _make_env(env_id: str, *, env_kwargs: Optional[Dict[str, Any]] = None):
    if env_id in mg.ENV_REGISTRY:
        return mg.make(env_id=env_id, **(env_kwargs or {}))
    try:
        import textarena as ta  # type: ignore
    except Exception as e:
        raise SystemExit(
            f"env_id={env_id!r} is not registered in mindgames, and importing `textarena` failed: {e}\n"
            "If you want to use TextArena envs, make sure the repo contains a sibling `textarena/` folder "
            "or install `textarena` in the current environment."
        ) from e
    return ta.make(env_id=env_id)


def _game_loop(
    env_id: str,
    num_players: int,
    agents: Dict[int, mg.Agent],
    seed: Optional[int],
    episode_id: int,
    out_fp,
    episode_json_dir: Optional[Path],
    episode_json_max_obs_chars: Optional[int],
    env_kwargs: Optional[Dict[str, Any]],
) -> None:
    env = _make_env(env_id=env_id, env_kwargs=env_kwargs)
    env.reset(num_players=num_players, seed=seed)

    done = False
    step_idx = 0
    episode_steps: list[Dict[str, Any]] = []

    while not done:
        player_id, observation = env.get_observation()
        t0 = time.time()
        action = agents[player_id](observation)
        infer_ms = int((time.time() - t0) * 1000)
        normalized_action = _normalize_action(env, action)

        done, step_info = env.step(action=action)

        step_rec = {
            "type": "step",
            "env_id": env_id,
            "episode_id": episode_id,
            "seed": seed,
            "step": step_idx,
            "player_id": player_id,
            "role": getattr(env.state, "role_mapping", {}).get(player_id, f"Player {player_id}"),
            "observation": observation,
            "action": action,
            "raw_action": action,
            "normalized_action": normalized_action,
            "infer_ms": infer_ms,
            "done": done,
            "step_info": step_info,
        }
        _jsonl_write(out_fp, step_rec)
        if episode_json_dir is not None:
            episode_steps.append(_compact_step_rec(step_rec, max_obs_chars=episode_json_max_obs_chars))

        step_idx += 1

    rewards, game_info = env.close()
    end_rec = {
        "type": "episode_end",
        "env_id": env_id,
        "episode_id": episode_id,
        "seed": seed,
        "rewards": rewards,
        "game_info": game_info,
    }
    _jsonl_write(out_fp, end_rec)

    if episode_json_dir is not None:
        episode_json_dir.mkdir(parents=True, exist_ok=True)
        out_path = episode_json_dir / f"episode_{episode_id:06d}.json"
        out_path.write_text(
            json.dumps(
                {
                    "env_id": env_id,
                    "episode_id": episode_id,
                    "seed": seed,
                    "steps": episode_steps,
                    "episode_end": end_rec,
                },
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--env-id", required=True, help="e.g. Hanabi-v0-train or TruthAndDeception-v0-train")
    ap.add_argument(
        "--env-kwargs",
        default=None,
        help="Optional JSON dict passed to env constructor (mindgames envs only), e.g. '{\"num_rounds\": 2}'.",
    )
    ap.add_argument("--num-players", type=int, required=True)
    ap.add_argument("--episodes", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True, help="Output JSONL path")
    ap.add_argument(
        "--resume",
        action="store_true",
        help="If set, append to an existing JSONL and skip completed episodes.",
    )
    ap.add_argument(
        "--episode-json-dir",
        default=None,
        help="Optional: also write one JSON file per episode into this directory (compact format).",
    )
    ap.add_argument(
        "--episode-json-max-obs-chars",
        type=int,
        default=0,
        help="When --episode-json-dir is set: truncate observation to this many chars (0 = no truncation).",
    )
    ap.add_argument(
        "--agent",
        action="append",
        default=[],
        help="Repeatable. Spec: human | scripted:<name> | hf:<hf_model> | openai:<model> | qwen:<model> | gemini:<model> | openrouter:<model> | ollama:<model>",
    )
    ap.add_argument(
        "--agent-gen",
        action="append",
        default=[],
        help="Optional per-agent JSON dict (repeatable, aligned with --agent order). Keys can include temperature/top_p/max_tokens/extra_body/chat_template_kwargs/etc.",
    )
    ap.add_argument(
        "--system-prompt",
        default="You are a competitive game player. Make sure you read the game instructions carefully, and always follow the required format.",
    )
    ap.add_argument("--openai-base-url", default=None, help="Override OPENAI_BASE_URL (for OpenAI/vLLM-compatible servers)")
    ap.add_argument("--openai-api-key", default=None, help="Override OPENAI_API_KEY (for OpenAI/vLLM-compatible servers)")
    ap.add_argument(
        "--timeout",
        type=float,
        default=300.0,
        help="Per-request timeout in seconds (OpenAI-compatible agents).",
    )
    ap.add_argument("--max-retries", type=int, default=10, help="Max attempts for API calls (OpenAI-compatible agents).")
    ap.add_argument("--retry-initial-delay", type=float, default=2.0, help="Initial retry delay in seconds.")
    ap.add_argument("--retry-max-delay", type=float, default=60.0, help="Maximum retry delay in seconds.")
    ap.add_argument("--temperature", type=float, default=None, help="If set, pass temperature to the backend; otherwise omit it.")
    ap.add_argument("--top-p", type=float, default=None, help="If set, pass top_p to the backend; otherwise omit it.")
    ap.add_argument("--top-k", type=int, default=None)
    ap.add_argument("--repetition-penalty", type=float, default=None)
    ap.add_argument("--presence-penalty", type=float, default=None)
    ap.add_argument("--frequency-penalty", type=float, default=None)
    ap.add_argument("--gen-seed", type=int, default=None, help="Generation seed (OpenAI/vLLM via extra_body.seed when supported)")
    ap.add_argument(
        "--extra-body",
        default=None,
        help="JSON dict merged into OpenAI `extra_body` (OpenAI/vLLM backends only).",
    )
    ap.add_argument(
        "--chat-template-kwargs",
        default=None,
        help="JSON dict merged into `extra_body.chat_template_kwargs` (vLLM-specific).",
    )
    ap.add_argument(
        "--disable-thinking",
        action="store_true",
        help="Convenience for Qwen thinking: sets chat_template_kwargs.enable_thinking=false (ignored if --chat-template-kwargs is set)",
    )
    ap.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help="For chat APIs (OpenAI/OpenRouter/Ollama). If omitted, do not send max_tokens (let the backend decide).",
    )
    args = ap.parse_args()
    episode_json_dir = Path(args.episode_json_dir) if args.episode_json_dir else None
    if args.env_kwargs is not None:
        try:
            env_kwargs = json.loads(args.env_kwargs)
        except Exception as e:
            raise SystemExit(f"Invalid --env-kwargs JSON: {e}") from e
        if not isinstance(env_kwargs, dict):
            raise SystemExit("--env-kwargs must be a JSON object (dict)")
    else:
        env_kwargs = None

    if not args.agent:
        raise SystemExit("Provide at least one --agent. If num_players>1, either repeat --agent or pass one to replicate.")

    specs = [_parse_agent_spec(a) for a in args.agent]
    if len(specs) == 1 and args.num_players > 1:
        specs = specs * args.num_players
    if len(specs) != args.num_players:
        raise SystemExit(f"Need exactly {args.num_players} agents; got {len(specs)} via --agent")

    agent_gen_raw: List[Optional[Dict[str, Any]]] = []
    if args.agent_gen:
        for idx, s in enumerate(args.agent_gen):
            try:
                obj = json.loads(s)
            except Exception as e:
                raise SystemExit(f"Invalid --agent-gen JSON at index {idx}: {e}") from e
            if not isinstance(obj, dict):
                raise SystemExit(f"--agent-gen entries must be JSON objects (dict); got {type(obj)} at index {idx}")
            agent_gen_raw.append(obj)
        if len(agent_gen_raw) == 1 and args.num_players > 1:
            agent_gen_raw = agent_gen_raw * args.num_players
        if len(agent_gen_raw) != args.num_players:
            raise SystemExit(f"Need 0, 1, or {args.num_players} --agent-gen entries; got {len(agent_gen_raw)}")
    else:
        agent_gen_raw = [None] * args.num_players

    if args.chat_template_kwargs is not None:
        try:
            chat_template_kwargs = json.loads(args.chat_template_kwargs)
        except Exception as e:
            raise SystemExit(f"Invalid --chat-template-kwargs JSON: {e}") from e
        if not isinstance(chat_template_kwargs, dict):
            raise SystemExit("--chat-template-kwargs must be a JSON object (dict)")
    elif args.disable_thinking:
        chat_template_kwargs = {"enable_thinking": False}
    else:
        chat_template_kwargs = None

    if args.extra_body is not None:
        try:
            extra_body = json.loads(args.extra_body)
        except Exception as e:
            raise SystemExit(f"Invalid --extra-body JSON: {e}") from e
        if not isinstance(extra_body, dict):
            raise SystemExit("--extra-body must be a JSON object (dict)")
    else:
        extra_body = None

    gen_kwargs = {
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "repetition_penalty": args.repetition_penalty,
        "presence_penalty": args.presence_penalty,
        "frequency_penalty": args.frequency_penalty,
        "gen_seed": args.gen_seed,
        "extra_body": extra_body,
        "chat_template_kwargs": chat_template_kwargs,
        "max_tokens": args.max_tokens,
    }

    request_timeout_s = float(args.timeout) if args.timeout and args.timeout > 0 else None
    retry_kwargs = {
        "max_retries": int(args.max_retries),
        "retry_initial_delay_s": float(args.retry_initial_delay),
        "retry_max_delay_s": float(args.retry_max_delay),
    }

    agents: Dict[int, mg.Agent] = {}
    for i in range(args.num_players):
        merged_gen = _merge_gen_kwargs(gen_kwargs, agent_gen_raw[i])
        agents[i] = _build_agent(
            specs[i],
            args.system_prompt,
            merged_gen,
            openai_api_key=args.openai_api_key,
            openai_base_url=args.openai_base_url,
            request_timeout_s=request_timeout_s,
            retry_kwargs=retry_kwargs,
        )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    completed_ids: set[int] = set()
    if args.resume and out_path.exists():
        if not out_path.is_file():
            raise SystemExit(f"--out exists but is not a file: {out_path}")
        completed_ids, seen_steps = _load_rollout_progress(out_path)
        partial = sorted(seen_steps - completed_ids)
        if partial:
            print(
                f"WARN: Found incomplete episodes in {out_path}: {partial}. "
                "Resuming will rerun them and append new records.",
                file=sys.stderr,
            )
        if set(range(args.episodes)).issubset(completed_ids):
            print(f"All {args.episodes} episodes already completed in {out_path}.")
            return 0

    file_mode = "a" if args.resume and out_path.exists() else "w"
    # Use line-buffered output so rollouts are visible in near-real time.
    with out_path.open(file_mode, encoding="utf-8", buffering=1) as f:
        for ep in range(args.episodes):
            if ep in completed_ids:
                continue
            seed = args.seed + ep if args.seed is not None else None
            _game_loop(
                env_id=args.env_id,
                num_players=args.num_players,
                agents=agents,
                seed=seed,
                episode_id=ep,
                out_fp=f,
                episode_json_dir=episode_json_dir,
                episode_json_max_obs_chars=(None if int(args.episode_json_max_obs_chars) == 0 else int(args.episode_json_max_obs_chars)),
                env_kwargs=env_kwargs,
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
