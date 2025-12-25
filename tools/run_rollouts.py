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


def _ensure_pkg_importable() -> None:
    # Ensure `mindgames/` (project root) is on sys.path when running as a script.
    project_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(project_root))


_ensure_pkg_importable()

import mindgames as ta  # noqa: E402


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
) -> ta.Agent:
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

        kwargs: Dict[str, Any] = {
            "temperature": gen_kwargs.get("temperature", 0.2),
            "top_p": gen_kwargs.get("top_p", 1.0),
            "max_tokens": gen_kwargs.get("max_tokens", 2048),
        }
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
        return ta.agents.HumanAgent()

    if spec.kind == "hf":
        agent = ta.agents.HFLocalAgent(
            model_name=spec.model,  # type: ignore[arg-type]
            max_new_tokens=int(gen_kwargs.get("max_new_tokens", 512)),
        )
        agent.system_prompt = system_prompt
        return agent

    if spec.kind == "openai":
        agent = ta.agents.OpenAIAgent(
            model_name=spec.model,  # type: ignore[arg-type]
            system_prompt=system_prompt,
            api_key=(openai_api_key or os.getenv("OPENAI_API_KEY")),
            base_url=(openai_base_url or os.getenv("OPENAI_BASE_URL")),
            **_openai_like_kwargs(),
        )
        return agent

    if spec.kind == "qwen":
        agent = ta.agents.QwenAgent(
            model_name=spec.model,  # type: ignore[arg-type]
            system_prompt=system_prompt,
            api_key=(openai_api_key or os.getenv("OPENAI_API_KEY")),
            base_url=(openai_base_url or os.getenv("OPENAI_BASE_URL")),
            **_openai_like_kwargs(),
        )
        return agent

    if spec.kind == "openrouter":
        agent = ta.agents.OpenRouterAgent(
            model_name=spec.model,  # type: ignore[arg-type]
            system_prompt=system_prompt,
            temperature=gen_kwargs.get("temperature", 0.2),
            top_p=gen_kwargs.get("top_p", 1.0),
            max_tokens=gen_kwargs.get("max_tokens", 2048),
        )
        return agent

    if spec.kind == "ollama":
        agent = ta.agents.OllamaAgent(
            model_name=spec.model,  # type: ignore[arg-type]
            system_prompt=system_prompt,
            temperature=gen_kwargs.get("temperature", 0.2),
            top_p=gen_kwargs.get("top_p", 1.0),
            max_tokens=gen_kwargs.get("max_tokens", 2048),
        )
        return agent

    raise ValueError(f"Unknown agent kind: {spec.kind} (supported: human, hf, openai, qwen, openrouter, ollama)")


def _jsonl_write(fp, obj: Dict[str, Any]) -> None:
    fp.write(json.dumps(obj, ensure_ascii=False) + "\n")


def _game_loop(
    env_id: str,
    num_players: int,
    agents: Dict[int, ta.Agent],
    seed: Optional[int],
    episode_id: int,
    out_fp,
) -> None:
    env = ta.make(env_id=env_id)
    env.reset(num_players=num_players, seed=seed)

    done = False
    step_idx = 0

    while not done:
        player_id, observation = env.get_observation()
        t0 = time.time()
        action = agents[player_id](observation)
        infer_ms = int((time.time() - t0) * 1000)

        done, step_info = env.step(action=action)

        _jsonl_write(
            out_fp,
            {
                "type": "step",
                "env_id": env_id,
                "episode_id": episode_id,
                "seed": seed,
                "step": step_idx,
                "player_id": player_id,
                "role": getattr(env.state, "role_mapping", {}).get(player_id, f"Player {player_id}"),
                "observation": observation,
                "action": action,
                "infer_ms": infer_ms,
                "done": done,
                "step_info": step_info,
            },
        )

        step_idx += 1

    rewards, game_info = env.close()
    _jsonl_write(
        out_fp,
        {
            "type": "episode_end",
            "env_id": env_id,
            "episode_id": episode_id,
            "seed": seed,
            "rewards": rewards,
            "game_info": game_info,
        },
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--env-id", required=True, help="e.g. Hanabi-v0-train or TruthAndDeception-v0-train")
    ap.add_argument("--num-players", type=int, required=True)
    ap.add_argument("--episodes", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True, help="Output JSONL path")
    ap.add_argument(
        "--agent",
        action="append",
        default=[],
        help="Repeatable. Spec: human | hf:<hf_model> | openai:<model> | qwen:<model> | openrouter:<model> | ollama:<model>",
    )
    ap.add_argument(
        "--system-prompt",
        default="You are a competitive game player. Make sure you read the game instructions carefully, and always follow the required format.",
    )
    ap.add_argument("--openai-base-url", default=None, help="Override OPENAI_BASE_URL (for OpenAI/vLLM-compatible servers)")
    ap.add_argument("--openai-api-key", default=None, help="Override OPENAI_API_KEY (for OpenAI/vLLM-compatible servers)")
    ap.add_argument("--timeout", type=float, default=None, help="Per-request timeout in seconds (for OpenAI/vLLM when supported)")
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--top-p", type=float, default=1.0)
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
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--max-tokens", type=int, default=2048, help="For chat APIs (OpenAI/OpenRouter/Ollama)")
    args = ap.parse_args()

    if not args.agent:
        raise SystemExit("Provide at least one --agent. If num_players>1, either repeat --agent or pass one to replicate.")

    specs = [_parse_agent_spec(a) for a in args.agent]
    if len(specs) == 1 and args.num_players > 1:
        specs = specs * args.num_players
    if len(specs) != args.num_players:
        raise SystemExit(f"Need exactly {args.num_players} agents; got {len(specs)} via --agent")

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
        "max_new_tokens": args.max_new_tokens,
    }

    agents: Dict[int, ta.Agent] = {
        i: _build_agent(
            specs[i],
            args.system_prompt,
            gen_kwargs,
            openai_api_key=args.openai_api_key,
            openai_base_url=args.openai_base_url,
            request_timeout_s=args.timeout,
        )
        for i in range(args.num_players)
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("a", encoding="utf-8") as f:
        for ep in range(args.episodes):
            seed = args.seed + ep if args.seed is not None else None
            _game_loop(
                env_id=args.env_id,
                num_players=args.num_players,
                agents=agents,
                seed=seed,
                episode_id=ep,
                out_fp=f,
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
