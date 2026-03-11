#!/usr/bin/env python3
from __future__ import annotations

import math
import os
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple

import mindgames as mg
try:
    # ms-swift >= 4.0
    from swift.infer_engine.protocol import RolloutOutput
    from swift.rollout import ContextManager, Env, context_managers, envs, multi_turns
    from swift.rollout.multi_turn import GYMScheduler
    from swift.utils import remove_response
except ImportError:
    # ms-swift < 4.0
    from swift.plugin import ContextManager, Env, context_managers, envs
    GYMScheduler = None
    RolloutOutput = None
    multi_turns = None


def _as_int(value: Any, default: int) -> int:
    if value is None:
        return default
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        return int(value)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return default
        try:
            return int(float(text))
        except ValueError:
            return default
    return default


def _as_bool(value: Any, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"1", "true", "yes", "y", "on"}:
            return True
        if text in {"0", "false", "no", "n", "off"}:
            return False
    return default


class HanabiRecentTurnsContextManager(ContextManager):
    """Keep only the most recent N user turns to prevent context blow-up."""

    def __init__(self, ctx_config: Dict[str, Any]):
        cfg = dict(ctx_config or {})
        super().__init__(cfg)
        env_max_turns = _as_int(os.getenv("HANABI_CTX_MAX_TURNS"), 1)
        env_keep_system = _as_bool(os.getenv("HANABI_CTX_KEEP_SYSTEM"), True)

        self.max_turns = max(1, _as_int(cfg.get("max_turns"), env_max_turns))
        self.keep_system = _as_bool(cfg.get("keep_system"), env_keep_system)

    def manage_context(self, history: List[Dict[str, Any]], trajectory_id: str) -> List[Dict[str, Any]]:
        del trajectory_id
        if not history:
            return history

        messages = list(history)
        system_messages: List[Dict[str, Any]] = []
        non_system_messages = messages
        if messages and messages[0].get("role") == "system":
            if self.keep_system:
                system_messages = [messages[0]]
            non_system_messages = messages[1:]

        if not non_system_messages:
            return system_messages

        user_seen = 0
        start_idx = 0
        for idx in range(len(non_system_messages) - 1, -1, -1):
            if non_system_messages[idx].get("role") == "user":
                user_seen += 1
                if user_seen >= self.max_turns:
                    start_idx = idx
                    break

        if user_seen < self.max_turns:
            start_idx = 0

        return system_messages + non_system_messages[start_idx:]


context_managers["hanabi_recent_turns"] = HanabiRecentTurnsContextManager


class HanabiGymEnv(Env):
    def __init__(self, env_config: Dict[str, Any]):
        super().__init__(env_config)
        self.env = None
        self.env_id = None
        self.num_players = 2
        self._player_reward_stats: Dict[int, Dict[str, float]] = {}
        self._marshal_agent_norm = False
        self._marshal_agent_norm_method = "mean_std"
        self._marshal_agent_norm_clip: Optional[float] = None
        self._marshal_agent_norm_warmup = 8
        self._marshal_agent_norm_eps = 1e-6

    def _build_env(self, env_config: Dict[str, Any]):
        plugin_only_keys = {
            "name",
            "env_id",
            "num_players",
            "seed",
            "marshal_agent_norm",
            "marshal_agent_norm_method",
            "marshal_agent_norm_clip",
            "marshal_agent_norm_warmup",
            "marshal_agent_norm_eps",
            "marshal_agent_norm_reset_stats",
        }
        env_id = env_config.get("env_id", "Hanabi-v0-train")
        kwargs = {k: v for k, v in env_config.items() if k not in plugin_only_keys}
        if "reward_on_score_gain" not in kwargs and "marshal_dense_reward" not in kwargs:
            kwargs["reward_on_score_gain"] = True
        if "max_steps" in kwargs and kwargs["max_steps"] is not None:
            kwargs["max_steps"] = int(kwargs["max_steps"])
        return env_id, mg.make(env_id, **kwargs)

    def _setup_agent_norm(self, env_config: Dict[str, Any]) -> None:
        self._marshal_agent_norm = bool(env_config.get("marshal_agent_norm", False))
        self._marshal_agent_norm_method = str(env_config.get("marshal_agent_norm_method", "mean_std")).lower()
        self._marshal_agent_norm_warmup = int(env_config.get("marshal_agent_norm_warmup", 8))
        self._marshal_agent_norm_eps = float(env_config.get("marshal_agent_norm_eps", 1e-6))
        clip_val = env_config.get("marshal_agent_norm_clip")
        self._marshal_agent_norm_clip = None if clip_val is None else float(clip_val)

        if self._marshal_agent_norm_method not in {"mean", "mean_std"}:
            raise ValueError(
                f"Unsupported marshal_agent_norm_method='{self._marshal_agent_norm_method}'. "
                "Use 'mean' or 'mean_std'."
            )

        if bool(env_config.get("marshal_agent_norm_reset_stats", False)):
            self._player_reward_stats = {}

    def _update_reward_stats(self, player_id: int, reward: float) -> None:
        stats = self._player_reward_stats.setdefault(player_id, {"count": 0.0, "mean": 0.0, "m2": 0.0})
        count = stats["count"] + 1.0
        delta = reward - stats["mean"]
        mean = stats["mean"] + delta / count
        delta2 = reward - mean
        m2 = stats["m2"] + delta * delta2
        stats.update({"count": count, "mean": mean, "m2": m2})

    def _normalize_reward_by_player(self, raw_reward: float, player_id: int) -> float:
        stats = self._player_reward_stats.setdefault(player_id, {"count": 0.0, "mean": 0.0, "m2": 0.0})
        count = int(stats["count"])
        warmup_ok = count >= self._marshal_agent_norm_warmup

        if not warmup_ok:
            reward = raw_reward
        elif self._marshal_agent_norm_method == "mean":
            reward = raw_reward - stats["mean"]
        else:
            variance = 0.0
            if count > 1:
                variance = max(stats["m2"] / (count - 1), 0.0)
            std = math.sqrt(variance)
            if std <= self._marshal_agent_norm_eps:
                reward = 0.0
            else:
                reward = (raw_reward - stats["mean"]) / (std + self._marshal_agent_norm_eps)

        if self._marshal_agent_norm_clip is not None:
            reward = max(-self._marshal_agent_norm_clip, min(self._marshal_agent_norm_clip, reward))

        self._update_reward_stats(player_id, raw_reward)
        return float(reward)

    async def reset(self, config) -> Tuple[str, Dict[str, Any], str]:
        env_config = dict(self.env_config or {})
        if getattr(config, "data_dict", None) and config.data_dict.get("env_config"):
            env_config.update(config.data_dict["env_config"])

        self._setup_agent_norm(env_config)
        self.env_id, self.env = self._build_env(env_config)
        self.num_players = int(env_config.get("num_players", 2))
        seed = env_config.get("seed")

        self.env.reset(num_players=self.num_players, seed=seed)
        _, obs = self.env.get_observation()
        info = {"env_id": self.env_id, "num_players": self.num_players}
        return obs, info, ""

    async def step(self, action) -> Tuple[str, float, bool, Dict[str, Any]]:
        if not action:
            action_text = ""
        else:
            last = action[-1]
            action_text = last.get("content", "") if isinstance(last, dict) else str(last)

        acting_player_id = self.env.state.current_player_id
        done, info = self.env.step(action_text)
        _, obs = self.env.get_observation()

        reward = 0.0
        if info is None:
            info = {}
        if "reward" in info:
            reward = float(info["reward"])
        elif "step_reward" in info:
            reward = float(info["step_reward"])
        else:
            rewards = None
            if isinstance(info.get("rewards"), dict):
                rewards = info["rewards"]
            elif isinstance(info.get("reward_dict"), dict):
                rewards = info["reward_dict"]
            elif done:
                rewards = getattr(self.env.state, "rewards", None)

            if isinstance(rewards, dict):
                reward = float(rewards.get(acting_player_id, 0.0))

        raw_reward = reward
        if self._marshal_agent_norm:
            reward = self._normalize_reward_by_player(raw_reward, acting_player_id)
            info.setdefault("raw_reward", raw_reward)
            info.setdefault("reward_norm_method", self._marshal_agent_norm_method)
            info.setdefault("reward_norm_player", acting_player_id)

        info.setdefault("acting_player_id", acting_player_id)
        info.setdefault("current_player_id", self.env.state.current_player_id)
        return obs, reward, done, info

    async def close(self) -> None:
        if self.env is not None:
            self.env.close()


envs["hanabi_env"] = HanabiGymEnv


if GYMScheduler is not None and RolloutOutput is not None and multi_turns is not None:

    class HanabiTokenizedGYMScheduler(GYMScheduler):
        """Return per-turn token ids so training can avoid re-tokenizing rollout text."""

        @staticmethod
        def _extract_logprobs(response_choice: Any) -> List[float]:
            if response_choice.logprobs is None:
                return []
            if "content" in response_choice.logprobs:
                return [item["logprob"] for item in response_choice.logprobs["content"]]
            return []

        async def run(self, infer_request, request_config, **kwargs):
            env_config = infer_request.data_dict.get("env_config", {})
            ctx_config = infer_request.data_dict.get("ctx_config", {})

            env = None
            context_manager = None
            try:
                env = await self._create_env(env_config)
                context_manager = await self._create_context_manager(ctx_config)

                observation, info, system_message = await env.reset(infer_request)

                messages: List[Dict[str, Any]] = []
                if system_message:
                    messages.append({"role": "system", "content": system_message})
                messages.append({"role": "user", "content": observation})

                current_request = deepcopy(infer_request)
                current_turn = 1
                done = False
                total_reward = 0.0
                step_rewards: List[float] = []
                trajectory_id = infer_request.uuid
                trajectory_info = [info]
                total_response_ids: List[List[int]] = []
                total_response_loss_mask: List[List[int]] = []
                total_rollout_logprobs: List[List[float]] = []
                response = None

                while not done and current_turn <= (self.max_turns or float("inf")):
                    messages = context_manager.manage_context(messages, trajectory_id)
                    current_request.messages = messages
                    remove_response(current_request.messages)

                    response = await self.infer_engine.infer_async(current_request, request_config, **kwargs)
                    response_choice = response.choices[0]
                    completion = response_choice.message.content
                    messages.append({"role": "assistant", "content": completion})

                    response_token_ids = list(response_choice.token_ids) if response_choice.token_ids else []
                    if response_token_ids:
                        total_response_ids.append(response_token_ids)
                        total_response_loss_mask.append([1] * len(response_token_ids))

                    current_logprobs = self._extract_logprobs(response_choice)
                    if current_logprobs:
                        total_rollout_logprobs.append(current_logprobs)

                    next_obs, reward, done, step_info = await env.step(deepcopy(messages))

                    total_reward += reward
                    step_rewards.append(reward)
                    trajectory_info.append(step_info)

                    if not done:
                        messages.append({"role": "user", "content": next_obs})
                        current_request.messages = messages
                        current_turn += 1

                if response is None:
                    raise RuntimeError("Hanabi gym rollout finished without producing a response.")

                final_rollout_logprobs = total_rollout_logprobs
                if total_rollout_logprobs:
                    total_logprob_count = sum(len(turn_lps) for turn_lps in total_rollout_logprobs)
                    total_loss_mask_1_count = sum(sum(mask) for mask in total_response_loss_mask)
                    if total_logprob_count != total_loss_mask_1_count:
                        final_rollout_logprobs = []

                return RolloutOutput(
                    response=response,
                    messages=messages,
                    response_token_ids=total_response_ids,
                    response_loss_mask=total_response_loss_mask,
                    rollout_infos={
                        "num_turns": current_turn,
                        "trajectory_id": trajectory_id,
                        "total_reward": total_reward,
                        "step_rewards": step_rewards,
                        "trajectory_info": trajectory_info,
                    },
                    rollout_logprobs=final_rollout_logprobs,
                )
            finally:
                if env is not None:
                    await self._close_env_async(env)


    multi_turns["hanabi_gym_scheduler"] = HanabiTokenizedGYMScheduler
