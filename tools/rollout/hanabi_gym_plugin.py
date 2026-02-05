#!/usr/bin/env python3
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import mindgames as mg
from swift.plugin import Env, envs


class HanabiGymEnv(Env):
    def __init__(self, env_config: Dict[str, Any]):
        super().__init__(env_config)
        self.env = None
        self.env_id = None
        self.num_players = 2

    def _build_env(self, env_config: Dict[str, Any]):
        env_id = env_config.get("env_id", "Hanabi-v0-train")
        kwargs = {}
        max_steps = env_config.get("max_steps")
        if max_steps is not None:
            kwargs["max_steps"] = int(max_steps)
        return env_id, mg.make(env_id, **kwargs)

    async def reset(self, config) -> Tuple[str, Dict[str, Any], str]:
        env_config = dict(self.env_config or {})
        if getattr(config, "data_dict", None) and config.data_dict.get("env_config"):
            env_config.update(config.data_dict["env_config"])

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

        info.setdefault("current_player_id", self.env.state.current_player_id)
        return obs, reward, done, info

    async def close(self) -> None:
        if self.env is not None:
            self.env.close()


envs["hanabi_env"] = HanabiGymEnv
