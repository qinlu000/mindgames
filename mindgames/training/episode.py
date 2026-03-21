from __future__ import annotations

from typing import Any, Optional
from uuid import uuid4

import mindgames as mg
from mindgames.envs.registration import get_env_spec, get_prompt_profile
from mindgames.prompting import get_legal_actions_for_env, normalize_action_for_env
from mindgames.training.contracts import EpisodeStepResult, GameName, GameStep
from mindgames.training.specs import (
    default_max_steps,
    default_reward_player,
    get_game_spec,
    infer_game_name,
    resolve_env_id,
)


def split_prompt_and_state(observation: str) -> tuple[Optional[str], str]:
    if "\n\n" not in observation:
        return None, observation.strip()
    prompt_block, remainder = observation.split("\n\n", 1)
    return prompt_block.strip(), remainder.strip()


def strip_dynamic_role_line(prompt_block: Optional[str]) -> str:
    if not prompt_block:
        return ""
    lines = prompt_block.splitlines()
    if lines and (lines[0].startswith("You are Player ") or lines[0].startswith("You are player ")):
        lines = lines[1:]
    return "\n".join(lines).strip()


def build_system_message(env_id: str, observation: str) -> Optional[str]:
    prompt_profile = get_prompt_profile(env_id)
    game_name = infer_game_name(env_id)
    spec = get_game_spec(game_name) if game_name is not None else None
    prompt_block, _state_text = split_prompt_and_state(observation)
    parts: list[str] = []

    if prompt_profile is not None and prompt_profile.system_prompt:
        parts.append(prompt_profile.system_prompt.strip())

    if spec is not None:
        parts.append(spec.turn_instruction)
        parts.append(spec.snapshot_instruction)

    static_rules = strip_dynamic_role_line(prompt_block)
    if static_rules:
        parts.append(static_rules)

    merged = "\n\n".join(part for part in parts if part)
    return merged or None


def format_state_message(observation: str) -> str:
    _prompt_block, state_text = split_prompt_and_state(observation)
    if not state_text:
        return "Current game state:"
    return f"Current game state:\n{state_text}"


class MindGamesEpisode:
    def __init__(
        self,
        *,
        env: Any,
        game: GameName,
        env_id: str,
        episode_id: Optional[str] = None,
        max_steps: Optional[int] = None,
        reward_player: Optional[int] = None,
    ):
        self.env = env
        self.game = game
        self.spec = get_game_spec(game)
        self.env_id = env_id
        self.episode_id = episode_id or str(uuid4())
        self.max_steps = self.spec.default_max_steps if max_steps is None else int(max_steps)
        self.reward_player = (
            self.spec.default_reward_player if reward_player is None else int(reward_player)
        )
        self._current_step = self._read_current_step()

    @classmethod
    def create(
        cls,
        *,
        game: GameName,
        seed: int,
        env_id: Optional[str] = None,
        max_steps: Optional[int] = None,
        reward_player: Optional[int] = None,
        episode_id: Optional[str] = None,
    ) -> "MindGamesEpisode":
        resolved_env_id = resolve_env_id(game, env_id)
        env = mg.make(resolved_env_id)
        env.reset(num_players=2, seed=seed)
        return cls(
            env=env,
            game=game,
            env_id=resolved_env_id,
            episode_id=episode_id,
            max_steps=max_steps,
            reward_player=reward_player,
        )

    @classmethod
    def attach(
        cls,
        *,
        env: Any,
        game: Optional[GameName] = None,
        env_id: Optional[str] = None,
        episode_id: Optional[str] = None,
        max_steps: Optional[int] = None,
        reward_player: Optional[int] = None,
    ) -> "MindGamesEpisode":
        resolved_env_id = env_id or getattr(env, "env_id", None)
        if not isinstance(resolved_env_id, str):
            raise ValueError("MindGamesEpisode.attach requires an env with a resolved env_id.")
        resolved_game = game or infer_game_name(resolved_env_id)
        if resolved_game is None:
            raise ValueError(f"Could not infer game family from env_id={resolved_env_id!r}.")
        return cls(
            env=env,
            game=resolved_game,
            env_id=resolved_env_id,
            episode_id=episode_id,
            max_steps=max_steps,
            reward_player=reward_player,
        )

    def _read_current_step(self) -> GameStep:
        actor_id, observation = self.env.get_observation()
        observation_text = str(observation)
        env_spec = get_env_spec(self.env)
        prompt_profile = env_spec.prompt_profile
        legal_actions = get_legal_actions_for_env(self.env, observation_text)
        turn_index = int(getattr(getattr(self.env, "state", None), "turn", 0))
        return GameStep(
            game=self.game,
            env_id=self.env_id,
            episode_id=self.episode_id,
            turn_index=turn_index,
            actor_id=int(actor_id),
            observation=observation_text,
            legal_actions=legal_actions,
            action_mode=(prompt_profile.action_mode if prompt_profile is not None else "structured"),
            obs_mode=env_spec.obs_mode,
            reward_mode=env_spec.reward_mode,
        )

    def has_active_step(self) -> bool:
        return self._current_step is not None

    def current_step(self) -> GameStep:
        if self._current_step is None:
            raise RuntimeError("Episode has already terminated.")
        return self._current_step

    def build_initial_prompt_messages(self) -> list[dict[str, str]]:
        step = self.current_step()
        system_message = build_system_message(self.env_id, step.observation)
        messages: list[dict[str, str]] = []
        if system_message:
            messages.append({"role": "system", "content": system_message})
        messages.append({"role": "user", "content": format_state_message(step.observation)})
        return messages

    def step(self, raw_action: str) -> EpisodeStepResult:
        step = self.current_step()
        normalized_action = normalize_action_for_env(self.env, step.observation, raw_action)
        done, step_info = self.env.step(normalized_action)
        safe_info = dict(step_info or {})

        if done:
            terminal_reward = self.spec.extract_terminal_reward(self.env, reward_player=self.reward_player)
            self._current_step = None
            return EpisodeStepResult(
                step=step,
                raw_action=raw_action,
                normalized_action=normalized_action,
                done=True,
                step_info=safe_info,
                next_step=None,
                reward_delta=terminal_reward,
                terminal_reward=terminal_reward,
                terminal_message=self.spec.build_terminal_message(self.env, reward=terminal_reward),
            )

        next_step = self._read_current_step()
        self._current_step = next_step
        return EpisodeStepResult(
            step=step,
            raw_action=raw_action,
            normalized_action=normalized_action,
            done=False,
            step_info=safe_info,
            next_step=next_step,
            reward_delta=0.0,
            terminal_reward=None,
            terminal_message=None,
        )

    def close(self) -> Any:
        return self.env.close()
