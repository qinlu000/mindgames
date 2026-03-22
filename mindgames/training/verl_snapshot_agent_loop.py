from __future__ import annotations

import asyncio
import logging
import os
import random
import time
from contextlib import contextmanager
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Optional
from uuid import uuid4

try:
    from verl.experimental.agent_loop.agent_loop import AgentLoopBase, AgentLoopOutput, register
    from verl.utils.profiler import simple_timer
except ModuleNotFoundError:
    @dataclass
    class AgentLoopOutput:  # type: ignore[no-redef]
        prompt_ids: list[int]
        response_ids: list[int]
        response_mask: list[int]
        response_logprobs: Optional[list[float]] = None
        routed_experts: Optional[Any] = None
        multi_modal_data: Optional[dict[str, Any]] = None
        reward_score: Optional[float] = None
        num_turns: int = 0
        metrics: dict[str, Any] | None = None
        extra_fields: dict[str, Any] | None = None

        def __post_init__(self) -> None:
            if self.metrics is None:
                self.metrics = {}
            if self.extra_fields is None:
                self.extra_fields = {}

    class AgentLoopBase:  # type: ignore[no-redef]
        def __init__(
            self,
            trainer_config: Any,
            server_manager: Any,
            tokenizer: Any,
            processor: Any,
            dataset_cls: Any,
            data_config: Any,
            **kwargs: Any,
        ):
            del dataset_cls, kwargs
            config = getattr(trainer_config, "config", trainer_config)
            self.config = config
            actor_rollout_ref = getattr(config, "actor_rollout_ref", None)
            rollout = getattr(actor_rollout_ref, "rollout", None) if actor_rollout_ref is not None else None
            self.rollout_config = rollout or getattr(config, "rollout", SimpleNamespace())
            self.server_manager = server_manager
            self.tokenizer = tokenizer
            self.processor = processor
            self.data_config = getattr(data_config, "config", data_config)
            self.loop = None

        async def process_vision_info(self, messages: list[dict]) -> dict[str, Any]:
            del messages
            return {}

        async def apply_chat_template(
            self,
            messages: list[dict],
            tools: list[dict] | None = None,
            images: list[Any] | None = None,
            videos: list[Any] | None = None,
            remove_system_prompt: bool = False,
        ) -> list[int]:
            del messages, tools, images, videos, remove_system_prompt
            raise NotImplementedError("apply_chat_template requires verl to be installed.")

    def register(agent_name: str):  # type: ignore[no-redef]
        def decorator(subclass: type[AgentLoopBase]) -> type[AgentLoopBase]:
            return subclass

        return decorator

    @contextmanager
    def simple_timer(name: str, metrics: dict[str, Any]):  # type: ignore[no-redef]
        start = time.perf_counter()
        try:
            yield
        finally:
            metrics[name] = float(time.perf_counter() - start)


logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

SNAPSHOT_AGENT_LOOP_NAME = "mindgames_snapshot_episode"


@dataclass(frozen=True)
class SnapshotStepCandidate:
    messages: list[dict[str, str]]
    prompt_ids: list[int]
    response_ids: list[int]
    response_logprobs: Optional[list[float]]
    actor_id: int
    turn_index: int
    normalized_action: str


def _coerce_rollout_value(config: Any, name: str, default: int) -> int:
    value = getattr(config, name, default)
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _merge_metric(metrics: dict[str, Any], step_metrics: dict[str, Any], key: str) -> None:
    if key not in step_metrics:
        return
    current = float(metrics.get(key, 0.0))
    metrics[key] = current + float(step_metrics[key])


def _merge_num_preempted(metrics: dict[str, Any], num_preempted: Any) -> None:
    value = -1 if num_preempted is None else int(num_preempted)
    current = metrics.get("num_preempted")
    if current is None or int(current) < 0:
        metrics["num_preempted"] = value
        return
    if value >= 0:
        metrics["num_preempted"] = int(current) + value


@register(SNAPSHOT_AGENT_LOOP_NAME)
class MindGamesSnapshotEpisodeAgentLoop(AgentLoopBase):
    """Run a full env episode and retain every decision-step snapshot."""

    def __init__(self, *args: Any, selection_strategy: str = "uniform", **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.selection_strategy = str(selection_strategy)
        self.prompt_length = _coerce_rollout_value(self.rollout_config, "prompt_length", 1024)
        self.response_length = _coerce_rollout_value(self.rollout_config, "response_length", 512)

    def _build_step_messages(self, step: Any) -> list[dict[str, str]]:
        from mindgames.training.episode import build_system_message, format_state_message

        messages: list[dict[str, str]] = []
        system_message = build_system_message(step.env_id, step.observation)
        if system_message:
            messages.append({"role": "system", "content": system_message})
        messages.append({"role": "user", "content": format_state_message(step.observation)})
        return messages

    def _parse_episode_kwargs(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        extra_info = kwargs.get("extra_info") or {}
        if not isinstance(extra_info, dict):
            raise ValueError("MindGames snapshot loop expects extra_info to be a dict.")
        raw_kwargs = extra_info.get("interaction_kwargs") or {}
        if not isinstance(raw_kwargs, dict):
            raise ValueError("MindGames snapshot loop expects interaction_kwargs to be a dict.")
        episode_kwargs = dict(raw_kwargs)
        episode_kwargs.pop("name", None)
        return episode_kwargs

    async def _decode_response(self, response_ids: list[int]) -> str:
        running_loop = asyncio.get_running_loop()
        return await running_loop.run_in_executor(
            None,
            lambda: self.tokenizer.decode(response_ids, skip_special_tokens=True),
        )

    def _select_candidate(self, candidates: list[SnapshotStepCandidate]) -> SnapshotStepCandidate:
        if not candidates:
            raise RuntimeError("MindGames snapshot loop did not collect any decision steps.")
        if self.selection_strategy == "first":
            return candidates[0]
        if self.selection_strategy == "last":
            return candidates[-1]
        if self.selection_strategy == "uniform":
            return random.choice(candidates)
        raise ValueError(f"Unsupported selection_strategy={self.selection_strategy!r}.")

    async def _collect_episode(
        self,
        *,
        episode: Any,
        sampling_params: dict[str, Any],
        request_id: str,
    ) -> tuple[list[SnapshotStepCandidate], float, dict[str, Any]]:
        candidates: list[SnapshotStepCandidate] = []
        metrics: dict[str, Any] = {"num_preempted": -1}
        final_reward = 0.0

        while episode.has_active_step():
            step = episode.current_step()
            messages = self._build_step_messages(step)
            prompt_ids = await self.apply_chat_template(messages)

            step_metrics: dict[str, Any] = {}
            with simple_timer("generate_sequences", step_metrics):
                output = await self.server_manager.generate(
                    request_id=request_id,
                    prompt_ids=prompt_ids,
                    sampling_params=sampling_params,
                    image_data=None,
                    video_data=None,
                )
            _merge_metric(metrics, step_metrics, "generate_sequences")
            _merge_num_preempted(metrics, getattr(output, "num_preempted", None))

            response_ids = list(getattr(output, "token_ids", []) or [])
            if not response_ids:
                raise RuntimeError("MindGames snapshot loop received an empty model response.")
            assistant_message = await self._decode_response(response_ids)
            transition = episode.step(assistant_message)
            candidates.append(
                SnapshotStepCandidate(
                    messages=[dict(message) for message in messages],
                    prompt_ids=list(prompt_ids),
                    response_ids=response_ids,
                    response_logprobs=(
                        list(getattr(output, "log_probs", []) or [])
                        if getattr(output, "log_probs", None) is not None
                        else None
                    ),
                    actor_id=int(step.actor_id),
                    turn_index=int(step.turn_index),
                    normalized_action=str(transition.normalized_action),
                )
            )
            if transition.done:
                final_reward = float(transition.terminal_reward or 0.0)

        return candidates, final_reward, metrics

    async def run(self, sampling_params: dict[str, Any], **kwargs: Any) -> AgentLoopOutput:
        from mindgames.training.episode import MindGamesEpisode

        episode_kwargs = self._parse_episode_kwargs(kwargs)
        request_id = uuid4().hex
        episode = MindGamesEpisode.create(
            game=episode_kwargs["game"],
            seed=int(episode_kwargs["seed"]),
            env_id=episode_kwargs.get("env_id"),
            max_steps=episode_kwargs.get("max_steps"),
            reward_player=episode_kwargs.get("reward_player"),
            episode_id=request_id,
        )
        try:
            candidates, final_reward, metrics = await self._collect_episode(
                episode=episode,
                sampling_params=sampling_params,
                request_id=request_id,
            )
        finally:
            episode.close()

        selected = self._select_candidate(candidates)
        response_mask = [1] * len(selected.response_ids)
        episode_step_data = [
            {
                "messages": [dict(message) for message in candidate.messages],
                "prompt_ids": list(candidate.prompt_ids),
                "response_ids": list(candidate.response_ids),
                "response_logprobs": (
                    None
                    if candidate.response_logprobs is None
                    else list(candidate.response_logprobs)
                ),
                "actor_id": int(candidate.actor_id),
                "turn_index": int(candidate.turn_index),
                "normalized_action": candidate.normalized_action,
            }
            for candidate in candidates
        ]
        output: AgentLoopOutput = AgentLoopOutput(
            prompt_ids=selected.prompt_ids,
            response_ids=selected.response_ids,
            response_mask=response_mask,
            response_logprobs=selected.response_logprobs,
            reward_score=float(final_reward),
            num_turns=2,
            metrics=metrics,
            extra_fields={
                "turn_scores": [float(final_reward)],
                "tool_rewards": [],
                "episode_steps": len(candidates),
                "episode_step_data": episode_step_data,
                "selected_actor_id": int(selected.actor_id),
                "selected_turn_index": int(selected.turn_index),
                "selected_normalized_action": selected.normalized_action,
                "selection_strategy": self.selection_strategy,
            },
        )
        return output
