from __future__ import annotations

from typing import Any

import numpy as np
import torch


def _as_int_list(values: Any) -> list[int]:
    if values is None:
        return []
    if isinstance(values, list):
        return [int(value) for value in values]
    return [int(value) for value in list(values)]


def _as_float_list(values: Any) -> list[float] | None:
    if values is None:
        return None
    if isinstance(values, list):
        return [float(value) for value in values]
    return [float(value) for value in list(values)]


def _left_pad(token_ids: list[int], *, pad_value: int, target_length: int) -> tuple[torch.Tensor, torch.Tensor]:
    token_ids = token_ids[-target_length:]
    pad_size = max(0, target_length - len(token_ids))
    padded = [pad_value] * pad_size + token_ids
    attention_mask = [0] * pad_size + [1] * len(token_ids)
    return (
        torch.tensor(padded, dtype=torch.long),
        torch.tensor(attention_mask, dtype=torch.long),
    )


def _right_pad(token_ids: list[int], *, pad_value: int, target_length: int) -> tuple[torch.Tensor, torch.Tensor]:
    token_ids = token_ids[:target_length]
    pad_size = max(0, target_length - len(token_ids))
    padded = token_ids + [pad_value] * pad_size
    attention_mask = [1] * len(token_ids) + [0] * pad_size
    return (
        torch.tensor(padded, dtype=torch.long),
        torch.tensor(attention_mask, dtype=torch.long),
    )


def _compute_position_ids(attention_mask: torch.Tensor) -> torch.Tensor:
    position_ids = attention_mask.cumsum(dim=-1) - 1
    position_ids = position_ids * attention_mask
    return position_ids.to(dtype=torch.long)


def _build_step_uid(base_uid: str, *, rollout_index: int, turn_index: int, actor_id: int) -> str:
    return f"{base_uid}:rollout:{rollout_index}:turn:{turn_index}:actor:{actor_id}"


def expand_episode_training_rows(
    *,
    episode_rollouts: list[dict[str, Any]],
    root_rows: list[dict[str, Any]],
    prompt_length: int,
    response_length: int,
    pad_token_id: int,
) -> tuple[dict[str, torch.Tensor], dict[str, np.ndarray], dict[str, Any]]:
    if len(episode_rollouts) != len(root_rows):
        raise ValueError("episode_rollouts and root_rows must have identical lengths.")

    prompt_tensors: list[torch.Tensor] = []
    response_tensors: list[torch.Tensor] = []
    response_mask_tensors: list[torch.Tensor] = []
    input_tensors: list[torch.Tensor] = []
    attention_tensors: list[torch.Tensor] = []
    position_tensors: list[torch.Tensor] = []
    reward_tensors: list[torch.Tensor] = []
    rollout_logprob_tensors: list[torch.Tensor] = []
    has_any_logprobs = False

    non_tensors: dict[str, list[Any]] = {}
    root_keys = set()
    for row in root_rows:
        root_keys.update(row.keys())
    managed_keys = {
        "uid",
        "raw_prompt",
        "turn_scores",
        "tool_rewards",
        "episode_steps",
        "episode_step_count",
        "step_turn_index",
        "step_actor_id",
        "step_normalized_action",
        "multi_modal_inputs",
        "__num_turns__",
    }
    for key in root_keys:
        non_tensors[key] = []

    non_tensors["uid"] = []
    non_tensors["raw_prompt"] = []
    non_tensors["turn_scores"] = []
    non_tensors["tool_rewards"] = []
    non_tensors["episode_steps"] = []
    non_tensors["episode_step_count"] = []
    non_tensors["step_turn_index"] = []
    non_tensors["step_actor_id"] = []
    non_tensors["step_normalized_action"] = []
    non_tensors["multi_modal_inputs"] = []
    non_tensors["__num_turns__"] = []

    for rollout_index, (episode_rollout, root_row) in enumerate(zip(episode_rollouts, root_rows, strict=True)):
        steps = episode_rollout.get("episode_step_data") or []
        if not steps:
            raise RuntimeError("Episode rollout did not contain any step data.")
        terminal_reward = float(episode_rollout.get("terminal_reward", 0.0))
        base_uid = str(root_row.get("uid", f"episode-{rollout_index}"))
        step_count = len(steps)

        for step in steps:
            prompt_ids = _as_int_list(step.get("prompt_ids"))
            response_ids = _as_int_list(step.get("response_ids"))
            if not response_ids:
                raise RuntimeError("Episode rollout step did not contain a model response.")
            response_logprobs = _as_float_list(step.get("response_logprobs"))
            actor_id = int(step.get("actor_id", 0))
            turn_index = int(step.get("turn_index", 0))

            prompt_tensor, prompt_attention = _left_pad(
                prompt_ids,
                pad_value=pad_token_id,
                target_length=prompt_length,
            )
            response_tensor, response_attention = _right_pad(
                response_ids,
                pad_value=pad_token_id,
                target_length=response_length,
            )
            response_mask = response_attention.clone()
            input_ids = torch.cat([prompt_tensor, response_tensor], dim=0)
            attention_mask = torch.cat([prompt_attention, response_attention], dim=0)
            position_ids = _compute_position_ids(attention_mask)
            reward_tensor = torch.zeros(response_length, dtype=torch.float32)
            last_response_index = int(response_attention.sum().item()) - 1
            reward_tensor[last_response_index] = terminal_reward

            prompt_tensors.append(prompt_tensor)
            response_tensors.append(response_tensor)
            response_mask_tensors.append(response_mask)
            input_tensors.append(input_ids)
            attention_tensors.append(attention_mask)
            position_tensors.append(position_ids)
            reward_tensors.append(reward_tensor)

            if response_logprobs is not None:
                has_any_logprobs = True
                rollout_logprob_tensors.append(
                    torch.tensor(
                        response_logprobs[:response_length] + [0.0] * max(0, response_length - len(response_logprobs)),
                        dtype=torch.float32,
                    )
                )
            else:
                rollout_logprob_tensors.append(torch.zeros(response_length, dtype=torch.float32))

            for key in root_keys - managed_keys:
                non_tensors[key].append(root_row.get(key))
            non_tensors["uid"].append(
                _build_step_uid(
                    base_uid,
                    rollout_index=rollout_index,
                    turn_index=turn_index,
                    actor_id=actor_id,
                )
            )
            non_tensors["raw_prompt"].append(step.get("messages", root_row.get("raw_prompt")))
            non_tensors["turn_scores"].append([terminal_reward])
            non_tensors["tool_rewards"].append([])
            non_tensors["episode_steps"].append(step_count)
            non_tensors["episode_step_count"].append(step_count)
            non_tensors["step_turn_index"].append(turn_index)
            non_tensors["step_actor_id"].append(actor_id)
            non_tensors["step_normalized_action"].append(str(step.get("normalized_action", "")))
            non_tensors["multi_modal_inputs"].append({})
            non_tensors["__num_turns__"].append(2)

    tensors: dict[str, torch.Tensor] = {
        "prompts": torch.stack(prompt_tensors, dim=0),
        "responses": torch.stack(response_tensors, dim=0),
        "response_mask": torch.stack(response_mask_tensors, dim=0),
        "input_ids": torch.stack(input_tensors, dim=0),
        "attention_mask": torch.stack(attention_tensors, dim=0),
        "position_ids": torch.stack(position_tensors, dim=0),
        "rm_scores": torch.stack(reward_tensors, dim=0),
    }
    if has_any_logprobs:
        tensors["rollout_log_probs"] = torch.stack(rollout_logprob_tensors, dim=0)

    array_batch = {
        key: np.array(values, dtype=object)
        for key, values in non_tensors.items()
    }
    array_batch["__num_turns__"] = np.array(array_batch["__num_turns__"], dtype=np.int32)
    array_batch["episode_steps"] = np.array(array_batch["episode_steps"], dtype=np.int32)
    array_batch["episode_step_count"] = np.array(array_batch["episode_step_count"], dtype=np.int32)
    array_batch["step_turn_index"] = np.array(array_batch["step_turn_index"], dtype=np.int32)
    array_batch["step_actor_id"] = np.array(array_batch["step_actor_id"], dtype=np.int32)

    return tensors, array_batch, {"reward_extra_keys": []}


__all__ = ["expand_episode_training_rows"]
