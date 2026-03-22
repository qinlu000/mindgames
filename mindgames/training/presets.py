from __future__ import annotations

from dataclasses import dataclass
from typing import Any


QWEN_LORA_TARGET_MODULES = "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"


@dataclass(frozen=True)
class TrainingPreset:
    name: str
    description: str
    cli_defaults: dict[str, Any]


TRAINING_PRESETS: dict[str, TrainingPreset] = {
    "mini_hanabi_ppo": TrainingPreset(
        name="mini_hanabi_ppo",
        description="Four-GPU LoRA preset for MiniHanabi PPO with a critic (VERL GAE path, 1024 PPO steps, GPU-resident training).",
        cli_defaults={
            "game": "mini_hanabi",
            "adv_estimator": "gae",
            "finetune_mode": "lora",
            "lora_rank": 32,
            "lora_alpha": 64,
            "lora_target_modules": QWEN_LORA_TARGET_MODULES,
            "gpu_memory_utilization": 0.9,
            "rollout_enable_sleep_mode": True,
            "max_steps": 28,
            "reward_player": -1,
            "train_size": 8192,
            "val_size": 32,
            "train_batch_size": 8,
            "rollout_response_length": 256,
            "rollout_max_model_len": 1280,
            "rollout_max_num_batched_tokens": 1280,
            "rollout_n": 1,
            "ppo_mini_batch_size": 8,
            "ppo_micro_batch_size_per_gpu": 1,
            "critic_ppo_micro_batch_size_per_gpu": 1,
            "param_offload": False,
            "optimizer_offload": False,
            "ref_param_offload": False,
            "n_gpus_per_node": 4,
            "save_freq": 100,
            "test_freq": 100,
        },
    ),
}


def get_training_preset(name: str) -> TrainingPreset:
    return TRAINING_PRESETS[name]


def list_training_presets() -> list[TrainingPreset]:
    return [TRAINING_PRESETS[name] for name in sorted(TRAINING_PRESETS)]
