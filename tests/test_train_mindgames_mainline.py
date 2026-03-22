import json
import subprocess
import sys
import unittest
from pathlib import Path


def _ensure_pkg_importable() -> None:
    project_root = Path(__file__).resolve().parents[1]
    repo_root = project_root.parent
    sys.path.insert(0, str(project_root))
    sys.path.insert(0, str(repo_root))


_ensure_pkg_importable()


class TestVerlMainline(unittest.TestCase):
    def test_mini_hanabi_ppo_preset_targets_1024_training_steps(self):
        from mindgames.training.presets import get_training_preset

        preset = get_training_preset("mini_hanabi_ppo")

        self.assertEqual(preset.cli_defaults["train_size"], 8192)
        self.assertEqual(preset.cli_defaults["train_batch_size"], 8)
        self.assertEqual(preset.cli_defaults["val_size"], 32)
        self.assertEqual(preset.cli_defaults["save_freq"], 100)
        self.assertEqual(preset.cli_defaults["test_freq"], 100)
        self.assertFalse(preset.cli_defaults["param_offload"])
        self.assertFalse(preset.cli_defaults["optimizer_offload"])
        self.assertFalse(preset.cli_defaults["ref_param_offload"])
        self.assertEqual(8192 // 8, 1024)

    def test_train_cli_dry_run_prints_resolved_config(self):
        repo_root = Path(__file__).resolve().parents[1]
        cmd = [
            sys.executable,
            str(repo_root / "tools" / "train" / "train_mindgames_verl.py"),
            "--game",
            "colonel_blotto",
            "--train-size",
            "4",
            "--val-size",
            "2",
            "--dry-run",
            "--print-config",
        ]

        proc = subprocess.run(cmd, cwd=str(repo_root), capture_output=True, text=True)
        self.assertEqual(proc.returncode, 0, msg=f"stderr:\n{proc.stderr}\nstdout:\n{proc.stdout}")
        payload = json.loads(proc.stdout)
        self.assertEqual(payload["game"], "colonel_blotto")
        self.assertEqual(payload["env_id"], "ColonelBlotto-v0-train")
        self.assertEqual(payload["train_size"], 4)
        self.assertEqual(payload["val_size"], 2)
        self.assertEqual(payload["adv_estimator"], "gae")
        self.assertEqual(payload["finetune_mode"], "full")
        self.assertTrue(payload["critic_enabled"])
        self.assertFalse(payload["lora_enabled"])
        self.assertEqual(payload["reward_player"], 0)
        self.assertEqual(
            payload["agent_loop_name"],
            "mindgames_snapshot_episode",
        )
        self.assertEqual(
            payload["agent_loop_class"],
            "mindgames.training.verl_snapshot_agent_loop.MindGamesSnapshotEpisodeAgentLoop",
        )
        self.assertTrue(payload["agent_loop_config"].endswith("agent_loop.yaml"))
        checkpoint_dir = Path(payload["checkpoint_dir"])
        self.assertEqual(checkpoint_dir.parent, repo_root / "checkpoints" / "mindgames-verl")
        self.assertTrue(checkpoint_dir.name.startswith("colonel_blotto-verl-"))
        self.assertIn(f"trainer.default_local_dir={payload['checkpoint_dir']}", payload["overrides"])
        self.assertIn("trainer.logger=[console]", payload["overrides"])
        self.assertIn('++ray_kwargs.ray_init.runtime_env.env_vars.NCCL_IB_DISABLE="1"', payload["overrides"])
        self.assertIn('++ray_kwargs.ray_init.runtime_env.env_vars.NCCL_P2P_DISABLE="1"', payload["overrides"])

    def test_train_cli_rejects_grpo_for_snapshot_episode_training(self):
        repo_root = Path(__file__).resolve().parents[1]
        cmd = [
            sys.executable,
            str(repo_root / "tools" / "train" / "train_mindgames_verl.py"),
            "--game",
            "mini_hanabi",
            "--adv-estimator",
            "grpo",
            "--dry-run",
        ]

        proc = subprocess.run(cmd, cwd=str(repo_root), capture_output=True, text=True)
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("does not support GRPO", proc.stderr)

    def test_train_cli_supports_gae_for_ppo(self):
        repo_root = Path(__file__).resolve().parents[1]
        model_path = "/workspace/models/Qwen3-8B"
        cmd = [
            sys.executable,
            str(repo_root / "tools" / "train" / "train_mindgames_verl.py"),
            "--game",
            "mini_hanabi",
            "--train-size",
            "2",
            "--val-size",
            "1",
            "--adv-estimator",
            "gae",
            "--model",
            model_path,
            "--dry-run",
            "--print-config",
        ]

        proc = subprocess.run(cmd, cwd=str(repo_root), capture_output=True, text=True)
        self.assertEqual(proc.returncode, 0, msg=f"stderr:\n{proc.stderr}\nstdout:\n{proc.stdout}")
        payload = json.loads(proc.stdout)
        self.assertEqual(payload["game"], "mini_hanabi")
        self.assertEqual(payload["adv_estimator"], "gae")
        self.assertEqual(payload["finetune_mode"], "full")
        self.assertTrue(payload["critic_enabled"])
        self.assertIn(f"trainer.default_local_dir={payload['checkpoint_dir']}", payload["overrides"])
        self.assertIn("trainer.logger=[console]", payload["overrides"])
        self.assertIn(f"critic.model.path={model_path}", payload["overrides"])
        self.assertIn("critic.model.fsdp_config.param_offload=True", payload["overrides"])
        self.assertIn("critic.model.fsdp_config.optimizer_offload=True", payload["overrides"])
        self.assertIn('++ray_kwargs.ray_init.runtime_env.env_vars.NCCL_IB_DISABLE="1"', payload["overrides"])
        self.assertIn('++ray_kwargs.ray_init.runtime_env.env_vars.NCCL_P2P_DISABLE="1"', payload["overrides"])
        self.assertNotIn("algorithm.adv_estimator=gae", payload["overrides"])
        self.assertNotIn("critic.enable=True", payload["overrides"])
        self.assertNotIn("actor_rollout_ref.actor.strategy=fsdp2", payload["overrides"])
        self.assertNotIn("actor_rollout_ref.ref.strategy=fsdp2", payload["overrides"])
        self.assertNotIn("critic.strategy=fsdp2", payload["overrides"])
        self.assertNotIn("actor_rollout_ref.actor.fsdp_config.model_dtype=bf16", payload["overrides"])
        self.assertNotIn("actor_rollout_ref.ref.fsdp_config.model_dtype=bf16", payload["overrides"])
        self.assertNotIn("critic.model.fsdp_config.model_dtype=bf16", payload["overrides"])
        self.assertNotIn("actor_rollout_ref.actor.use_torch_compile=False", payload["overrides"])
        self.assertNotIn("actor_rollout_ref.ref.use_torch_compile=False", payload["overrides"])
        self.assertNotIn("critic.model.fsdp_config.use_torch_compile=False", payload["overrides"])
        self.assertNotIn("critic.model.use_remove_padding=True", payload["overrides"])
        self.assertNotIn("+critic.model.override_config.attn_implementation=eager", payload["overrides"])
        self.assertNotIn("actor_rollout_ref.actor.optim.lr=", "\n".join(payload["overrides"]))
        self.assertNotIn("critic.optim.lr=", "\n".join(payload["overrides"]))
        self.assertNotIn("actor_rollout_ref.model.lora_rank=32", payload["overrides"])

    def test_train_cli_preset_exposes_mini_hanabi_ppo_starter(self):
        repo_root = Path(__file__).resolve().parents[1]
        cmd = [
            sys.executable,
            str(repo_root / "tools" / "train" / "train_mindgames_verl.py"),
            "--preset",
            "mini_hanabi_ppo",
            "--train-size",
            "2",
            "--val-size",
            "1",
            "--dry-run",
            "--print-config",
        ]

        proc = subprocess.run(cmd, cwd=str(repo_root), capture_output=True, text=True)
        self.assertEqual(proc.returncode, 0, msg=f"stderr:\n{proc.stderr}\nstdout:\n{proc.stdout}")
        payload = json.loads(proc.stdout)
        self.assertEqual(payload["preset"], "mini_hanabi_ppo")
        self.assertEqual(payload["game"], "mini_hanabi")
        self.assertEqual(payload["adv_estimator"], "gae")
        self.assertEqual(payload["finetune_mode"], "lora")
        self.assertTrue(payload["critic_enabled"])
        self.assertTrue(payload["lora_enabled"])
        self.assertEqual(payload["lora_rank"], 32)
        self.assertEqual(payload["lora_alpha"], 64)
        self.assertTrue(payload["rollout_enable_sleep_mode"])
        self.assertIsNone(payload["rollout_enforce_eager"])
        self.assertEqual(
            payload["lora_target_modules"],
            [
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
        )
        self.assertEqual(payload["reward_player"], -1)
        self.assertIn(f"trainer.default_local_dir={payload['checkpoint_dir']}", payload["overrides"])
        self.assertIn("trainer.logger=[console]", payload["overrides"])
        self.assertIn("trainer.n_gpus_per_node=4", payload["overrides"])
        self.assertIn("trainer.save_freq=100", payload["overrides"])
        self.assertIn("trainer.test_freq=100", payload["overrides"])
        self.assertIn("actor_rollout_ref.rollout.max_model_len=1280", payload["overrides"])
        self.assertIn("actor_rollout_ref.rollout.max_num_batched_tokens=1280", payload["overrides"])
        self.assertIn("+actor_rollout_ref.rollout.enable_sleep_mode=True", payload["overrides"])
        self.assertIn("actor_rollout_ref.rollout.gpu_memory_utilization=0.9", payload["overrides"])
        self.assertIn("actor_rollout_ref.actor.fsdp_config.param_offload=False", payload["overrides"])
        self.assertIn("actor_rollout_ref.actor.fsdp_config.optimizer_offload=False", payload["overrides"])
        self.assertIn("actor_rollout_ref.ref.fsdp_config.param_offload=False", payload["overrides"])
        self.assertNotIn("data.truncation=error", payload["overrides"])
        self.assertNotIn("algorithm.use_kl_in_reward=False", payload["overrides"])
        self.assertNotIn("actor_rollout_ref.rollout.mode=async", payload["overrides"])
        self.assertNotIn("actor_rollout_ref.actor.use_kl_loss=False", payload["overrides"])
        self.assertNotIn("critic.model.tokenizer_path=/workspace/models/Qwen3-8B", payload["overrides"])
        self.assertNotIn("actor_rollout_ref.actor.optim.lr=", "\n".join(payload["overrides"]))
        self.assertNotIn("critic.optim.lr=", "\n".join(payload["overrides"]))
        self.assertNotIn("algorithm.adv_estimator=gae", payload["overrides"])
        self.assertNotIn("critic.enable=True", payload["overrides"])
        self.assertIn("actor_rollout_ref.model.lora_rank=32", payload["overrides"])
        self.assertIn("actor_rollout_ref.model.lora_alpha=64", payload["overrides"])
        self.assertIn(
            'actor_rollout_ref.model.target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]',
            payload["overrides"],
        )
        self.assertIn("actor_rollout_ref.rollout.load_format=safetensors", payload["overrides"])
        self.assertIn("critic.model.lora_rank=32", payload["overrides"])
        self.assertIn("critic.model.lora_alpha=64", payload["overrides"])
        self.assertIn(
            'critic.model.target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]',
            payload["overrides"],
        )
        self.assertIn("critic.model.fsdp_config.param_offload=False", payload["overrides"])
        self.assertIn("critic.model.fsdp_config.optimizer_offload=False", payload["overrides"])
        self.assertIn('++ray_kwargs.ray_init.runtime_env.env_vars.NCCL_IB_DISABLE="1"', payload["overrides"])
        self.assertIn('++ray_kwargs.ray_init.runtime_env.env_vars.NCCL_P2P_DISABLE="1"', payload["overrides"])
        self.assertNotIn("actor_rollout_ref.actor.strategy=fsdp2", payload["overrides"])
        self.assertNotIn("actor_rollout_ref.ref.strategy=fsdp2", payload["overrides"])
        self.assertNotIn("critic.strategy=fsdp2", payload["overrides"])
        self.assertNotIn("actor_rollout_ref.actor.fsdp_config.model_dtype=bf16", payload["overrides"])
        self.assertNotIn("actor_rollout_ref.ref.fsdp_config.model_dtype=bf16", payload["overrides"])
        self.assertNotIn("critic.model.fsdp_config.model_dtype=bf16", payload["overrides"])
        self.assertNotIn("actor_rollout_ref.actor.use_torch_compile=False", payload["overrides"])
        self.assertNotIn("actor_rollout_ref.ref.use_torch_compile=False", payload["overrides"])
        self.assertNotIn("critic.model.fsdp_config.use_torch_compile=False", payload["overrides"])
        self.assertNotIn("critic.model.use_remove_padding=True", payload["overrides"])
        self.assertNotIn("+critic.model.override_config.attn_implementation=eager", payload["overrides"])

    def test_train_cli_wandb_enables_wandb_logger(self):
        repo_root = Path(__file__).resolve().parents[1]
        cmd = [
            sys.executable,
            str(repo_root / "tools" / "train" / "train_mindgames_verl.py"),
            "--game",
            "mini_hanabi",
            "--train-size",
            "2",
            "--val-size",
            "1",
            "--adv-estimator",
            "gae",
            "--wandb",
            "--dry-run",
            "--print-config",
        ]

        proc = subprocess.run(cmd, cwd=str(repo_root), capture_output=True, text=True)
        self.assertEqual(proc.returncode, 0, msg=f"stderr:\n{proc.stderr}\nstdout:\n{proc.stdout}")
        payload = json.loads(proc.stdout)
        self.assertIn("trainer.logger=[console,wandb]", payload["overrides"])


    def test_train_cli_disable_thinking_sets_chat_template_override(self):
        repo_root = Path(__file__).resolve().parents[1]
        cmd = [
            sys.executable,
            str(repo_root / "tools" / "train" / "train_mindgames_verl.py"),
            "--game",
            "mini_hanabi",
            "--train-size",
            "2",
            "--val-size",
            "1",
            "--disable-thinking",
            "--dry-run",
            "--print-config",
        ]

        proc = subprocess.run(cmd, cwd=str(repo_root), capture_output=True, text=True)
        self.assertEqual(proc.returncode, 0, msg=f"stderr:\n{proc.stderr}\nstdout:\n{proc.stdout}")
        payload = json.loads(proc.stdout)
        self.assertFalse(payload["enable_thinking"])
        self.assertIn(
            '++data.apply_chat_template_kwargs.enable_thinking=false',
            payload["overrides"],
        )


if __name__ == "__main__":
    unittest.main()
