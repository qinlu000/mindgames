import argparse
import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch


def _ensure_pkg_importable() -> None:
    project_root = Path(__file__).resolve().parents[1]
    repo_root = project_root.parent
    sys.path.insert(0, str(project_root))
    sys.path.insert(0, str(repo_root))


_ensure_pkg_importable()


class TestTrainingVerlLaunch(unittest.TestCase):
    def test_prepare_run_plan_returns_thin_plan_object(self):
        from mindgames.training.verl_launch import (
            SNAPSHOT_AGENT_LOOP_CLASS,
            SNAPSHOT_AGENT_LOOP_NAME,
            TRAINER_MAIN_MODULE,
            VerlLaunchConfig,
            prepare_run_plan,
        )

        repo_root = Path(__file__).resolve().parents[1]
        args = argparse.Namespace(
            game="mini_hanabi",
            env_id=None,
            model="/workspace/models/Qwen3-8B",
            train_size=2,
            val_size=1,
            train_seed_start=0,
            val_seed_start=100000,
            max_steps=None,
            reward_player=None,
            dry_run=True,
            print_config=True,
            train_batch_size=16,
            max_prompt_length=1024,
            rollout_prompt_length=1024,
            rollout_response_length=5120,
            rollout_n=2,
            tensor_model_parallel_size=1,
            gpu_memory_utilization=0.45,
            rollout_max_model_len=None,
            rollout_max_num_batched_tokens=6144,
            rollout_max_num_seqs=1,
            rollout_enable_sleep_mode=None,
            rollout_enforce_eager=None,
            adv_estimator="gae",
            finetune_mode="full",
            lora_rank=32,
            lora_alpha=64,
            lora_target_modules="all-linear",
            lora_adapter_path=None,
            ppo_mini_batch_size=16,
            ppo_micro_batch_size_per_gpu=1,
            critic_ppo_micro_batch_size_per_gpu=1,
            log_prob_micro_batch_size_per_gpu=1,
            ref_log_prob_micro_batch_size_per_gpu=1,
            learning_rate=None,
            critic_learning_rate=None,
            entropy_coeff=1e-3,
            n_gpus_per_node=4,
            total_epochs=1,
            test_freq=1000,
            save_freq=1000,
            project_name="mindgames-verl",
            experiment_name=None,
            wandb=False,
            val_before_train=True,
            gradient_checkpointing=True,
            param_offload=True,
            optimizer_offload=True,
            ref_param_offload=True,
            preset=None,
        )

        config = VerlLaunchConfig.from_namespace(args)
        plan = prepare_run_plan(config, root_dir=repo_root, repo_parent=repo_root.parent)

        self.assertEqual(plan.config.game, "mini_hanabi")
        self.assertEqual(plan.config.env_id, "MiniHanabi-v0-train")
        self.assertEqual(plan.agent_loop_name, SNAPSHOT_AGENT_LOOP_NAME)
        self.assertEqual(plan.agent_loop_class, SNAPSHOT_AGENT_LOOP_CLASS)
        self.assertEqual(TRAINER_MAIN_MODULE, "mindgames.training.verl_main_ppo")
        self.assertEqual(len(plan.train_rows), 2)
        self.assertEqual(len(plan.val_rows), 1)
        payload = plan.to_payload()
        self.assertIsNone(payload["preset"])
        self.assertEqual(payload["agent_loop_name"], SNAPSHOT_AGENT_LOOP_NAME)
        self.assertEqual(payload["agent_loop_class"], SNAPSHOT_AGENT_LOOP_CLASS)
        self.assertEqual(payload["finetune_mode"], "full")
        self.assertFalse(payload["lora_enabled"])
        self.assertEqual(
            payload["checkpoint_dir"],
            str(repo_root / "checkpoints" / "mindgames-verl" / plan.config.experiment_name),
        )
        self.assertIn("actor_rollout_ref.rollout.multi_turn.enable=False", payload["overrides"])
        self.assertIn(
            f"actor_rollout_ref.rollout.agent.default_agent_loop={SNAPSHOT_AGENT_LOOP_NAME}",
            payload["overrides"],
        )
        self.assertIn(
            f"trainer.default_local_dir={repo_root / 'checkpoints' / 'mindgames-verl' / plan.config.experiment_name}",
            payload["overrides"],
        )
        self.assertIn("trainer.logger=[console]", payload["overrides"])
        self.assertIsNone(payload["enable_thinking"])
        self.assertIn('++ray_kwargs.ray_init.runtime_env.env_vars.NCCL_IB_DISABLE="1"', payload["overrides"])
        self.assertIn('++ray_kwargs.ray_init.runtime_env.env_vars.NCCL_P2P_DISABLE="1"', payload["overrides"])
        self.assertNotIn("data.apply_chat_template_kwargs=", "\n".join(payload["overrides"]))
        self.assertNotIn("actor_rollout_ref.actor.optim.lr=", "\n".join(payload["overrides"]))
        self.assertNotIn("critic.optim.lr=", "\n".join(payload["overrides"]))
        self.assertNotIn("actor_rollout_ref.rollout.load_format=safetensors", payload["overrides"])
        self.assertTrue(payload["agent_loop_config"].endswith("agent_loop.yaml"))

    def test_prepare_run_plan_emits_enable_thinking_override(self):
        from mindgames.training.verl_launch import VerlLaunchConfig, prepare_run_plan

        repo_root = Path(__file__).resolve().parents[1]
        args = argparse.Namespace(
            game="mini_hanabi",
            env_id=None,
            model="/workspace/models/Qwen3-8B",
            train_size=1,
            val_size=1,
            train_seed_start=0,
            val_seed_start=100000,
            max_steps=None,
            reward_player=None,
            dry_run=True,
            print_config=True,
            train_batch_size=8,
            max_prompt_length=1024,
            rollout_prompt_length=1024,
            rollout_response_length=512,
            rollout_n=1,
            tensor_model_parallel_size=1,
            gpu_memory_utilization=None,
            rollout_max_model_len=None,
            rollout_max_num_batched_tokens=1536,
            rollout_max_num_seqs=1,
            rollout_enable_sleep_mode=None,
            rollout_enforce_eager=None,
            adv_estimator="gae",
            finetune_mode="full",
            lora_rank=32,
            lora_alpha=64,
            lora_target_modules="all-linear",
            lora_adapter_path=None,
            ppo_mini_batch_size=8,
            ppo_micro_batch_size_per_gpu=1,
            critic_ppo_micro_batch_size_per_gpu=1,
            log_prob_micro_batch_size_per_gpu=1,
            ref_log_prob_micro_batch_size_per_gpu=1,
            learning_rate=None,
            critic_learning_rate=None,
            entropy_coeff=1e-3,
            n_gpus_per_node=4,
            total_epochs=1,
            test_freq=1000,
            save_freq=1000,
            project_name="mindgames-verl",
            experiment_name=None,
            wandb=False,
            val_before_train=True,
            gradient_checkpointing=True,
            param_offload=True,
            optimizer_offload=True,
            ref_param_offload=True,
            enable_thinking=False,
            preset=None,
        )

        config = VerlLaunchConfig.from_namespace(args)
        plan = prepare_run_plan(config, root_dir=repo_root, repo_parent=repo_root.parent)

        self.assertFalse(plan.to_payload()["enable_thinking"])
        self.assertIn(
            '++data.apply_chat_template_kwargs.enable_thinking=false',
            plan.overrides,
        )

    def test_snapshot_episode_training_rejects_grpo(self):
        from mindgames.training.verl_launch import VerlLaunchConfig

        args = argparse.Namespace(
            game="mini_hanabi",
            env_id=None,
            model="/workspace/models/Qwen3-8B",
            train_size=1,
            val_size=1,
            train_seed_start=0,
            val_seed_start=100000,
            max_steps=None,
            reward_player=None,
            dry_run=True,
            print_config=True,
            train_batch_size=8,
            max_prompt_length=1024,
            rollout_prompt_length=1024,
            rollout_response_length=512,
            rollout_n=2,
            tensor_model_parallel_size=1,
            gpu_memory_utilization=None,
            rollout_max_model_len=None,
            rollout_max_num_batched_tokens=1536,
            rollout_max_num_seqs=1,
            rollout_enable_sleep_mode=None,
            rollout_enforce_eager=None,
            adv_estimator="grpo",
            finetune_mode="full",
            lora_rank=32,
            lora_alpha=64,
            lora_target_modules="all-linear",
            lora_adapter_path=None,
            ppo_mini_batch_size=8,
            ppo_micro_batch_size_per_gpu=1,
            critic_ppo_micro_batch_size_per_gpu=1,
            log_prob_micro_batch_size_per_gpu=1,
            ref_log_prob_micro_batch_size_per_gpu=1,
            learning_rate=None,
            critic_learning_rate=None,
            entropy_coeff=1e-3,
            n_gpus_per_node=4,
            total_epochs=1,
            test_freq=1000,
            save_freq=1000,
            project_name="mindgames-verl",
            experiment_name=None,
            wandb=False,
            val_before_train=True,
            gradient_checkpointing=True,
            param_offload=True,
            optimizer_offload=True,
            ref_param_offload=True,
            enable_thinking=None,
            preset=None,
        )

        with self.assertRaisesRegex(ValueError, "does not support GRPO"):
            VerlLaunchConfig.from_namespace(args)

    def test_redact_sensitive_override_masks_wandb_api_key(self):
        from mindgames.training.verl_launch import redact_sensitive_override

        self.assertEqual(
            redact_sensitive_override(
                '++ray_kwargs.ray_init.runtime_env.env_vars.WANDB_API_KEY="secret-key"'
            ),
            '++ray_kwargs.ray_init.runtime_env.env_vars.WANDB_API_KEY="<redacted>"',
        )
        self.assertEqual(
            redact_sensitive_override(
                '++ray_kwargs.ray_init.runtime_env.env_vars.WANDB_ENTITY="dummy-entity"'
            ),
            '++ray_kwargs.ray_init.runtime_env.env_vars.WANDB_ENTITY="dummy-entity"',
        )

    def test_launch_verl_invokes_main_ppo_with_runtime_pythonpath(self):
        from mindgames.training.verl_launch import VerlLaunchConfig, launch_verl, prepare_run_plan

        repo_root = Path(__file__).resolve().parents[1]
        config = VerlLaunchConfig(
            preset="mini_hanabi_ppo",
            game="mini_hanabi",
            env_id="MiniHanabi-v0-train",
            model="/workspace/models/Qwen3-8B",
            train_size=1,
            val_size=1,
            train_seed_start=0,
            val_seed_start=100000,
            max_steps=12,
            reward_player=-1,
            dry_run=False,
            print_config=False,
            train_batch_size=8,
            max_prompt_length=1024,
            rollout_prompt_length=1024,
            rollout_response_length=5120,
            rollout_n=1,
            tensor_model_parallel_size=1,
            gpu_memory_utilization=0.45,
            rollout_max_model_len=None,
            rollout_max_num_batched_tokens=6144,
            rollout_max_num_seqs=1,
            rollout_enable_sleep_mode=False,
            rollout_enforce_eager=True,
            adv_estimator="gae",
            finetune_mode="lora",
            lora_rank=32,
            lora_alpha=64,
            lora_target_modules="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
            lora_adapter_path=None,
            ppo_mini_batch_size=8,
            ppo_micro_batch_size_per_gpu=1,
            critic_ppo_micro_batch_size_per_gpu=1,
            log_prob_micro_batch_size_per_gpu=1,
            ref_log_prob_micro_batch_size_per_gpu=1,
            learning_rate=None,
            critic_learning_rate=None,
            entropy_coeff=1e-3,
            n_gpus_per_node=1,
            total_epochs=1,
            test_freq=1000,
            save_freq=1000,
            project_name="mindgames-verl",
            experiment_name="mini-hanabi-ppo-smoke",
            wandb=False,
            val_before_train=True,
            gradient_checkpointing=True,
            param_offload=True,
            optimizer_offload=True,
            ref_param_offload=True,
        )
        plan = prepare_run_plan(config, root_dir=repo_root, repo_parent=repo_root.parent)

        with patch("mindgames.training.verl_launch.subprocess.run") as run_mock:
            launch_verl(plan, root_dir=repo_root, repo_parent=repo_root.parent)

        run_mock.assert_called_once()
        command = run_mock.call_args.args[0]
        kwargs = run_mock.call_args.kwargs
        self.assertEqual(command[:3], [sys.executable, "-m", "mindgames.training.verl_main_ppo"])
        self.assertIn("actor_rollout_ref.model.lora_rank=32", command)
        self.assertIn("actor_rollout_ref.rollout.load_format=safetensors", command)
        self.assertIn("+actor_rollout_ref.rollout.enable_sleep_mode=False", command)
        self.assertIn("actor_rollout_ref.rollout.enforce_eager=True", command)
        self.assertIn(
            'actor_rollout_ref.model.target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]',
            command,
        )
        self.assertIn("critic.model.lora_rank=32", command)
        self.assertIn("critic.model.fsdp_config.param_offload=True", command)
        self.assertIn("critic.model.fsdp_config.optimizer_offload=True", command)
        self.assertIn(
            f"trainer.default_local_dir={repo_root / 'checkpoints' / 'mindgames-verl' / 'mini-hanabi-ppo-smoke'}",
            command,
        )
        self.assertIn('++ray_kwargs.ray_init.runtime_env.env_vars.NCCL_IB_DISABLE="1"', command)
        self.assertIn('++ray_kwargs.ray_init.runtime_env.env_vars.NCCL_P2P_DISABLE="1"', command)
        self.assertIn(
            'critic.model.target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]',
            command,
        )
        self.assertNotIn("algorithm.adv_estimator=gae", command)
        self.assertNotIn("critic.enable=True", command)
        self.assertNotIn("actor_rollout_ref.actor.strategy=fsdp2", command)
        self.assertNotIn("actor_rollout_ref.ref.strategy=fsdp2", command)
        self.assertNotIn("critic.strategy=fsdp2", command)
        self.assertNotIn("actor_rollout_ref.actor.fsdp_config.model_dtype=bf16", command)
        self.assertNotIn("critic.model.fsdp_config.model_dtype=bf16", command)
        self.assertNotIn("actor_rollout_ref.actor.use_torch_compile=False", command)
        self.assertNotIn("actor_rollout_ref.ref.use_torch_compile=False", command)
        self.assertNotIn("critic.model.fsdp_config.use_torch_compile=False", command)
        self.assertNotIn("critic.model.use_remove_padding=True", command)
        self.assertNotIn("+critic.model.override_config.attn_implementation=eager", command)
        self.assertNotIn("actor_rollout_ref.actor.optim.lr=", "\n".join(command))
        self.assertNotIn("critic.optim.lr=", "\n".join(command))
        self.assertEqual(kwargs["cwd"], str(repo_root))
        self.assertTrue(kwargs["check"])
        self.assertIn(str(repo_root), kwargs["env"]["PYTHONPATH"])
        self.assertIn(str(repo_root.parent), kwargs["env"]["PYTHONPATH"])
        self.assertEqual(kwargs["env"]["NCCL_IB_DISABLE"], "1")
        self.assertEqual(kwargs["env"]["NCCL_P2P_DISABLE"], "1")

    def test_launch_verl_passes_wandb_logger_and_runtime_env(self):
        from mindgames.training.verl_launch import VerlLaunchConfig, launch_verl, prepare_run_plan

        repo_root = Path(__file__).resolve().parents[1]
        config = VerlLaunchConfig(
            preset=None,
            game="mini_hanabi",
            env_id="MiniHanabi-v0-train",
            model="/workspace/models/Qwen3-8B",
            train_size=1,
            val_size=1,
            train_seed_start=0,
            val_seed_start=100000,
            max_steps=12,
            reward_player=-1,
            dry_run=False,
            print_config=False,
            train_batch_size=8,
            max_prompt_length=1024,
            rollout_prompt_length=1024,
            rollout_response_length=512,
            rollout_n=1,
            tensor_model_parallel_size=1,
            gpu_memory_utilization=None,
            rollout_max_model_len=None,
            rollout_max_num_batched_tokens=1280,
            rollout_max_num_seqs=1,
            rollout_enable_sleep_mode=True,
            rollout_enforce_eager=None,
            adv_estimator="gae",
            finetune_mode="full",
            lora_rank=32,
            lora_alpha=64,
            lora_target_modules="all-linear",
            lora_adapter_path=None,
            ppo_mini_batch_size=8,
            ppo_micro_batch_size_per_gpu=1,
            critic_ppo_micro_batch_size_per_gpu=1,
            log_prob_micro_batch_size_per_gpu=1,
            ref_log_prob_micro_batch_size_per_gpu=1,
            learning_rate=None,
            critic_learning_rate=None,
            entropy_coeff=1e-3,
            n_gpus_per_node=4,
            total_epochs=1,
            test_freq=1000,
            save_freq=1000,
            project_name="mindgames-verl",
            experiment_name="mini-hanabi-ppo-wandb",
            wandb=True,
            val_before_train=True,
            gradient_checkpointing=True,
            param_offload=True,
            optimizer_offload=True,
            ref_param_offload=True,
        )
        with (
            patch.dict(
                os.environ,
                {
                    "WANDB_API_KEY": "dummy-key",
                    "WANDB_ENTITY": "dummy-entity",
                    "WANDB_BASE_URL": "https://api.wandb.ai",
                },
                clear=False,
            ),
            patch("mindgames.training.verl_launch.subprocess.run") as run_mock,
        ):
            plan = prepare_run_plan(config, root_dir=repo_root, repo_parent=repo_root.parent)
            launch_verl(plan, root_dir=repo_root, repo_parent=repo_root.parent)

        command = run_mock.call_args.args[0]
        kwargs = run_mock.call_args.kwargs
        self.assertIn("trainer.logger=[console,wandb]", command)
        self.assertIn('++ray_kwargs.ray_init.runtime_env.env_vars.WANDB_API_KEY="dummy-key"', command)
        self.assertIn('++ray_kwargs.ray_init.runtime_env.env_vars.WANDB_ENTITY="dummy-entity"', command)
        self.assertIn('++ray_kwargs.ray_init.runtime_env.env_vars.WANDB_BASE_URL="https://api.wandb.ai"', command)
        self.assertEqual(kwargs["env"]["WANDB_API_KEY"], "dummy-key")
        self.assertEqual(kwargs["env"]["WANDB_ENTITY"], "dummy-entity")
        self.assertEqual(kwargs["env"]["WANDB_BASE_URL"], "https://api.wandb.ai")

    def test_launch_verl_loads_wandb_key_from_netrc_when_env_missing(self):
        from mindgames.training.verl_launch import VerlLaunchConfig, launch_verl, prepare_run_plan

        repo_root = Path(__file__).resolve().parents[1]
        config = VerlLaunchConfig(
            preset=None,
            game="mini_hanabi",
            env_id="MiniHanabi-v0-train",
            model="/workspace/models/Qwen3-8B",
            train_size=1,
            val_size=1,
            train_seed_start=0,
            val_seed_start=100000,
            max_steps=12,
            reward_player=-1,
            dry_run=False,
            print_config=False,
            train_batch_size=8,
            max_prompt_length=1024,
            rollout_prompt_length=1024,
            rollout_response_length=512,
            rollout_n=1,
            tensor_model_parallel_size=1,
            gpu_memory_utilization=None,
            rollout_max_model_len=None,
            rollout_max_num_batched_tokens=1280,
            rollout_max_num_seqs=1,
            rollout_enable_sleep_mode=True,
            rollout_enforce_eager=None,
            adv_estimator="gae",
            finetune_mode="full",
            lora_rank=32,
            lora_alpha=64,
            lora_target_modules="all-linear",
            lora_adapter_path=None,
            ppo_mini_batch_size=8,
            ppo_micro_batch_size_per_gpu=1,
            critic_ppo_micro_batch_size_per_gpu=1,
            log_prob_micro_batch_size_per_gpu=1,
            ref_log_prob_micro_batch_size_per_gpu=1,
            learning_rate=None,
            critic_learning_rate=None,
            entropy_coeff=1e-3,
            n_gpus_per_node=4,
            total_epochs=1,
            test_freq=1000,
            save_freq=1000,
            project_name="mindgames-verl",
            experiment_name="mini-hanabi-ppo-wandb-netrc",
            wandb=True,
            val_before_train=True,
            gradient_checkpointing=True,
            param_offload=True,
            optimizer_offload=True,
            ref_param_offload=True,
        )
        netrc_mock = unittest.mock.Mock()
        netrc_mock.authenticators.return_value = ("user", None, "netrc-key")
        with (
            patch.dict(
                os.environ,
                {"WANDB_ENTITY": "dummy-entity"},
                clear=True,
            ),
            patch("mindgames.training.verl_launch.netrc.netrc", return_value=netrc_mock),
            patch("mindgames.training.verl_launch.subprocess.run") as run_mock,
        ):
            plan = prepare_run_plan(config, root_dir=repo_root, repo_parent=repo_root.parent)
            launch_verl(plan, root_dir=repo_root, repo_parent=repo_root.parent)

        command = run_mock.call_args.args[0]
        kwargs = run_mock.call_args.kwargs
        self.assertIn('++ray_kwargs.ray_init.runtime_env.env_vars.WANDB_API_KEY="netrc-key"', command)
        self.assertEqual(kwargs["env"]["WANDB_API_KEY"], "netrc-key")
        self.assertEqual(kwargs["env"]["WANDB_ENTITY"], "dummy-entity")

    def test_prepare_run_plan_formats_custom_lora_target_modules(self):
        from mindgames.training.verl_launch import VerlLaunchConfig, prepare_run_plan

        repo_root = Path(__file__).resolve().parents[1]
        args = argparse.Namespace(
            game="mini_hanabi",
            env_id=None,
            model="/workspace/models/Qwen3-8B",
            train_size=1,
            val_size=1,
            train_seed_start=0,
            val_seed_start=100000,
            max_steps=None,
            reward_player=None,
            dry_run=True,
            print_config=True,
            train_batch_size=8,
            max_prompt_length=1024,
            rollout_prompt_length=1024,
            rollout_response_length=512,
            rollout_n=1,
            tensor_model_parallel_size=1,
            gpu_memory_utilization=0.4,
            rollout_max_model_len=1536,
            rollout_max_num_batched_tokens=1536,
            rollout_max_num_seqs=1,
            rollout_enable_sleep_mode=None,
            rollout_enforce_eager=None,
            adv_estimator="gae",
            finetune_mode="lora",
            lora_rank=32,
            lora_alpha=64,
            lora_target_modules="q_proj,k_proj,v_proj,o_proj",
            lora_adapter_path=None,
            ppo_mini_batch_size=8,
            ppo_micro_batch_size_per_gpu=1,
            critic_ppo_micro_batch_size_per_gpu=1,
            log_prob_micro_batch_size_per_gpu=1,
            ref_log_prob_micro_batch_size_per_gpu=1,
            learning_rate=None,
            critic_learning_rate=None,
            entropy_coeff=1e-3,
            n_gpus_per_node=4,
            total_epochs=1,
            test_freq=1000,
            save_freq=1000,
            project_name="mindgames-verl",
            experiment_name=None,
            wandb=False,
            val_before_train=True,
            gradient_checkpointing=True,
            param_offload=True,
            optimizer_offload=True,
            ref_param_offload=True,
            preset=None,
        )

        config = VerlLaunchConfig.from_namespace(args)
        plan = prepare_run_plan(config, root_dir=repo_root, repo_parent=repo_root.parent)

        self.assertEqual(plan.to_payload()["lora_target_modules"], ["q_proj", "k_proj", "v_proj", "o_proj"])
        self.assertIn(
            'actor_rollout_ref.model.target_modules=["q_proj","k_proj","v_proj","o_proj"]',
            plan.overrides,
        )
        self.assertIn(
            'critic.model.target_modules=["q_proj","k_proj","v_proj","o_proj"]',
            plan.overrides,
        )

    def test_prepare_run_plan_keeps_explicit_learning_rate_overrides(self):
        from mindgames.training.verl_launch import VerlLaunchConfig, prepare_run_plan

        repo_root = Path(__file__).resolve().parents[1]
        args = argparse.Namespace(
            game="mini_hanabi",
            env_id=None,
            model="/workspace/models/Qwen3-8B",
            train_size=1,
            val_size=1,
            train_seed_start=0,
            val_seed_start=100000,
            max_steps=None,
            reward_player=None,
            dry_run=True,
            print_config=True,
            train_batch_size=8,
            max_prompt_length=1024,
            rollout_prompt_length=1024,
            rollout_response_length=512,
            rollout_n=1,
            tensor_model_parallel_size=1,
            gpu_memory_utilization=0.4,
            rollout_max_model_len=1536,
            rollout_max_num_batched_tokens=1536,
            rollout_max_num_seqs=1,
            rollout_enable_sleep_mode=None,
            rollout_enforce_eager=None,
            adv_estimator="gae",
            finetune_mode="lora",
            lora_rank=32,
            lora_alpha=64,
            lora_target_modules="all-linear",
            lora_adapter_path=None,
            ppo_mini_batch_size=8,
            ppo_micro_batch_size_per_gpu=1,
            critic_ppo_micro_batch_size_per_gpu=1,
            log_prob_micro_batch_size_per_gpu=1,
            ref_log_prob_micro_batch_size_per_gpu=1,
            learning_rate=2e-6,
            critic_learning_rate=3e-5,
            entropy_coeff=1e-3,
            n_gpus_per_node=4,
            total_epochs=1,
            test_freq=1000,
            save_freq=1000,
            project_name="mindgames-verl",
            experiment_name=None,
            wandb=False,
            val_before_train=True,
            gradient_checkpointing=True,
            param_offload=True,
            optimizer_offload=True,
            ref_param_offload=True,
            preset=None,
        )

        config = VerlLaunchConfig.from_namespace(args)
        plan = prepare_run_plan(config, root_dir=repo_root, repo_parent=repo_root.parent)

        self.assertIn("actor_rollout_ref.actor.optim.lr=2e-06", plan.overrides)
        self.assertIn("critic.optim.lr=3e-05", plan.overrides)


if __name__ == "__main__":
    unittest.main()
