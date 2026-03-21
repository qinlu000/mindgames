import argparse
import sys
import unittest
from pathlib import Path


def _ensure_pkg_importable() -> None:
    project_root = Path(__file__).resolve().parents[1]
    repo_root = project_root.parent
    sys.path.insert(0, str(project_root))
    sys.path.insert(0, str(repo_root))


_ensure_pkg_importable()


class TestTrainingVerlLaunch(unittest.TestCase):
    def test_prepare_run_plan_returns_thin_plan_object(self):
        from mindgames.training.verl_launch import (
            INTERACTION_CLASS,
            REWARD_FUNCTION_PATH,
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
            rollout_update_weights_bucket_megabytes=4096,
            adv_estimator="grpo",
            ppo_mini_batch_size=16,
            ppo_micro_batch_size_per_gpu=1,
            critic_ppo_micro_batch_size_per_gpu=1,
            log_prob_micro_batch_size_per_gpu=1,
            ref_log_prob_micro_batch_size_per_gpu=1,
            learning_rate=5e-7,
            critic_learning_rate=1e-5,
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
        )

        config = VerlLaunchConfig.from_namespace(args)
        plan = prepare_run_plan(config, root_dir=repo_root, repo_parent=repo_root.parent)

        self.assertEqual(plan.config.game, "mini_hanabi")
        self.assertEqual(plan.config.env_id, "MiniHanabi-v0-train")
        self.assertEqual(plan.interaction_class, INTERACTION_CLASS)
        self.assertEqual(plan.reward_function_path, REWARD_FUNCTION_PATH)
        self.assertEqual(len(plan.train_rows), 2)
        self.assertEqual(len(plan.val_rows), 1)
        payload = plan.to_payload()
        self.assertEqual(payload["interaction_class"], INTERACTION_CLASS)
        self.assertEqual(payload["reward_function_path"], REWARD_FUNCTION_PATH)
        self.assertIn("actor_rollout_ref.rollout.multi_turn.enable=True", payload["overrides"])


if __name__ == "__main__":
    unittest.main()
