import os
import subprocess
import tempfile
import unittest
from pathlib import Path


class TestTrainWrappers(unittest.TestCase):
    def _run_script(self, script: str, *, env: dict[str, str]) -> subprocess.CompletedProcess[str]:
        repo_root = Path(__file__).resolve().parents[1]
        cmd = ["bash", str(repo_root / script)]
        merged_env = os.environ.copy()
        merged_env.update(env)
        return subprocess.run(cmd, cwd=str(repo_root), capture_output=True, text=True, env=merged_env)

    def test_rlhf_base_rejects_ppo(self):
        out_dir = tempfile.mkdtemp(prefix="mindgames_ppo_base_")
        proc = self._run_script(
            "tools/train/train_rlhf_base.sh",
            env={
                "RLHF_TYPE": "ppo",
                "USE_VLLM": "false",
                "MODEL": "Qwen/Qwen3-8B",
                "DATASET": "data/hanabi.grpo.jsonl",
                "OUTPUT_DIR": out_dir,
                "DRY_RUN": "true",
            },
        )

        self.assertNotEqual(proc.returncode, 0, msg=f"stderr:\n{proc.stderr}\nstdout:\n{proc.stdout}")
        self.assertIn("only supports RLHF_TYPE=grpo", proc.stderr)

    def test_hanabi_wrapper_rejects_ppo(self):
        proc = self._run_script(
            "tools/train/train_hanabi_rlhf_simple.sh",
            env={
                "RLHF_TYPE": "ppo",
                "DRY_RUN": "true",
            },
        )

        self.assertNotEqual(proc.returncode, 0, msg=f"stderr:\n{proc.stderr}\nstdout:\n{proc.stdout}")
        self.assertIn("only supports RLHF_TYPE=grpo", proc.stderr)

    def test_grpo_base_alias_keeps_grpo_branch_dry_run(self):
        out_dir = tempfile.mkdtemp(prefix="mindgames_grpo_base_")
        proc = self._run_script(
            "tools/train/train_grpo_base.sh",
            env={
                "RLHF_TYPE": "grpo",
                "USE_VLLM": "false",
                "MODEL": "Qwen/Qwen3-8B",
                "DATASET": "data/hanabi.grpo.jsonl",
                "OUTPUT_DIR": out_dir,
                "DRY_RUN": "true",
            },
        )

        self.assertEqual(proc.returncode, 0, msg=f"stderr:\n{proc.stderr}\nstdout:\n{proc.stdout}")
        self.assertIn("--rlhf_type grpo", proc.stderr)
        self.assertIn("--num_generations 8", proc.stderr)
        self.assertNotIn("--reward_model", proc.stderr)

    def test_hanabi_dapo_wrapper_injects_dapo_flags(self):
        out_dir = tempfile.mkdtemp(prefix="mindgames_dapo_hanabi_")
        proc = self._run_script(
            "tools/train/train_dapo_hanabi_server_simple.sh",
            env={
                "MODEL": "Qwen/Qwen3-8B",
                "DATASET": "data/hanabi.grpo.jsonl",
                "OUTPUT_DIR": out_dir,
                "USE_VLLM": "false",
                "DRY_RUN": "true",
            },
        )

        self.assertEqual(proc.returncode, 0, msg=f"stderr:\n{proc.stderr}\nstdout:\n{proc.stdout}")
        self.assertIn("--rlhf_type grpo", proc.stderr)
        self.assertIn("--loss_type dapo", proc.stderr)
        self.assertIn("--beta 0", proc.stderr)
        self.assertNotIn("--use_valid_tokens_only", proc.stderr)


if __name__ == "__main__":
    unittest.main()
