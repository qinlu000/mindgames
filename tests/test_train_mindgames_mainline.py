import json
import subprocess
import sys
import unittest
from pathlib import Path


class TestVerlMainline(unittest.TestCase):
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
        self.assertEqual(payload["adv_estimator"], "grpo")
        self.assertFalse(payload["critic_enabled"])
        self.assertEqual(payload["reward_player"], 0)
        self.assertEqual(
            payload["interaction_class"],
            "mindgames.training.verl_adapter.MindGamesInteraction",
        )
        self.assertEqual(
            payload["reward_function_path"],
            "pkg://mindgames.training.verl_adapter",
        )

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
        self.assertTrue(payload["critic_enabled"])
        self.assertIn(f"critic.model.path={model_path}", payload["overrides"])
        self.assertIn("critic.enable=True", payload["overrides"])


if __name__ == "__main__":
    unittest.main()
