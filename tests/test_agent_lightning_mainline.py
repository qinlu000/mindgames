import json
import subprocess
import sys
import unittest
from pathlib import Path


class TestAgentLightningMainline(unittest.TestCase):
    def test_train_cli_dry_run_prints_resolved_config(self):
        repo_root = Path(__file__).resolve().parents[1]
        cmd = [
            sys.executable,
            str(repo_root / "tools" / "train" / "train_agent_lightning_games_verl.py"),
            "--mode",
            "train",
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
        self.assertEqual(payload["train_tasks"], 4)
        self.assertEqual(payload["val_tasks"], 2)
        self.assertTrue(payload["same_llm_all_seats"])
        self.assertEqual(payload["reward_player"], 0)


if __name__ == "__main__":
    unittest.main()
