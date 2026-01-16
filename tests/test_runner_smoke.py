import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class TestRunnerSmoke(unittest.TestCase):
    def test_run_rollouts_hanabi_scripted_agent(self):
        repo_root = Path(__file__).resolve().parents[1]
        out_dir = Path(tempfile.mkdtemp(prefix="mindgames_runner_smoke_"))
        out_path = out_dir / "rollouts.jsonl"

        cmd = [
            sys.executable,
            str(repo_root / "tools" / "run_rollouts.py"),
            "--env-id",
            "Hanabi-v0-train",
            "--num-players",
            "2",
            "--episodes",
            "1",
            "--seed",
            "0",
            "--agent",
            "scripted:hanabi_discard0",
            "--out",
            str(out_path),
        ]

        proc = subprocess.run(cmd, cwd=str(repo_root), capture_output=True, text=True)
        self.assertEqual(proc.returncode, 0, msg=f"stderr:\n{proc.stderr}\nstdout:\n{proc.stdout}")
        self.assertTrue(out_path.exists())

        records = [json.loads(line) for line in out_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        end = records[-1]
        self.assertEqual(end["type"], "episode_end")

        steps = [r for r in records if r.get("type") == "step"]
        self.assertEqual(len(steps), 41)

        rewards = end.get("rewards") or {}
        self.assertEqual(float(rewards.get("0", rewards.get(0))), 0.0)
        self.assertEqual(float(rewards.get("1", rewards.get(1))), 0.0)


if __name__ == "__main__":
    unittest.main()
