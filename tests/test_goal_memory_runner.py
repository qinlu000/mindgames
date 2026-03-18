import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class TestGoalMemoryRunner(unittest.TestCase):
    def test_run_rollouts_hanabi_with_goal_memory_wrapper(self):
        repo_root = Path(__file__).resolve().parents[1]
        out_dir = Path(tempfile.mkdtemp(prefix="mindgames_goal_memory_runner_"))
        out_path = out_dir / "rollouts.jsonl"

        payload = json.dumps(
            {
                "selected_goal_id": "safe_self_slot0",
                "goal_ops": [
                    {
                        "op": "set",
                        "goal_id": "safe_self_slot0",
                        "goal": "discard fallback card if needed",
                        "target": "self_slot0",
                        "priority": "low",
                        "ttl": 1,
                    }
                ],
                "action": "[Discard] 0",
            },
            ensure_ascii=False,
        )

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
            f"scripted:const={payload}",
            "--goal-memory-enabled",
            "--out",
            str(out_path),
        ]

        proc = subprocess.run(cmd, cwd=str(repo_root), capture_output=True, text=True)
        self.assertEqual(proc.returncode, 0, msg=f"stderr:\n{proc.stderr}\nstdout:\n{proc.stdout}")

        records = [json.loads(line) for line in out_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        steps = [r for r in records if r.get("type") == "step"]
        self.assertTrue(steps)
        self.assertIn("goal_memory", steps[0])
        self.assertIn("goal_turn_output", steps[0])
        self.assertEqual(steps[0]["goal_turn_output"]["selected_goal_id"], "safe_self_slot0")


if __name__ == "__main__":
    unittest.main()
