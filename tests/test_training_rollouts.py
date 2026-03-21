import sys
import unittest
from pathlib import Path


def _ensure_pkg_importable() -> None:
    project_root = Path(__file__).resolve().parents[1]
    repo_root = project_root.parent
    sys.path.insert(0, str(project_root))
    sys.path.insert(0, str(repo_root))


_ensure_pkg_importable()


class TestTrainingRollouts(unittest.TestCase):
    def test_run_mindgames_episode_returns_step_trace_and_end_record(self):
        import mindgames as mg
        from mindgames.training import MindGamesEpisode, run_mindgames_episode

        class _ScriptedAgent(mg.Agent):
            def __call__(self, observation: str) -> str:  # noqa: ARG002
                return "[Discard A]"

        episode = MindGamesEpisode.create(
            game="mini_hanabi",
            seed=0,
            env_id="MiniHanabi-v0-train",
            episode_id="trace-test",
        )
        try:
            trace = run_mindgames_episode(
                episode=episode,
                agents={0: _ScriptedAgent(), 1: _ScriptedAgent()},
                seed=0,
                numeric_episode_id=42,
            )
        finally:
            # The episode is closed inside run_mindgames_episode; this is just defensive.
            try:
                episode.close()
            except Exception:
                pass

        self.assertGreaterEqual(len(trace.step_records), 1)
        first = trace.step_records[0]
        self.assertEqual(first["type"], "step")
        self.assertEqual(first["env_id"], "MiniHanabi-v0-train")
        self.assertEqual(first["episode_id"], 42)
        self.assertEqual(first["seed"], 0)
        self.assertIn("observation", first)
        self.assertIn("normalized_action", first)

        end = trace.end_record
        self.assertEqual(end["type"], "episode_end")
        self.assertEqual(end["env_id"], "MiniHanabi-v0-train")
        self.assertEqual(end["episode_id"], 42)
        self.assertIn("rewards", end)
        self.assertIn("game_info", end)


if __name__ == "__main__":
    unittest.main()
