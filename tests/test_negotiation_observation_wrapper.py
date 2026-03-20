import sys
import unittest
from pathlib import Path


def _ensure_pkg_importable() -> None:
    project_root = Path(__file__).resolve().parents[1]
    repo_root = project_root.parent
    sys.path.insert(0, str(project_root))
    sys.path.insert(0, str(repo_root))


_ensure_pkg_importable()


TEST_RESOURCES = {
    0: {"Wheat": 6, "Wood": 0, "Sheep": 0, "Brick": 0, "Ore": 1},
    1: {"Wheat": 0, "Wood": 3, "Sheep": 0, "Brick": 0, "Ore": 0},
}

TEST_VALUES = {
    0: {"Wheat": 1, "Wood": 10, "Sheep": 1, "Brick": 1, "Ore": 1},
    1: {"Wheat": 8, "Wood": 1, "Sheep": 1, "Brick": 1, "Ore": 1},
}


class TestNegotiationObservationWrapper(unittest.TestCase):
    def test_default_env_clips_long_public_messages(self):
        import mindgames as mg

        env = mg.make(
            "Negotiation-v0",
            starting_resources=TEST_RESOURCES,
            resource_values=TEST_VALUES,
        )
        env.reset(num_players=2, seed=0)
        _, _ = env.get_observation()

        done, _ = env.step("hello " + ("x" * 5000))
        self.assertFalse(done)

        _, obs = env.get_observation()
        self.assertIn("[truncated]", obs)
        self.assertLess(len(obs), 2_500)
        self.assertNotIn("x" * 500, obs)

    def test_env_kwargs_can_tighten_single_turn_history_budget(self):
        import mindgames as mg

        env = mg.make(
            "Negotiation-v0",
            starting_resources=TEST_RESOURCES,
            resource_values=TEST_VALUES,
            observation_max_public_event_chars=80,
            observation_max_public_history_chars=400,
        )
        env.reset(num_players=2, seed=0)
        _, _ = env.get_observation()

        done, _ = env.step("hello " + ("x" * 5000))
        self.assertFalse(done)

        _, obs = env.get_observation()
        self.assertIn("[Negotiator 0] hello", obs)
        self.assertIn("[truncated]", obs)
        self.assertNotIn("x" * 60, obs)


if __name__ == "__main__":
    unittest.main()
