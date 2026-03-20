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


class TestNegotiationActionClipWrapper(unittest.TestCase):
    def test_does_not_promote_non_prefixed_control_tags(self):
        from mindgames.envs.Negotiation.env import NegotiationEnv
        from mindgames.wrappers import NegotiationActionClipWrapper

        env = NegotiationActionClipWrapper(NegotiationEnv(), max_num_characters=40)
        action = "hello " + ("x" * 120) + " [Deny] maybe later"

        out = env.action(action)

        self.assertEqual(out, action[:40])
        self.assertNotIn("[Deny]", out)

    def test_preserves_prefixed_control_order(self):
        from mindgames.envs.Negotiation.env import NegotiationEnv
        from mindgames.wrappers import NegotiationActionClipWrapper

        env = NegotiationActionClipWrapper(NegotiationEnv(), max_num_characters=80)
        action = "  [Offer: 2 Wheat -> 1 Wood] [Deny] " + ("x" * 120)

        out = env.action(action)

        self.assertTrue(out.startswith("[Offer: 2 Wheat -> 1 Wood] [Deny] "))
        self.assertLessEqual(len(out), 80)

    def test_train_env_keeps_pending_offer_when_suffix_deny_is_clipped(self):
        import mindgames as mg

        env = mg.make(
            "Negotiation-v0-train",
            starting_resources=TEST_RESOURCES,
            resource_values=TEST_VALUES,
        )
        env.reset(num_players=2, seed=0)
        _, _ = env.get_observation()

        done, _ = env.step("[Offer: 2 Wheat -> 1 Wood]")
        self.assertFalse(done)
        self.assertEqual(env.state.current_player_id, 1)

        long_chat = "I am thinking " + ("x" * 1500) + " [Deny] maybe later"
        done, _ = env.step(long_chat)

        self.assertFalse(done)
        self.assertEqual(env.state.current_player_id, 1)
        self.assertIsNotNone(env.state.game_state["pending_offer"])
        self.assertEqual(len(env.state.game_state["trade_history"]), 1)

        _, obs = env.get_observation()
        self.assertIn("There is a pending offer to you.", obs)


if __name__ == "__main__":
    unittest.main()
