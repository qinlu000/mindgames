import sys
import unittest
from pathlib import Path


def _ensure_pkg_importable() -> None:
    project_root = Path(__file__).resolve().parents[1]  # .../mindgames
    repo_root = project_root.parent  # .../ (contains mindgames/ and textarena/)
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


class TestNegotiationEnv(unittest.TestCase):
    def test_negotiation_registered(self):
        import mindgames as mg

        self.assertIn("Negotiation-v0", mg.ENV_REGISTRY)
        self.assertIn("Negotiation-v0-train", mg.ENV_REGISTRY)
        self.assertIn("Negotiation-v0-short", mg.ENV_REGISTRY)
        self.assertIn("Negotiation-v0-long", mg.ENV_REGISTRY)

    def test_prompt_uses_public_inventories_and_private_values(self):
        import mindgames as mg

        env = mg.make(
            "Negotiation-v0",
            starting_resources=TEST_RESOURCES,
            resource_values=TEST_VALUES,
        )
        env.reset(num_players=2, seed=0)
        player_id, obs = env.get_observation()

        self.assertEqual(player_id, 0)
        self.assertIn("Both players can see both players' current inventories.", obs)
        self.assertIn("Player 0 inventory: 6 Wheat, 1 Ore", obs)
        self.assertIn("Player 1 inventory: 3 Wood", obs)
        self.assertIn("Your private per-unit values: Wheat=1, Wood=10, Sheep=1, Brick=1, Ore=1", obs)
        self.assertNotIn("Player 1 private per-unit values", obs)

    def test_even_turn_requirement(self):
        from mindgames.envs.Negotiation.env import NegotiationEnv

        with self.assertRaises(ValueError):
            NegotiationEnv(max_turns=3)

    def test_value_generator_produces_multiple_rankings(self):
        from mindgames.envs.Negotiation.env import NegotiationEnv

        env = NegotiationEnv()
        rankings = set()
        for _ in range(40):
            values = env._generate_resource_values()
            for pid in [0, 1]:
                ranking = tuple(sorted(values[pid], key=values[pid].get))
                rankings.add(ranking)
        self.assertGreater(len(rankings), 5)

    def test_offer_accept_executes_trade_and_resolves_by_gain(self):
        from mindgames.envs.Negotiation.env import NegotiationEnv

        env = NegotiationEnv(
            max_turns=2,
            starting_resources=TEST_RESOURCES,
            resource_values=TEST_VALUES,
        )
        env.reset(num_players=2, seed=0)

        done, _ = env.step("[Offer: 2 Wheat -> 1 Wood]")
        self.assertFalse(done)
        self.assertEqual(env.state.current_player_id, 1)

        done, info = env.step("[Accept]")
        self.assertTrue(done)
        self.assertIn("Turn limit reached", info["reason"])
        self.assertEqual(env.state.game_state["resources"][0]["Wheat"], 4)
        self.assertEqual(env.state.game_state["resources"][0]["Wood"], 1)
        self.assertEqual(env.state.game_state["resources"][1]["Wheat"], 2)
        self.assertEqual(env.state.game_state["resources"][1]["Wood"], 2)
        self.assertEqual(env.state.rewards, {0: -1, 1: 1})

    def test_deny_and_counteroffer_sets_new_pending_offer(self):
        from mindgames.envs.Negotiation.env import NegotiationEnv

        env = NegotiationEnv(
            max_turns=4,
            starting_resources=TEST_RESOURCES,
            resource_values=TEST_VALUES,
        )
        env.reset(num_players=2, seed=0)

        done, _ = env.step("[Offer: 2 Wheat -> 1 Wood]")
        self.assertFalse(done)

        done, _ = env.step("No deal. [Deny] [Offer: 1 Wood -> 1 Ore]")
        self.assertFalse(done)
        self.assertEqual(env.state.current_player_id, 0)

        pending_offer = env.state.game_state["pending_offer"]
        self.assertIsNotNone(pending_offer)
        self.assertEqual(pending_offer["from_player"], 1)
        self.assertEqual(pending_offer["to_player"], 0)
        self.assertEqual(pending_offer["give_bundle"], {"Wood": 1})
        self.assertEqual(pending_offer["request_bundle"], {"Ore": 1})

    def test_train_wrapper_preserves_long_deny_tag(self):
        import mindgames as mg

        env = mg.make(
            "Negotiation-v0-train",
            starting_resources=TEST_RESOURCES,
            resource_values=TEST_VALUES,
        )
        env.reset(num_players=2, seed=0)

        done, _ = env.step("[Offer: 2 Wheat -> 1 Wood]")
        self.assertFalse(done)

        done, _ = env.step("[Deny] " + ("x" * 1500))
        self.assertFalse(done)
        self.assertEqual(env.state.current_player_id, 0)
        self.assertIsNone(env.state.game_state["pending_offer"])

    def test_train_observation_stays_bounded(self):
        import mindgames as mg

        env = mg.make("Negotiation-v0-long-train")
        env.reset(num_players=2, seed=0)
        action = "public note " + ("x" * 200)

        for _ in range(20):
            _, _ = env.get_observation()
            done, _ = env.step(action)
            self.assertFalse(done)

        player_id, obs = env.get_observation()
        self.assertIn("Recent public history", obs)
        self.assertLess(len(obs), 5000)
        self.assertIn(f"Turn {env.state.turn + 1} of {env.max_turns}", obs)
        self.assertIn(f"Player {player_id} inventory", obs)


if __name__ == "__main__":
    unittest.main()
