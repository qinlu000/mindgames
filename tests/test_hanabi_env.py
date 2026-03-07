import sys
import unittest
from copy import deepcopy
from pathlib import Path


def _ensure_pkg_importable() -> None:
    project_root = Path(__file__).resolve().parents[1]  # .../mindgames
    repo_root = project_root.parent  # .../ (contains mindgames/ and textarena/)
    sys.path.insert(0, str(project_root))
    sys.path.insert(0, str(repo_root))


_ensure_pkg_importable()


class TestHanabiEnv(unittest.TestCase):
    def test_marshal_dense_reward_toggle(self):
        from mindgames.envs.Hanabi.env import HanabiEnv

        env = HanabiEnv(marshal_dense_reward=False)
        env.reset(num_players=2, seed=0)
        _, info = env.step("[Discard] 0")
        self.assertNotIn("step_reward", info)

        env_dense = HanabiEnv(marshal_dense_reward=True, marshal_fuse_penalty=1.0)
        env_dense.reset(num_players=2, seed=0)
        _, info_dense = env_dense.step("[Discard] 0")
        self.assertIn("step_reward", info_dense)
        self.assertIsInstance(info_dense["step_reward"], float)

    def test_deck_exhaustion_final_round_2p(self):
        # For 2 players, initial hands use 5 cards each, so the deck has:
        # 50 - 2*5 = 40 cards remaining after dealing.
        # If both players always discard, the last card is drawn on turn 40.
        # Hanabi rules: after the last card is drawn, each player gets one final turn.
        # Therefore the episode should end after 42 completed turns.
        from mindgames.envs.Hanabi.env import HanabiEnv

        env = HanabiEnv()
        env.reset(num_players=2, seed=0)

        steps = 0
        done = False
        while not done and steps < 200:
            done, _ = env.step("[Discard] 0")
            steps += 1

        self.assertTrue(done, "Expected episode to terminate by deck exhaustion.")
        self.assertEqual(steps, 42, "Deck-exhaustion final-round length is off-by-one.")
        self.assertEqual(env.state.game_info[0]["turn_count"], 21)
        self.assertEqual(env.state.game_info[1]["turn_count"], 21)

    def test_playing_rank5_regains_info_token(self):
        from mindgames.envs.Hanabi.env import HanabiEnv, Card, Suit

        env = HanabiEnv()
        env.reset(num_players=2, seed=0)

        pid = env.state.current_player_id
        env.state.game_state["info_tokens"] = 7
        env.state.game_state["fireworks"][Suit.RED] = 4
        env.state.game_state["player_hands"][pid][0] = Card(suit=Suit.RED, rank=5)

        done, _ = env.step("[Play] 0")
        self.assertFalse(done)
        self.assertEqual(env.state.game_state["fireworks"][Suit.RED], 5)
        self.assertEqual(env.state.game_state["info_tokens"], 8)

    def test_no_state_mutation_after_game_done(self):
        from mindgames.envs.Hanabi.env import HanabiEnv

        env = HanabiEnv()
        env.reset(num_players=2, seed=0)

        done = False
        # Force terminal state by repeatedly making likely-invalid plays.
        for _ in range(20):
            done, _ = env.step("[Play] 0")
            if done:
                break
        self.assertTrue(done, "Expected game to end before post-terminal check.")

        snapshot = deepcopy(env.state.game_state)
        current_player_id = env.state.current_player_id

        done_after, _ = env.step("[Discard] 0")
        self.assertTrue(done_after)
        self.assertEqual(current_player_id, env.state.current_player_id)
        self.assertEqual(snapshot, env.state.game_state, "Game state mutated after terminal step.")


if __name__ == "__main__":
    unittest.main()
