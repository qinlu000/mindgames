import sys
import unittest
from copy import deepcopy
from pathlib import Path


def _ensure_pkg_importable() -> None:
    project_root = Path(__file__).resolve().parents[1]
    repo_root = project_root.parent
    sys.path.insert(0, str(project_root))
    sys.path.insert(0, str(repo_root))


_ensure_pkg_importable()


class TestMiniHanabiEnv(unittest.TestCase):
    def test_registered_raw_env_can_reset(self):
        import mindgames

        env = mindgames.make("MiniHanabi-v0-raw")
        env.reset(num_players=2, seed=0)

        self.assertEqual(env.state.current_player_id, 0)
        self.assertEqual(env.state.game_state["info_tokens"], 3)
        self.assertEqual(env.state.game_state["fuse_tokens"], 2)
        self.assertEqual(len(env.state.game_state["deck"]), 14)

    def test_fixed_slot_replacement_uses_same_slot_and_resets_knowledge(self):
        from mindgames.envs.MiniHanabi.env import MiniHanabiEnv, Card, SlotKnowledge

        env = MiniHanabiEnv()
        env.reset(num_players=2, seed=0)

        pid = env.state.current_player_id
        env.state.game_state["player_hands"][pid] = [Card("Red", 1), Card("Blue", 2)]
        env.state.game_state["deck"] = [Card("Green", 3)]
        env.state.game_state["knowledge"][pid][0].known_rank = 1
        env.state.game_state["knowledge"][pid][0].not_colors.add("Blue")

        done, _ = env.step("[Discard] 0")
        self.assertFalse(done)
        self.assertEqual(env.state.game_state["discard_pile"][-1], Card("Red", 1))
        self.assertEqual(env.state.game_state["player_hands"][pid][0], Card("Green", 3))
        self.assertEqual(env.state.game_state["player_hands"][pid][1], Card("Blue", 2))
        self.assertEqual(env.state.game_state["knowledge"][pid][0], SlotKnowledge())

    def test_color_hint_updates_positive_and_negative_constraints(self):
        from mindgames.envs.MiniHanabi.env import MiniHanabiEnv, Card

        env = MiniHanabiEnv()
        env.reset(num_players=2, seed=0)

        env.state.game_state["player_hands"][1] = [Card("Red", 1), Card("Blue", 2)]

        done, _ = env.step("[Hint Color Red]")
        self.assertFalse(done)

        knowledge = env.state.game_state["knowledge"][1]
        self.assertEqual(knowledge[0].known_color, "Red")
        self.assertEqual(knowledge[0].last_touched_turn, 1)
        self.assertIn("Red", knowledge[1].not_colors)
        self.assertEqual(env.state.game_state["info_tokens"], 2)

    def test_invalid_hint_does_not_mutate_knowledge(self):
        from mindgames.envs.MiniHanabi.env import MiniHanabiEnv, Card, SlotKnowledge

        env = MiniHanabiEnv()
        env.reset(num_players=2, seed=0)

        env.state.game_state["player_hands"][1] = [Card("Blue", 1), Card("Green", 2)]
        before = deepcopy(env.state.game_state["knowledge"][1])

        done, _ = env.step("[Hint Color Red]")
        self.assertFalse(done)
        self.assertEqual(env.state.current_player_id, 0)
        self.assertEqual(env.state.turn, 0)
        self.assertEqual(env.state.game_state["info_tokens"], 3)
        self.assertEqual(env.state.game_state["knowledge"][1], before)
        self.assertEqual(env.state.game_state["knowledge"][1], [SlotKnowledge(), SlotKnowledge()])

    def test_invalid_color_hint_reports_specific_error(self):
        import mindgames as mg

        env = mg.make("MiniHanabi-v0-train")
        env.reset(num_players=2, seed=0)
        _, _ = env.get_observation()

        done, _ = env.step("[Hint Color Yellow]")
        self.assertFalse(done)

        _, obs = env.get_observation()
        self.assertIn("Unknown color 'Yellow'.", obs)
        self.assertIn("Valid colors are Red, Blue, and Green.", obs)

    def test_successful_rank5_play_restores_info_token(self):
        from mindgames.envs.MiniHanabi.env import MiniHanabiEnv, Card

        env = MiniHanabiEnv()
        env.reset(num_players=2, seed=0)

        pid = env.state.current_player_id
        env.state.game_state["info_tokens"] = 2
        env.state.game_state["fireworks"]["Green"] = 4
        env.state.game_state["player_hands"][pid][0] = Card("Green", 5)
        env.state.game_state["deck"] = []

        done, _ = env.step("[Play A]")
        self.assertFalse(done)
        self.assertEqual(env.state.game_state["fireworks"]["Green"], 5)
        self.assertEqual(env.state.game_state["info_tokens"], 3)
        self.assertIsNone(env.state.game_state["player_hands"][pid][0])

    def test_own_cards_are_hidden_in_board_view(self):
        from mindgames.envs.MiniHanabi.env import MiniHanabiEnv, Card

        env = MiniHanabiEnv()
        env.reset(num_players=2, seed=0)

        env.state.game_state["player_hands"][0] = [Card("Green", 3), Card("Red", 2)]
        env.state.game_state["player_hands"][1] = [Card("Blue", 1), Card("Green", 1)]

        board = env.get_board_str(for_player_id=0)
        self.assertIn("Slot A: Blue 1", board)
        self.assertIn("Slot B: Green 1", board)
        self.assertNotIn("Green 3", board)
        self.assertNotIn("Red 2", board)

    def test_no_state_mutation_after_game_done(self):
        from mindgames.envs.MiniHanabi.env import MiniHanabiEnv, Card

        env = MiniHanabiEnv()
        env.reset(num_players=2, seed=0)

        pid = env.state.current_player_id
        env.state.game_state["fuse_tokens"] = 1
        env.state.game_state["fireworks"]["Blue"] = 0
        env.state.game_state["player_hands"][pid][0] = Card("Blue", 2)
        env.state.game_state["deck"] = []

        done, _ = env.step("[Play A]")
        self.assertTrue(done)

        snapshot = deepcopy(env.state.game_state)
        current_player_id = env.state.current_player_id

        done_after, _ = env.step("[Discard A]")
        self.assertTrue(done_after)
        self.assertEqual(current_player_id, env.state.current_player_id)
        self.assertEqual(snapshot, env.state.game_state)

    def test_turn_cap_ends_after_twenty_eight_consumed_turns(self):
        from mindgames.envs.MiniHanabi.env import MiniHanabiEnv, Card

        env = MiniHanabiEnv()
        env.reset(num_players=2, seed=0)
        env.state.game_state["deck"] = [Card("Red", 1) for _ in range(28)]

        done = False
        for step_idx in range(28):
            done, _ = env.step("[Discard A]")
            if step_idx < 27:
                self.assertFalse(done)

        self.assertTrue(done)
        self.assertEqual(env.state.turn, 28)
        self.assertEqual(env.state.rewards, {0: 0.0, 1: 0.0})


if __name__ == "__main__":
    unittest.main()
