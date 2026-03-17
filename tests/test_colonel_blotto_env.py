import sys
import unittest
from pathlib import Path


def _ensure_pkg_importable() -> None:
    project_root = Path(__file__).resolve().parents[1]  # .../mindgames
    repo_root = project_root.parent  # .../ (contains mindgames/ and textarena/)
    sys.path.insert(0, str(project_root))
    sys.path.insert(0, str(repo_root))


_ensure_pkg_importable()


class TestColonelBlottoEnv(unittest.TestCase):
    def test_colonel_blotto_registered(self):
        import mindgames as mg

        self.assertIn("ColonelBlotto-v0", mg.ENV_REGISTRY)
        self.assertIn("ColonelBlotto-v0-train", mg.ENV_REGISTRY)

    def test_round_resolution_and_turn_rotation(self):
        import mindgames as mg

        env = mg.make("ColonelBlotto-v0-train")
        env.reset(num_players=2, seed=0)

        self.assertEqual(env.state.current_player_id, 0)

        done, _ = env.step("[A10 B6 C4]")
        self.assertFalse(done)
        self.assertEqual(env.state.current_player_id, 1)
        self.assertEqual(env.state.game_state["current_round"], 1)
        self.assertEqual(env.state.game_state["scores"], {0: 0, 1: 0})

        done, _ = env.step("[A9 B5 C6]")
        self.assertFalse(done)
        self.assertEqual(env.state.current_player_id, 0)
        self.assertEqual(env.state.game_state["current_round"], 2)
        self.assertEqual(env.state.game_state["scores"], {0: 1, 1: 0})

    def test_invalid_allocation_keeps_turn_then_loses(self):
        from mindgames.envs.ColonelBlotto.env import ColonelBlottoEnv

        env = ColonelBlottoEnv(num_rounds=3)
        env.reset(num_players=2, seed=0)

        done, _ = env.step("[A1 B1 C1]")
        self.assertFalse(done)
        self.assertEqual(env.state.current_player_id, 0)
        self.assertFalse(env.state.done)

        done, _ = env.step("[A1 B1 C1]")
        self.assertTrue(done)
        self.assertEqual(env.state.rewards, {0: -1, 1: 1})
        self.assertTrue(env.state.game_info[0]["invalid_move"])

    def test_shared_state_is_not_reused_between_players(self):
        from mindgames.envs.ColonelBlotto.env import ColonelBlottoEnv

        env = ColonelBlottoEnv()
        env.reset(num_players=2, seed=0)

        env.state.game_state["player_states"][0]["current_allocation"]["A"] = 20
        self.assertEqual(env.state.game_state["player_states"][1]["current_allocation"]["A"], 0)

    def test_game_ends_after_configured_rounds(self):
        from mindgames.envs.ColonelBlotto.env import ColonelBlottoEnv

        env = ColonelBlottoEnv(num_rounds=2)
        env.reset(num_players=2, seed=0)

        done, _ = env.step("[A10 B6 C4]")
        self.assertFalse(done)
        done, _ = env.step("[A9 B5 C6]")
        self.assertFalse(done)
        done, _ = env.step("[A8 B7 C5]")
        self.assertFalse(done)
        done, _ = env.step("[A7 B6 C7]")
        self.assertTrue(done)
        self.assertEqual(env.state.rewards, {0: 1, 1: -1})


if __name__ == "__main__":
    unittest.main()
