import sys
import unittest
from pathlib import Path


def _ensure_pkg_importable() -> None:
    project_root = Path(__file__).resolve().parents[1]  # .../mindgames
    repo_root = project_root.parent  # .../ (contains mindgames/ and textarena/)
    sys.path.insert(0, str(project_root))
    sys.path.insert(0, str(repo_root))


_ensure_pkg_importable()


class TestCodenamesEnv(unittest.TestCase):
    def test_codenames_registered(self):
        import mindgames as mg

        self.assertIn("Codenames-v0", mg.ENV_REGISTRY)
        self.assertIn("Codenames-v0-train", mg.ENV_REGISTRY)
        self.assertIn("Codenames-v0-hardcore", mg.ENV_REGISTRY)

    def test_basic_turn_flow(self):
        import mindgames as mg

        env = mg.make("Codenames-v0-train")
        env.reset(num_players=4, seed=0)

        self.assertEqual(env.state.current_player_id, 0)

        done, _ = env.step("[ocean 1]")
        self.assertFalse(done)
        self.assertEqual(env.state.current_player_id, 1)

        done, _ = env.step("[pass]")
        self.assertFalse(done)
        self.assertEqual(env.state.current_player_id, 2)

    def test_local_wordlists_load(self):
        from mindgames.envs.Codenames.env import CodenamesEnv

        basic = CodenamesEnv(hardcore=False)
        hardcore = CodenamesEnv(hardcore=True)

        self.assertGreaterEqual(len(basic.word_list), 25)
        self.assertGreater(len(hardcore.word_list), len(basic.word_list))


if __name__ == "__main__":
    unittest.main()
