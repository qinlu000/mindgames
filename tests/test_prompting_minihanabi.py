import sys
import unittest
from pathlib import Path


def _ensure_pkg_importable() -> None:
    project_root = Path(__file__).resolve().parents[1]
    repo_root = project_root.parent
    sys.path.insert(0, str(project_root))
    sys.path.insert(0, str(repo_root))


_ensure_pkg_importable()


class TestMiniHanabiPrompting(unittest.TestCase):
    def test_mini_hanabi_env_spec_exposes_prompt_metadata(self):
        import mindgames as mg

        env_spec = mg.get_env_spec("MiniHanabi-v0")
        self.assertIsNotNone(env_spec.prompt_profile)
        self.assertEqual(env_spec.prompt_profile.template_name, "qwen3")
        self.assertEqual(env_spec.reward_mode, "team_score")
        self.assertEqual(env_spec.obs_mode, "board_state")
        self.assertIsNotNone(env_spec.resolve_action_parser())

    def test_mini_hanabi_action_parser_uses_visible_partner_hand(self):
        import mindgames as mg
        from mindgames.prompting import get_legal_actions_for_env

        env = mg.make("MiniHanabi-v0-train")
        env.reset(num_players=2, seed=0)
        _, observation = env.get_observation()

        legal_actions = get_legal_actions_for_env(env, observation)
        self.assertIsNotNone(legal_actions)
        self.assertIn("[Play A]", legal_actions)
        self.assertIn("[Discard B]", legal_actions)
        self.assertIn("[Hint Color Red]", legal_actions)
        self.assertNotIn("[Hint Color Blue]", legal_actions)
        self.assertNotIn("[Hint Color Green]", legal_actions)
        self.assertIn("[Hint Rank 4]", legal_actions)
        self.assertIn("[Hint Rank 1]", legal_actions)
        self.assertNotIn("[Hint Rank 2]", legal_actions)

    def test_mini_hanabi_normalizer_matches_legal_action_from_boxed_output(self):
        import mindgames as mg
        from mindgames.prompting import normalize_action_for_env

        env = mg.make("MiniHanabi-v0-train")
        env.reset(num_players=2, seed=0)
        _, observation = env.get_observation()

        raw_output = (
            "<|im_start|>assistant\n"
            "Let me reason step by step. "
            "The best move is \\boxed{Play A}."
        )
        normalized = normalize_action_for_env(env, observation, raw_output)
        self.assertEqual(normalized, "[Play A]")


if __name__ == "__main__":
    unittest.main()
