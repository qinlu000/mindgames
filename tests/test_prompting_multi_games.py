import sys
import unittest
from pathlib import Path


def _ensure_pkg_importable() -> None:
    project_root = Path(__file__).resolve().parents[1]
    repo_root = project_root.parent
    sys.path.insert(0, str(project_root))
    sys.path.insert(0, str(repo_root))


_ensure_pkg_importable()


class TestPromptingMultiGames(unittest.TestCase):
    def test_colonel_blotto_env_spec_exposes_prompt_metadata(self):
        import mindgames as mg

        env_spec = mg.get_env_spec("ColonelBlotto-v0")
        self.assertIsNotNone(env_spec.prompt_profile)
        self.assertEqual(env_spec.prompt_profile.template_name, "qwen3")
        self.assertEqual(env_spec.prompt_profile.action_mode, "structured")
        self.assertEqual(env_spec.reward_mode, "zero_sum_terminal")
        self.assertEqual(env_spec.obs_mode, "board_state")
        self.assertIsNotNone(env_spec.resolve_action_parser())

    def test_colonel_blotto_action_parser_enumerates_allocations(self):
        import mindgames as mg
        from mindgames.prompting import get_legal_actions_for_env

        env = mg.make("ColonelBlotto-v0-train")
        env.reset(num_players=2, seed=0)
        _, observation = env.get_observation()

        legal_actions = get_legal_actions_for_env(env, observation)
        self.assertIsNotNone(legal_actions)
        self.assertEqual(len(legal_actions), 231)
        self.assertIn("[A10 B5 C5]", legal_actions)
        self.assertIn("[A0 B0 C20]", legal_actions)

    def test_colonel_blotto_normalizer_uses_legal_action_space(self):
        import mindgames as mg
        from mindgames.prompting import normalize_action_for_env

        env = mg.make("ColonelBlotto-v0-train")
        env.reset(num_players=2, seed=0)
        _, observation = env.get_observation()

        raw_output = "Final answer: \\boxed{A10 B5 C5}"
        normalized = normalize_action_for_env(env, observation, raw_output)
        self.assertEqual(normalized, "[A10 B5 C5]")

    def test_negotiation_env_spec_exposes_prompt_metadata(self):
        import mindgames as mg

        env_spec = mg.get_env_spec("Negotiation-v0")
        self.assertIsNotNone(env_spec.prompt_profile)
        self.assertEqual(env_spec.prompt_profile.template_name, "qwen3")
        self.assertEqual(env_spec.prompt_profile.action_mode, "chat")
        self.assertEqual(env_spec.reward_mode, "value_gain")
        self.assertEqual(env_spec.obs_mode, "public_private_chat")
        self.assertIsNotNone(env_spec.resolve_action_parser())

    def test_negotiation_action_parser_emits_control_actions(self):
        import mindgames as mg
        from mindgames.prompting import get_legal_actions_for_env

        env = mg.make("Negotiation-v0-train")
        env.reset(num_players=2, seed=0)
        _, observation = env.get_observation()

        legal_actions = get_legal_actions_for_env(env, observation)
        self.assertIsNotNone(legal_actions)
        self.assertTrue(any(action.startswith("[Offer: ") for action in legal_actions))
        self.assertIn("What trade helps you most?", legal_actions)

    def test_negotiation_normalizer_preserves_plain_chat(self):
        import mindgames as mg
        from mindgames.prompting import normalize_action_for_env

        env = mg.make("Negotiation-v0-train")
        env.reset(num_players=2, seed=0)
        _, observation = env.get_observation()

        raw_output = "Final answer: I'm open to a trade that improves both sides."
        normalized = normalize_action_for_env(env, observation, raw_output)
        self.assertEqual(normalized, "I'm open to a trade that improves both sides.")


if __name__ == "__main__":
    unittest.main()
