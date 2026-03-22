import sys
import unittest
from pathlib import Path


def _ensure_pkg_importable() -> None:
    project_root = Path(__file__).resolve().parents[1]
    repo_root = project_root.parent
    sys.path.insert(0, str(project_root))
    sys.path.insert(0, str(repo_root))


_ensure_pkg_importable()


class TestMindGamesEpisode(unittest.TestCase):
    def test_mini_hanabi_step_contract_is_self_contained(self):
        from mindgames.training import MindGamesEpisode

        episode = MindGamesEpisode.create(
            game="mini_hanabi",
            seed=0,
            env_id="MiniHanabi-v0-train",
            episode_id="mini-hanabi-test",
        )
        try:
            step = episode.current_step()
            self.assertEqual(step.game, "mini_hanabi")
            self.assertEqual(step.env_id, "MiniHanabi-v0-train")
            self.assertEqual(step.episode_id, "mini-hanabi-test")
            self.assertEqual(step.actor_id, 0)
            self.assertEqual(step.turn_index, 0)
            self.assertEqual(step.obs_mode, "board_state")
            self.assertEqual(step.reward_mode, "team_score")
            self.assertEqual(step.action_mode, "structured")
            self.assertIn("MiniHanabi-v0", step.observation)
            self.assertIn("Action formats:", step.observation)
            self.assertIsNotNone(step.legal_actions)
            self.assertIn("[Play A]", step.legal_actions)
        finally:
            episode.close()

    def test_step_transition_advances_to_next_self_contained_observation(self):
        from mindgames.training import MindGamesEpisode

        episode = MindGamesEpisode.create(
            game="mini_hanabi",
            seed=0,
            env_id="MiniHanabi-v0-train",
            episode_id="mini-hanabi-step",
        )
        try:
            transition = episode.step("[Discard A]")
            self.assertFalse(transition.done)
            self.assertEqual(transition.normalized_action, "[Discard A]")
            self.assertEqual(transition.reward_delta, 0.0)
            self.assertIsNone(transition.terminal_reward)
            self.assertIsNone(transition.terminal_message)
            self.assertIsNotNone(transition.next_step)
            assert transition.next_step is not None
            self.assertEqual(transition.next_step.actor_id, 1)
            self.assertEqual(transition.next_step.turn_index, 1)
            self.assertIn("Current player: Player 1", transition.next_step.observation)
        finally:
            episode.close()

    def test_negotiation_step_contract_carries_public_history_in_observation(self):
        from mindgames.training import MindGamesEpisode

        episode = MindGamesEpisode.create(
            game="negotiation",
            seed=0,
            env_id="Negotiation-v0-train",
            episode_id="negotiation-test",
        )
        try:
            step = episode.current_step()
            self.assertEqual(step.action_mode, "chat")
            self.assertEqual(step.obs_mode, "public_private_chat")
            self.assertEqual(step.reward_mode, "value_gain")
            self.assertIn("No pending offer to you.", step.observation)
            self.assertIn("Public inventories:", step.observation)
            self.assertIsNotNone(step.legal_actions)
            self.assertTrue(any(action.startswith("[Offer: ") for action in step.legal_actions))
        finally:
            episode.close()


if __name__ == "__main__":
    unittest.main()
