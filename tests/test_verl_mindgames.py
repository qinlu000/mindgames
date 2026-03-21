import asyncio
import sys
import unittest
from pathlib import Path


def _ensure_pkg_importable() -> None:
    project_root = Path(__file__).resolve().parents[1]
    repo_root = project_root.parent
    sys.path.insert(0, str(project_root))
    sys.path.insert(0, str(repo_root))


_ensure_pkg_importable()


class TestVerlMindGames(unittest.TestCase):
    def test_build_dataset_row_contains_prompt_and_interaction_kwargs(self):
        from mindgames.training import build_dataset_row

        row = build_dataset_row(
            game="mini_hanabi",
            seed=0,
            index=7,
            env_id="MiniHanabi-v0-train",
            max_steps=12,
            reward_player=-1,
        )

        self.assertEqual(row["data_source"], "mindgames/mini_hanabi")
        self.assertEqual(row["extra_info"]["index"], 7)
        self.assertEqual(row["extra_info"]["interaction_kwargs"]["name"], "mindgames")
        self.assertEqual(len(row["prompt"]), 2)
        self.assertEqual(row["prompt"][0]["role"], "system")
        self.assertEqual(row["prompt"][1]["role"], "user")
        self.assertIn("current player may change between turns", row["prompt"][0]["content"])
        self.assertIn("self-contained snapshot", row["prompt"][0]["content"])
        self.assertNotIn("You are Player 0", row["prompt"][0]["content"])
        self.assertTrue(row["prompt"][1]["content"].startswith("Current game state:\n"))
        self.assertIn("MiniHanabi-v0", row["prompt"][1]["content"])

    def test_compute_score_uses_last_turn_score(self):
        from mindgames.training import compute_score

        result = compute_score(
            data_source="mindgames/mini_hanabi",
            solution_str="[Discard A]",
            ground_truth="",
            extra_info={"turn_scores": [0.0, 0.25, 0.5], "tool_rewards": []},
        )

        self.assertEqual(result["score"], 0.5)
        self.assertEqual(result["terminal_reward"], 0.5)

    def test_interaction_can_step_once(self):
        from mindgames.training import MindGamesInteraction

        async def run_step():
            interaction = MindGamesInteraction(config={})
            instance_id = await interaction.start_interaction(
                game="mini_hanabi",
                seed=0,
                env_id="MiniHanabi-v0-train",
                max_steps=12,
                reward_player=-1,
            )

            finished, response, reward, metrics = await interaction.generate_response(
                instance_id,
                [{"role": "assistant", "content": "[Discard A]"}],
            )
            self.assertFalse(finished)
            self.assertGreaterEqual(reward, 0.0)
            self.assertIn("normalized_action", metrics)
            self.assertEqual(metrics["normalized_action"], "[Discard A]")
            self.assertIsInstance(response, str)
            self.assertTrue(response.startswith("Current game state:\n"))
            self.assertIn("MiniHanabi-v0", response)

        asyncio.run(run_step())


if __name__ == "__main__":
    unittest.main()
