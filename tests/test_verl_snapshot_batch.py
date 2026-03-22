import sys
import unittest
from pathlib import Path


def _ensure_pkg_importable() -> None:
    project_root = Path(__file__).resolve().parents[1]
    repo_root = project_root.parent
    sys.path.insert(0, str(project_root))
    sys.path.insert(0, str(repo_root))


_ensure_pkg_importable()


class TestVerlSnapshotBatch(unittest.TestCase):
    def test_expand_episode_training_rows_emits_all_steps(self):
        from mindgames.training.verl_snapshot_batch import expand_episode_training_rows

        tensors, non_tensors, meta_info = expand_episode_training_rows(
            episode_rollouts=[
                {
                    "terminal_reward": 0.75,
                    "episode_step_data": [
                        {
                            "messages": [{"role": "user", "content": "Current game state:\nSTATE ZERO"}],
                            "prompt_ids": [1, 2, 3],
                            "response_ids": [11],
                            "response_logprobs": [-0.1],
                            "actor_id": 0,
                            "turn_index": 0,
                            "normalized_action": "[Discard A]",
                        },
                        {
                            "messages": [{"role": "user", "content": "Current game state:\nSTATE ONE"}],
                            "prompt_ids": [4, 5],
                            "response_ids": [22, 23],
                            "response_logprobs": [-0.2, -0.3],
                            "actor_id": 1,
                            "turn_index": 1,
                            "normalized_action": "[Play B]",
                        },
                    ],
                }
            ],
            root_rows=[
                {
                    "uid": "episode-0",
                    "data_source": "mindgames/mini_hanabi",
                    "reward_model": {"ground_truth": ""},
                    "extra_info": {"index": 7},
                    "index": 7,
                    "interaction_kwargs": {"game": "mini_hanabi", "seed": 0},
                    "tools_kwargs": {},
                    "raw_prompt": [{"role": "user", "content": "INITIAL"}],
                }
            ],
            prompt_length=6,
            response_length=4,
            pad_token_id=0,
        )

        self.assertEqual(tuple(tensors["prompts"].shape), (2, 6))
        self.assertEqual(tuple(tensors["responses"].shape), (2, 4))
        self.assertEqual(tuple(tensors["rm_scores"].shape), (2, 4))
        self.assertAlmostEqual(float(tensors["rm_scores"][0, 0].item()), 0.75)
        self.assertAlmostEqual(float(tensors["rm_scores"][1, 1].item()), 0.75)
        self.assertEqual(non_tensors["step_turn_index"].tolist(), [0, 1])
        self.assertEqual(non_tensors["step_actor_id"].tolist(), [0, 1])
        self.assertEqual(non_tensors["episode_step_count"].tolist(), [2, 2])
        self.assertEqual(non_tensors["step_normalized_action"].tolist(), ["[Discard A]", "[Play B]"])
        self.assertEqual(non_tensors["uid"].tolist(), [
            "episode-0:rollout:0:turn:0:actor:0",
            "episode-0:rollout:0:turn:1:actor:1",
        ])
        self.assertEqual(
            non_tensors["raw_prompt"][0][0]["content"],
            "Current game state:\nSTATE ZERO",
        )
        self.assertEqual(
            non_tensors["raw_prompt"][1][0]["content"],
            "Current game state:\nSTATE ONE",
        )
        self.assertEqual(meta_info["reward_extra_keys"], [])


if __name__ == "__main__":
    unittest.main()
