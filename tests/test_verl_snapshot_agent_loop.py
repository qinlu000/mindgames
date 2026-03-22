import asyncio
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


def _ensure_pkg_importable() -> None:
    project_root = Path(__file__).resolve().parents[1]
    repo_root = project_root.parent
    sys.path.insert(0, str(project_root))
    sys.path.insert(0, str(repo_root))


_ensure_pkg_importable()


class _FakeTokenizer:
    def __init__(self, decode_map):
        self.decode_map = decode_map

    def decode(self, token_ids, skip_special_tokens=True):
        del skip_special_tokens
        return self.decode_map[tuple(token_ids)]


class _FakeTokenOutput:
    def __init__(self, token_ids, log_probs=None, num_preempted=0):
        self.token_ids = token_ids
        self.log_probs = log_probs
        self.num_preempted = num_preempted


class _FakeServerManager:
    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.calls = []

    async def generate(self, **kwargs):
        self.calls.append(kwargs)
        return self.outputs.pop(0)


class _DummyLoopMixin:
    def __init__(self, *args, **kwargs):
        self.prompt_messages = []
        super().__init__(*args, **kwargs)

    async def apply_chat_template(
        self,
        messages,
        tools=None,
        images=None,
        videos=None,
        remove_system_prompt=False,
    ):
        del tools, images, videos, remove_system_prompt
        self.prompt_messages.append(messages)
        return [len(self.prompt_messages), len(messages[-1]["content"])]


class TestMindGamesSnapshotAgentLoop(unittest.TestCase):
    def test_run_rolls_episode_and_exports_all_step_snapshots(self):
        from mindgames.training.verl_snapshot_agent_loop import MindGamesSnapshotEpisodeAgentLoop

        class DummyLoop(_DummyLoopMixin, MindGamesSnapshotEpisodeAgentLoop):
            pass

        step0 = SimpleNamespace(
            env_id="MiniHanabi-v0-train",
            observation="You are Player 0.\n\nSTATE ZERO",
            actor_id=0,
            turn_index=0,
        )
        step1 = SimpleNamespace(
            env_id="MiniHanabi-v0-train",
            observation="You are Player 1.\n\nSTATE ONE",
            actor_id=1,
            turn_index=1,
        )

        transition0 = SimpleNamespace(normalized_action='[Discard A]', done=False, terminal_reward=None)
        transition1 = SimpleNamespace(normalized_action='[Play B]', done=True, terminal_reward=0.75)

        class FakeEpisode:
            def __init__(self):
                self._steps = [step0, step1]
                self._index = 0
                self.closed = False

            def has_active_step(self):
                return self._index < len(self._steps)

            def current_step(self):
                return self._steps[self._index]

            def step(self, _action):
                if self._index == 0:
                    self._index += 1
                    return transition0
                self._index += 1
                return transition1

            def close(self):
                self.closed = True
                return None

        fake_episode = FakeEpisode()
        decode_map = {
            (11,): '[Discard A]',
            (22,): '[Play B]',
        }
        loop = DummyLoop(
            trainer_config=SimpleNamespace(
                config=SimpleNamespace(
                    actor_rollout_ref=SimpleNamespace(
                        rollout=SimpleNamespace(prompt_length=128, response_length=32)
                    )
                )
            ),
            server_manager=_FakeServerManager(
                [
                    _FakeTokenOutput([11], log_probs=[-0.1]),
                    _FakeTokenOutput([22], log_probs=[-0.2]),
                ]
            ),
            tokenizer=_FakeTokenizer(decode_map),
            processor=None,
            dataset_cls=None,
            data_config=SimpleNamespace(config={}),
            selection_strategy='last',
        )

        async def run_loop():
            with patch('mindgames.training.episode.MindGamesEpisode.create', return_value=fake_episode):
                return await loop.run(
                    sampling_params={'temperature': 1.0},
                    extra_info={
                        'interaction_kwargs': {
                            'name': 'mindgames',
                            'game': 'mini_hanabi',
                            'seed': 0,
                            'env_id': 'MiniHanabi-v0-train',
                            'max_steps': 2,
                            'reward_player': -1,
                        }
                    },
                )

        output = asyncio.run(run_loop())

        self.assertEqual(output.reward_score, 0.75)
        self.assertEqual(output.response_ids, [22])
        self.assertEqual(output.response_mask, [1])
        self.assertEqual(output.response_logprobs, [-0.2])
        self.assertEqual(output.extra_fields['episode_steps'], 2)
        self.assertEqual(len(output.extra_fields['episode_step_data']), 2)
        self.assertEqual(output.extra_fields['episode_step_data'][0]['turn_index'], 0)
        self.assertEqual(output.extra_fields['episode_step_data'][0]['normalized_action'], '[Discard A]')
        self.assertEqual(output.extra_fields['episode_step_data'][1]['turn_index'], 1)
        self.assertEqual(output.extra_fields['episode_step_data'][1]['normalized_action'], '[Play B]')
        self.assertEqual(output.extra_fields['selected_turn_index'], 1)
        self.assertEqual(output.extra_fields['selected_actor_id'], 1)
        self.assertEqual(output.extra_fields['selected_normalized_action'], '[Play B]')
        self.assertEqual(output.extra_fields['turn_scores'], [0.75])
        self.assertEqual(len(loop.prompt_messages), 2)
        self.assertIn('STATE ZERO', loop.prompt_messages[0][-1]['content'])
        self.assertIn('STATE ONE', loop.prompt_messages[1][-1]['content'])
        self.assertNotIn('STATE ZERO', loop.prompt_messages[1][-1]['content'])
        self.assertTrue(fake_episode.closed)


if __name__ == '__main__':
    unittest.main()
