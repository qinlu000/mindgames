import json
import unittest

import mindgames as mg


class StubAgent(mg.Agent):
    def __init__(self, responses, system_prompt: str = "base prompt"):
        super().__init__()
        self.responses = list(responses)
        self.system_prompt = system_prompt
        self.last_message = None
        self.last_system_prompt = None

    def __call__(self, observation: str) -> str:
        if not self.responses:
            raise AssertionError(f"No stub responses left for observation: {observation}")
        self.last_system_prompt = self.system_prompt
        response = self.responses.pop(0)
        self.last_message = {"role": "assistant", "content": response}
        return response


class TestGoalMemoryAgentWrapper(unittest.TestCase):
    def test_selected_reveal_goal_is_completed_after_matching_action(self):
        payload = {
            "selected_goal_id": "save_p1_slot4",
            "goal_ops": [
                {
                    "op": "set",
                    "goal_id": "save_p1_slot4",
                    "goal": "save partner critical card",
                    "target": "player1_slot4",
                    "priority": "high",
                    "ttl": 1,
                }
            ],
            "action": "[Reveal] player 1 card 4 rank 5",
        }
        wrapper = mg.agents.GoalMemoryAgentWrapper(StubAgent([json.dumps(payload)]))
        wrapper.reset_episode(episode_id=12, player_id=0)
        wrapper.set_turn_context(episode_id=12, turn_id=0, player_id=0)

        action = wrapper("You are player 0. Hanabi turn.")
        self.assertEqual(action, "[Reveal] player 1 card 4 rank 5")

        wrapper.record_step_result(action=action, normalized_action=action, step_info={}, done=False)
        snapshot = wrapper.get_goal_memory_snapshot()
        goal = next(item for item in snapshot["goals"] if item["goal_id"] == "save_p1_slot4")
        self.assertEqual(goal["status"], "completed")
        self.assertEqual(goal["goal"], "save partner critical card")
        self.assertEqual(goal["target"], "player1_slot4")
        self.assertEqual(goal["priority"], "high")

        turn_output = wrapper.get_last_goal_turn_output()
        self.assertEqual(turn_output["selected_goal_id"], "save_p1_slot4")
        self.assertIn("goal_events", turn_output)

    def test_self_slot_goals_rebase_after_discard(self):
        payload = {
            "selected_goal_id": "safe_self_slot0",
            "goal_ops": [
                {
                    "op": "set",
                    "goal_id": "safe_self_slot0",
                    "goal": "discard fallback card if nothing better appears",
                    "target": "self_slot0",
                    "priority": "low",
                    "ttl": 1,
                },
                {
                    "op": "set",
                    "goal_id": "play_self_slot2",
                    "goal": "play the likely good card soon",
                    "target": "self_slot2",
                    "priority": "high",
                    "ttl": 2,
                },
            ],
            "action": "[Discard] 0",
        }
        wrapper = mg.agents.GoalMemoryAgentWrapper(StubAgent([json.dumps(payload)]))
        wrapper.reset_episode(episode_id=3, player_id=0)
        wrapper.set_turn_context(episode_id=3, turn_id=0, player_id=0)

        action = wrapper("You are player 0. Hanabi turn.")
        self.assertEqual(action, "[Discard] 0")
        wrapper.record_step_result(action=action, normalized_action=action, step_info={}, done=False)

        snapshot = wrapper.get_goal_memory_snapshot()
        goals = {item["goal_id"]: item for item in snapshot["goals"]}
        self.assertEqual(goals["safe_self_slot0"]["status"], "completed")
        self.assertEqual(goals["play_self_slot2"]["status"], "active")
        self.assertEqual(goals["play_self_slot2"]["target"], "self_slot1")

    def test_goal_memory_prompt_rewrites_action_only_contract(self):
        payload = {"selected_goal_id": None, "goal_ops": [], "action": "[Discard] 0"}
        wrapper = mg.agents.GoalMemoryAgentWrapper(StubAgent([json.dumps(payload)]))
        wrapper.reset_episode(episode_id=21, player_id=0)
        wrapper.set_turn_context(episode_id=21, turn_id=0, player_id=0)

        observation = (
            "You are Player 0 in a 2-player Hanabi game.\n"
            "You have 3 action types: Play, Discard, Reveal. Output EXACTLY ONE action, nothing else.\n"
            "Current state..."
        )
        wrapper(observation)

        prompt = wrapper.get_last_goal_prompt()
        self.assertIsNotNone(prompt)
        self.assertIn("Return EXACTLY ONE JSON object", prompt)
        self.assertIn("goals you explicitly set in previous turns", prompt)
        self.assertIn("Your active goals from previous turns", prompt)
        self.assertIn('"op": "set|remove"', prompt)
        self.assertIn('"priority": "high|medium|low"', prompt)
        self.assertNotIn("Output EXACTLY ONE action, nothing else.", prompt)
        self.assertIn("The `action` field must contain exactly one legal Hanabi action.", prompt)
        self.assertIn("JSON object and nothing else", wrapper.agent.last_system_prompt)
        self.assertNotIn("Output EXACTLY ONE valid action", wrapper.agent.last_system_prompt)

    def test_plain_action_falls_back_without_goal_ops(self):
        wrapper = mg.agents.GoalMemoryAgentWrapper(StubAgent(["[Discard] 0"]))
        wrapper.reset_episode(episode_id=9, player_id=0)
        wrapper.set_turn_context(episode_id=9, turn_id=0, player_id=0)

        action = wrapper("You are player 0. Hanabi turn.")
        self.assertEqual(action, "[Discard] 0")
        turn_output = wrapper.get_last_goal_turn_output()
        self.assertEqual(turn_output["parse_error"], "json_not_found")
        self.assertEqual(turn_output["goal_ops"], [])
        self.assertEqual(wrapper.get_goal_memory_snapshot()["goals"], [])


if __name__ == "__main__":
    unittest.main()
