import json
import unittest

import mindgames as mg


class StubAgent(mg.Agent):
    def __init__(self, responses, system_prompt: str = "base prompt"):
        super().__init__()
        self.responses = list(responses)
        self.system_prompt = system_prompt
        self.last_message = None

    def __call__(self, observation: str) -> str:
        if not self.responses:
            raise AssertionError(f"No stub responses left for observation: {observation}")
        response = self.responses.pop(0)
        self.last_message = {"role": "assistant", "content": response}
        return response


class TestGoalMemoryAgentWrapper(unittest.TestCase):
    def test_structured_reveal_goal_is_completed_after_matching_action(self):
        payload = {
            "selected_goal_id": "save_p1_slot4",
            "goal_ops": [
                {
                    "op": "upsert_goal",
                    "goal_id": "save_p1_slot4",
                    "reason": "protect likely critical card",
                    "goal": {
                        "goal_type": "save_partner_card",
                        "target": {"entity_type": "card_slot", "player": 1, "slot": 4},
                        "priority": 0.93,
                        "confidence": 0.71,
                        "ttl": 1,
                        "reason": "protect likely critical card",
                        "belief_refs": ["belief_partner1_slot4"],
                        "preconditions": ["info_tokens > 0"],
                        "success_conditions": ["target_card_clued"],
                        "abort_conditions": ["target_slot_shifted_out"],
                    },
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

        turn_output = wrapper.get_last_goal_turn_output()
        self.assertEqual(turn_output["selected_goal_id"], "save_p1_slot4")
        self.assertIn("goal_events", turn_output)

    def test_self_slot_goals_rebase_after_discard(self):
        payload = {
            "selected_goal_id": "safe_self_slot0",
            "goal_ops": [
                {
                    "op": "upsert_goal",
                    "goal_id": "safe_self_slot0",
                    "reason": "fallback discard",
                    "goal": {
                        "goal_type": "safe_discard_fallback",
                        "target": {"entity_type": "card_slot", "player": 0, "slot": 0},
                        "priority": 0.30,
                        "confidence": 0.55,
                        "ttl": 1,
                        "reason": "fallback discard",
                        "belief_refs": [],
                        "preconditions": [],
                        "success_conditions": [],
                        "abort_conditions": [],
                    },
                },
                {
                    "op": "upsert_goal",
                    "goal_id": "play_self_slot2",
                    "reason": "hold immediate play target",
                    "goal": {
                        "goal_type": "play_self_slot",
                        "target": {"entity_type": "card_slot", "player": 0, "slot": 2},
                        "priority": 0.88,
                        "confidence": 0.73,
                        "ttl": 2,
                        "reason": "hold immediate play target",
                        "belief_refs": ["belief_self_slot2"],
                        "preconditions": [],
                        "success_conditions": ["slot_played_successfully"],
                        "abort_conditions": ["slot_shifted_out"],
                    },
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
        self.assertEqual(goals["play_self_slot2"]["target"]["slot"], 1)

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
