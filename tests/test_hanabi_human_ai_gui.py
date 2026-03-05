import unittest

from mindgames.apps.hanabi_human_ai_gui import HanabiHumanAIGame


class TestHanabiHumanAIGUI(unittest.TestCase):
    def _make_session(self) -> HanabiHumanAIGame:
        return HanabiHumanAIGame(
            env_id="Hanabi-v0-train",
            env_kwargs={},
            num_players=2,
            human_players="0",
            llm_agent="scripted:hanabi_discard0",
            seed=0,
            system_prompt="",
            openai_api_key=None,
            openai_base_url=None,
            timeout_s=30.0,
            max_retries=2,
            retry_initial_delay_s=0.0,
            temperature=None,
            top_p=None,
            top_k=None,
            max_tokens=None,
            disable_thinking=False,
            max_auto_steps_per_tick=200,
        )

    def test_start_has_human_turn_and_action_options(self):
        session = self._make_session()
        state = session.start_new_game(seed=0)
        self.assertTrue(state["ok"])
        self.assertFalse(state["done"])
        self.assertTrue(state["is_human_turn"])
        self.assertEqual(state["current_player_id"], 0)
        options = state["action_options"]
        self.assertIsNotNone(options)
        self.assertIn("play_indices", options)
        self.assertIn("discard_indices", options)

    def test_submit_discard_advances_game(self):
        session = self._make_session()
        state = session.start_new_game(seed=0)
        before_steps = state["game_state"]["step_count"]

        result = session.submit_action({"type": "discard", "card_index": 0})
        self.assertTrue(result["ok"], msg=result.get("error"))

        state_after = result["state"]
        self.assertGreaterEqual(state_after["game_state"]["step_count"], before_steps + 1)
        self.assertTrue(state_after["done"] or state_after["is_human_turn"])
        self.assertGreaterEqual(len(state_after["recent_steps"]), 1)

    def test_invalid_card_index_is_rejected(self):
        session = self._make_session()
        session.start_new_game(seed=0)
        result = session.submit_action({"type": "discard", "card_index": 999})
        self.assertFalse(result["ok"])
        self.assertIn("Invalid discard card index", result["error"])

        state = session.get_public_state()
        self.assertTrue(state["is_human_turn"])
        self.assertFalse(state["done"])

    def test_submit_reveal_rank_keeps_selected_target_card_index(self):
        session = self._make_session()
        state = session.start_new_game(seed=0)
        self.assertTrue(state["is_human_turn"])

        reveal_targets = (state.get("action_options") or {}).get("reveal_targets") or []
        self.assertTrue(reveal_targets, "Expected at least one reveal target.")
        target = reveal_targets[0]
        cards = list(target.get("cards") or [])
        self.assertTrue(cards, "Expected target player to have visible cards.")

        # Prefer index 3 to match real GUI reports; fallback to the last visible card.
        chosen = next((c for c in cards if int(c.get("index")) == 3), cards[-1])
        chosen_idx = int(chosen.get("index"))
        chosen_rank = int(chosen.get("rank"))
        target_player = int(target.get("player_id"))

        result = session.submit_action(
            {
                "type": "reveal_rank",
                "target_player": target_player,
                "card_index": chosen_idx,
                "hint_value": str(chosen_rank),
            }
        )
        self.assertTrue(result["ok"], msg=result.get("error"))

        state_after = result["state"]
        steps = list(state_after.get("recent_steps") or [])
        self.assertTrue(steps, "Expected at least one recorded step after action submit.")
        human_reveals = [
            rec
            for rec in steps
            if int(rec.get("player_id")) == 0 and str(rec.get("normalized_action") or "").startswith("[Reveal]")
        ]
        self.assertTrue(human_reveals, "Expected a human reveal action in recent_steps.")
        expected = f"[Reveal] player {target_player} card {chosen_idx} rank {chosen_rank}"
        self.assertEqual(human_reveals[-1]["normalized_action"], expected)

    def test_observation_includes_previous_turn_hint_for_human(self):
        session = self._make_session()
        state = session.start_new_game(seed=0)
        self.assertTrue(state["is_human_turn"])
        session._current_observation = (
            (session._current_observation or "")
            + "\n\nRecent events (oldest -> newest):\nCard 2 from player 0 is white.\n"
        )

        session.step_history.append(
            {
                "step": 999,
                "player_id": 1,
                "actor": "ai",
                "observation": "",
                "action": "[Reveal] player 0 card 2 color white",
                "normalized_action": "[Reveal] player 0 card 2 color white",
                "infer_ms": 123,
                "reasoning": None,
                "step_info": {},
                "done": False,
            }
        )

        state_after = session.get_public_state()
        expected = "Previous turn hint: Player 1 revealed your card 2 color white."
        self.assertEqual(state_after["previous_turn_hint"], expected)
        self.assertIn(expected, state_after["observation"])

    def test_previous_turn_hint_absent_when_last_action_not_reveal(self):
        session = self._make_session()
        state = session.start_new_game(seed=0)
        self.assertTrue(state["is_human_turn"])

        session.step_history.append(
            {
                "step": 999,
                "player_id": 1,
                "actor": "ai",
                "observation": "",
                "action": "[Discard] 0",
                "normalized_action": "[Discard] 0",
                "infer_ms": 50,
                "reasoning": None,
                "step_info": {},
                "done": False,
            }
        )

        state_after = session.get_public_state()
        self.assertIsNone(state_after["previous_turn_hint"])
        self.assertNotIn("Previous turn hint:", state_after["observation"])

    def test_previous_turn_hint_absent_for_unobserved_invalid_reveal_attempt(self):
        session = self._make_session()
        state = session.start_new_game(seed=0)
        self.assertTrue(state["is_human_turn"])

        # No matching reveal message is present in observation (e.g., invalid reveal attempt).
        session.step_history.append(
            {
                "step": 999,
                "player_id": 1,
                "actor": "ai",
                "observation": "",
                "action": "[Reveal] player 0 card 2 color white",
                "normalized_action": "[Reveal] player 0 card 2 color white",
                "infer_ms": 50,
                "reasoning": None,
                "step_info": {},
                "done": False,
            }
        )

        state_after = session.get_public_state()
        self.assertIsNone(state_after["previous_turn_hint"])
        self.assertNotIn("Previous turn hint:", state_after["observation"])


if __name__ == "__main__":
    unittest.main()
