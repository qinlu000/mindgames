import re

from mindgames.core import GAME_ID, ObservationType
from mindgames.envs.Hanabi.env import HanabiEnv, Suit


class HanabiStandardEnv(HanabiEnv):
    """
    Hanabi with standard hinting: color/rank hints apply to all matching cards
    in the target player's hand.
    """

    def _initial_prompt(self, player_id: int, game_state: dict) -> str:
        return (
            f"You are Player {player_id} in a {self.state.num_players}-player Hanabi game.\n"
            "You can see other players' cards but NOT your own. Work as a team to build fireworks.\n\n"
            "Goal: For each color, play ranks 1→5 in order. Wrong plays cost 1 fuse token.\n"
            "Colors are independent; you may play the next required rank of any color at any time.\n"
            "You have 3 action types: Play, Discard, Reveal. Output EXACTLY ONE action, nothing else.\n"
            "Valid formats (case-insensitive):\n"
            "- [Play] X\n"
            "- [Discard] X\n"
            "- [Reveal] player N color C\n"
            "- [Reveal] player N rank R\n\n"
            "Index rules:\n"
            "- X is a 0-based card index from YOUR hand for [Play]/[Discard].\n\n"
            "Reveal rules in this env (standard Hanabi):\n"
            "- You must reveal about another player (not yourself).\n"
            "- Reveal costs 1 info token; if you have 0 info tokens, [Reveal] is invalid.\n"
            "- Provide exactly one hint type: either 'color C' OR 'rank R' (not both).\n"
            "- The hint applies to ALL cards in that player's hand matching the hint.\n"
            "- The hint must be truthful about at least one card.\n"
            "- Colors: white, yellow, green, blue, red. Ranks: 1-5.\n\n"
            "Discard rule: if info tokens < 8, discarding gains +1 info token.\n"
            "Play rule: if you successfully play a rank-5 card and info tokens < 8, gain +1 info token.\n"
            "Game ends if fuse tokens reach 0, all fireworks reach 5, or the deck is exhausted and the final round ends."
        )

    @staticmethod
    def _parse_hint(action: str):
        player = re.findall(r"\bplayer\s+(\d+)\b", action, flags=re.IGNORECASE)
        color = re.findall(r"\bcolor\s+(white|yellow|green|blue|red)\b", action, flags=re.IGNORECASE)
        rank = re.findall(r"\brank\s+([1-5])\b", action, flags=re.IGNORECASE)
        card_index = re.findall(r"\bcard\s+(\d+)\b", action, flags=re.IGNORECASE)
        color = [c.lower() for c in color]
        return player, color, rank, card_index

    def _handle_reveal(self, action: str) -> None:
        if self.state.game_state["info_tokens"] == 0:
            reason = "Player attempted to give a hint without having any info tokens."
            self.state.set_invalid_move(reason=reason)
            self.state.add_observation(
                from_id=GAME_ID,
                to_id=self.state.current_player_id,
                message="Invalid hint: you have no info tokens remaining.",
                observation_type=ObservationType.GAME_MESSAGE,
            )
            return

        player, color, rank, card_index = self._parse_hint(action)
        if card_index:
            reason = "Standard Hanabi hints do not target a specific card index."
            self.state.set_invalid_move(reason=reason)
            self.state.add_observation(
                from_id=GAME_ID,
                to_id=self.state.current_player_id,
                message="Invalid hint. Use '[Reveal] player N color C' or '[Reveal] player N rank R'.",
                observation_type=ObservationType.GAME_MESSAGE,
            )
            return

        if player == [] or (color == [] and rank == []):
            reason = "The player provided an incomplete hint."
            self.state.set_invalid_move(reason=reason)
            self.state.add_observation(
                from_id=GAME_ID,
                to_id=self.state.current_player_id,
                message="Invalid hint. Provide a player and exactly one hint type: color or rank.",
                observation_type=ObservationType.GAME_MESSAGE,
            )
            return

        if color and rank:
            reason = "The player provided both a color hint and a rank hint."
            self.state.set_invalid_move(reason=reason)
            self.state.add_observation(
                from_id=GAME_ID,
                to_id=self.state.current_player_id,
                message="Provide exactly one hint type: either 'color <C>' OR 'rank <R>', not both.",
                observation_type=ObservationType.GAME_MESSAGE,
            )
            return

        target_pid = int(player[0])
        if target_pid == self.state.current_player_id:
            reason = "The player attempts to reveal information about their own cards."
            self.state.set_invalid_move(reason=reason)
            self.state.add_observation(
                from_id=GAME_ID,
                to_id=self.state.current_player_id,
                message="You attempted to reveal information about your own cards. This is not allowed.",
                observation_type=ObservationType.GAME_MESSAGE,
            )
            return

        if target_pid < 0 or target_pid >= self.num_players:
            reason = "The player attempts to reveal information about a non-existing teammate."
            self.state.set_invalid_move(reason=reason)
            self.state.add_observation(
                from_id=GAME_ID,
                to_id=self.state.current_player_id,
                message=f"You attempted to reveal information about a non-existing teammate. "
                        f"Please consider teammates between 0 and {self.num_players - 1} (Including).",
                observation_type=ObservationType.GAME_MESSAGE,
            )
            return

        target_hand = self.state.game_state["player_hands"][target_pid]
        if color:
            try:
                Suit(color[0])
            except ValueError:
                reason = "The player provided a color that is not in the game."
                self.state.set_invalid_move(reason=reason)
                self.state.add_observation(
                    from_id=GAME_ID,
                    to_id=self.state.current_player_id,
                    message=f"You provided an invalid color. Valid colors are 'white', 'yellow', "
                            f"'green', 'red' and 'blue'. You tried: '{color[0]}'.",
                    observation_type=ObservationType.GAME_MESSAGE,
                )
                return
            matched = [i for i, card in enumerate(target_hand) if card.suit.value == color[0]]
            if not matched:
                reason = "The player provided an untruthful color hint."
                self.state.set_invalid_move(reason=reason)
                self.state.add_observation(
                    from_id=GAME_ID,
                    to_id=self.state.current_player_id,
                    message=f"Invalid hint: Player {target_pid} has no '{color[0]}' cards.",
                    observation_type=ObservationType.GAME_MESSAGE,
                )
                return
            hint = self._format_hint(
                target_pid,
                matched,
                singular_phrase=f"is {color[0]}",
                plural_phrase=f"are {color[0]}",
            )
        else:
            try:
                rank_value = int(rank[0])
            except ValueError:
                rank_value = -1
            if rank_value < 1 or rank_value > 5:
                reason = "The player provided an invalid rank."
                self.state.set_invalid_move(reason=reason)
                self.state.add_observation(
                    from_id=GAME_ID,
                    to_id=self.state.current_player_id,
                    message=f"You provided an invalid rank. Valid ranks are between 1 and 5 (including). "
                            f"You provided: '{rank[0]}'.",
                    observation_type=ObservationType.GAME_MESSAGE,
                )
                return
            matched = [i for i, card in enumerate(target_hand) if card.rank == rank_value]
            if not matched:
                reason = "The player provided an untruthful rank hint."
                self.state.set_invalid_move(reason=reason)
                self.state.add_observation(
                    from_id=GAME_ID,
                    to_id=self.state.current_player_id,
                    message=f"Invalid hint: Player {target_pid} has no rank {rank_value} cards.",
                    observation_type=ObservationType.GAME_MESSAGE,
                )
                return
            hint = self._format_hint(
                target_pid,
                matched,
                singular_phrase=f"has rank {rank_value}",
                plural_phrase=f"have rank {rank_value}",
            )

        self.state.game_state["info_tokens"] -= 1
        self.state.add_observation(
            from_id=self.state.current_player_id,
            to_id=-1,
            message=hint,
            observation_type=ObservationType.GAME_MESSAGE,
        )

    @staticmethod
    def _format_hint(
        target_pid: int,
        card_indices: list[int],
        *,
        singular_phrase: str,
        plural_phrase: str,
    ) -> str:
        indices = ", ".join(str(i) for i in sorted(card_indices))
        if len(card_indices) == 1:
            return f"Card {indices} from player {target_pid} {singular_phrase}."
        return f"Cards {indices} from player {target_pid} {plural_phrase}."
