from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Optional, Tuple

from mindgames.core import Env, GAME_ID, Info, ObservationType
from mindgames.state import TeamMultiPlayerState


SLOT_LABELS = ("A", "B")
COLORS = ("Red", "Blue", "Green")
RANKS = (1, 2, 3)
SLOT_TOKEN_TO_INDEX = {"a": 0, "b": 1, "0": 0, "1": 1}


@dataclass(eq=True, frozen=True)
class Card:
    color: str
    rank: int

    def __post_init__(self) -> None:
        if self.color not in COLORS:
            raise ValueError(f"Invalid color: {self.color!r}")
        if self.rank not in RANKS:
            raise ValueError(f"Invalid rank: {self.rank!r}")

    def short(self) -> str:
        return f"{self.color}{self.rank}"

    def __str__(self) -> str:
        return f"{self.color} {self.rank}"


@dataclass(eq=True)
class SlotKnowledge:
    known_color: Optional[str] = None
    known_rank: Optional[int] = None
    not_colors: set[str] = field(default_factory=set)
    not_ranks: set[int] = field(default_factory=set)
    last_touched_turn: Optional[int] = None


class MiniHanabiEnv(Env):
    def __init__(self, info_tokens: int = 2, fuse_tokens: int = 2, max_turns: int = 12):
        self.num_players = 2
        self.hand_size = 2
        self.max_info_tokens = int(info_tokens)
        self.max_fuse_tokens = int(fuse_tokens)
        self.max_turns = int(max_turns)
        if self.max_info_tokens <= 0:
            raise ValueError("info_tokens must be positive")
        if self.max_fuse_tokens <= 0:
            raise ValueError("fuse_tokens must be positive")
        if self.max_turns <= 0:
            raise ValueError("max_turns must be positive")

    def reset(self, num_players: int, seed: Optional[int] = None):
        if num_players != 2:
            raise ValueError(f"MiniHanabi requires exactly 2 players, got {num_players}")

        self.state = TeamMultiPlayerState(
            num_players=num_players,
            max_turns=self.max_turns,
            seed=seed,
            error_allowance=1,
        )
        self._rng = random.Random(seed)

        deck = self._generate_deck()
        self._rng.shuffle(deck)

        player_hands = {
            pid: [self._draw_card(deck) for _ in range(self.hand_size)] for pid in range(num_players)
        }
        knowledge = {
            pid: [SlotKnowledge() for _ in range(self.hand_size)] for pid in range(num_players)
        }

        game_state = {
            "info_tokens": self.max_info_tokens,
            "fuse_tokens": self.max_fuse_tokens,
            "fireworks": {color: 0 for color in COLORS},
            "deck": deck,
            "deck_size": len(deck),
            "player_hands": player_hands,
            "knowledge": knowledge,
            "discard_pile": [],
        }
        self.state.reset(game_state=game_state, player_prompt_function=self._prompt)
        self._emit_board()

    def step(self, action: str) -> Tuple[bool, Info]:
        if self.state.done:
            return self.state.step(rotate_player=False)

        acting_player_id = self.state.current_player_id
        score_before = self._calculate_score()
        self.state.add_observation(
            from_id=acting_player_id,
            to_id=acting_player_id,
            message=action,
            observation_type=ObservationType.PLAYER_ACTION,
        )

        action_kind, action_value, invalid_message = self._parse_action(action)
        if action_kind == "play":
            self._handle_play(slot_index=action_value)
        elif action_kind == "discard":
            self._handle_discard(slot_index=action_value)
        elif action_kind == "hint_color":
            self._handle_hint(hint_type="color", value=action_value)
        elif action_kind == "hint_rank":
            self._handle_hint(hint_type="rank", value=action_value)
        else:
            self._invalidate(
                reason="Invalid action format.",
                message=invalid_message,
            )

        self._set_step_info(acting_player_id=acting_player_id, score_before=score_before)

        if self.state.done:
            return self.state.step(rotate_player=False)

        if self.state.game_info[acting_player_id]["invalid_move"]:
            self.state.add_observation(
                from_id=GAME_ID,
                to_id=-1,
                message=(
                    f"Player {acting_player_id} made too many invalid attempts in a row. "
                    "The turn is skipped."
                ),
                observation_type=ObservationType.GAME_MESSAGE,
            )
            self.state.game_info[acting_player_id]["invalid_move"] = False
            self.state.made_invalid_move = False
            self.state.error_count = 0
            self._maybe_end_on_turn_cap()
            done, info = self.state.step(rotate_player=False)
            if not done:
                self._rotate_players()
            return done, info

        if self.state.made_invalid_move:
            return self.state.step(rotate_player=False)

        self._maybe_end_on_turn_cap()
        done, info = self.state.step(rotate_player=False)
        if not done:
            self._rotate_players()
        return done, info

    def get_board_str(self, for_player_id: Optional[int] = None) -> str:
        if for_player_id is None:
            for_player_id = self.state.current_player_id
        partner_id = self._partner_id(for_player_id)
        lines = [
            "MiniHanabi-v0",
            f"Turn: {self.state.turn}/{self.max_turns}",
            f"Current player: Player {self.state.current_player_id}",
            f"Score: {self._calculate_score()}/9",
            (
                "Resources: "
                f"info {self.state.game_state['info_tokens']}/{self.max_info_tokens} | "
                f"fuse {self.state.game_state['fuse_tokens']}/{self.max_fuse_tokens} | "
                f"deck {len(self.state.game_state['deck'])}"
            ),
            "Fireworks: "
            + ", ".join(f"{color}:{self.state.game_state['fireworks'][color]}" for color in COLORS),
            "Discards: " + self._format_discard_pile(),
            "",
            f"Visible partner hand (Player {partner_id}):",
        ]
        for slot_index, slot_label in enumerate(SLOT_LABELS):
            card = self.state.game_state["player_hands"][partner_id][slot_index]
            if card is None:
                lines.append(f"- Slot {slot_label}: empty")
            else:
                lines.append(f"- Slot {slot_label}: {card}")

        lines.extend(["", "Your hand knowledge:"])
        for slot_index, slot_label in enumerate(SLOT_LABELS):
            card = self.state.game_state["player_hands"][for_player_id][slot_index]
            knowledge = self.state.game_state["knowledge"][for_player_id][slot_index]
            lines.append(f"- Slot {slot_label}: {self._format_knowledge(card=card, knowledge=knowledge)}")

        lines.extend(
            [
                "",
                "Valid actions:",
                "- [Play A], [Play B], [Discard A], [Discard B]",
                "- [Hint Color Red], [Hint Color Blue], [Hint Color Green]",
                "- [Hint Rank 1], [Hint Rank 2], [Hint Rank 3]",
            ]
        )
        return "\n".join(lines)

    def _prompt(self, player_id: int, game_state: dict) -> str:
        return (
            f"You are Player {player_id} in MiniHanabi-v0, a 2-player cooperative hidden-information game.\n"
            "Goal: build Red, Blue, and Green fireworks from rank 1 to rank 3.\n"
            "You can see your partner's cards but not your own.\n"
            "Your hand has two fixed slots, A and B; slots never shift. After a play or discard, any replacement card goes into the same slot. If the deck is empty, that slot becomes empty.\n"
            "Hints are public, truthful, and standard-style: a color hint touches all partner cards of that color; a rank hint touches all partner cards of that rank.\n"
            "Touched slots gain positive information, while untouched occupied slots learn that they are not the hinted color or rank.\n"
            "You start with 2 info tokens and 2 fuse tokens. Giving a hint costs 1 info token. Discarding regains 1 info token if below cap. Successfully playing a rank-3 card regains 1 info token if below cap.\n"
            "A wrong play loses 1 fuse token and discards the card. The game ends when score reaches 9, fuse reaches 0, or 12 turns are consumed.\n"
            "Output exactly one action and nothing else. Valid formats are [Play A], [Discard B], [Hint Color Red], [Hint Rank 2]. Slot aliases 0/1 are also accepted."
        )

    def _emit_board(self) -> None:
        current_player_id = self.state.current_player_id
        self.state.add_observation(
            to_id=current_player_id,
            message=self.get_board_str(for_player_id=current_player_id),
            observation_type=ObservationType.GAME_BOARD,
        )

    def _rotate_players(self) -> None:
        self.state.manually_set_current_player_id((self.state.current_player_id + 1) % self.num_players)
        if not self.state.made_invalid_move:
            self._emit_board()

    def _parse_action(self, action: str) -> tuple[str, Optional[object], str]:
        tokens = self._tokenize_action(action)
        if not tokens:
            return "invalid", None, self._invalid_action_message(
                "No action detected.",
                "Use exactly one action: [Play A], [Play B], [Discard A], [Discard B], "
                "[Hint Color Red/Blue/Green], or [Hint Rank 1/2/3].",
            )

        verb = tokens[0].lower()
        if verb in {"play", "discard"}:
            if len(tokens) != 2:
                return "invalid", None, self._invalid_action_message(
                    f"{verb.capitalize()} must name exactly one slot.",
                    "Use [Play A], [Play B], [Discard A], or [Discard B]. Slot aliases 0/1 are also accepted.",
                )
            slot_index = self._parse_slot(tokens[1])
            if slot_index is not None:
                return verb, slot_index, ""
            return "invalid", None, self._invalid_action_message(
                f"Unknown slot {tokens[1]!r}.",
                "Use slot A or B. Slot aliases 0/1 are also accepted.",
            )

        if verb == "hint":
            if len(tokens) != 3:
                return "invalid", None, self._invalid_action_message(
                    "Hint actions need both a type and a value.",
                    "Use [Hint Color Red/Blue/Green] or [Hint Rank 1/2/3].",
                )
            hint_kind = tokens[1].lower()
            if hint_kind == "color":
                color = self._canonical_color(tokens[2])
                if color is not None:
                    return "hint_color", color, ""
                return "invalid", None, self._invalid_action_message(
                    f"Unknown color {tokens[2]!r}.",
                    "Valid colors are Red, Blue, and Green.",
                )
            if hint_kind == "rank":
                rank = self._canonical_rank(tokens[2])
                if rank is not None:
                    return "hint_rank", rank, ""
                return "invalid", None, self._invalid_action_message(
                    f"Unknown rank {tokens[2]!r}.",
                    "Valid ranks are 1, 2, and 3.",
                )
            return "invalid", None, self._invalid_action_message(
                f"Unknown hint type {tokens[1]!r}.",
                "Use [Hint Color Red/Blue/Green] or [Hint Rank 1/2/3].",
            )

        return "invalid", None, self._invalid_action_message(
            f"Unknown action verb {tokens[0]!r}.",
            "Use [Play ...], [Discard ...], or [Hint ...].",
        )

    def _tokenize_action(self, action: str) -> list[str]:
        text = " ".join(str(action).strip().split())
        if not text:
            return []

        if text.startswith("[") and "]" in text:
            close_index = text.find("]")
            inside = text[1:close_index].strip()
            trailing = text[close_index + 1 :].strip()
            text = inside if not trailing else f"{inside} {trailing}"

        tokens: list[str] = []
        for raw_token in text.split():
            token = raw_token.strip().strip("[]").strip(",")
            if token.endswith(":"):
                token = token[:-1]
            if token:
                tokens.append(token)
        return tokens

    def _invalid_action_message(self, summary: str, guidance: str) -> str:
        return f"{summary} {guidance}"

    def _parse_slot(self, token: str) -> Optional[int]:
        return SLOT_TOKEN_TO_INDEX.get(token.lower())

    def _canonical_color(self, token: str) -> Optional[str]:
        normalized = token.lower()
        for color in COLORS:
            if color.lower() == normalized:
                return color
        return None

    def _canonical_rank(self, token: str) -> Optional[int]:
        try:
            rank = int(token)
        except ValueError:
            return None
        return rank if rank in RANKS else None

    def _handle_play(self, slot_index: int) -> None:
        acting_player_id = self.state.current_player_id
        card = self.state.game_state["player_hands"][acting_player_id][slot_index]
        if card is None:
            self._invalidate(
                reason="Tried to play an empty slot.",
                message=f"Slot {SLOT_LABELS[slot_index]} is empty, so it cannot be played.",
            )
            return

        expected_rank = self.state.game_state["fireworks"][card.color] + 1
        slot_label = SLOT_LABELS[slot_index]
        if card.rank == expected_rank:
            self.state.game_state["fireworks"][card.color] += 1
            message = f"Player {acting_player_id} plays slot {slot_label}. It was {card} and the play succeeds."
            if card.rank == max(RANKS) and self.state.game_state["info_tokens"] < self.max_info_tokens:
                self.state.game_state["info_tokens"] += 1
                message += " Completing a color restores 1 info token."
            self.state.add_observation(
                from_id=GAME_ID,
                to_id=-1,
                message=message,
                observation_type=ObservationType.GAME_MESSAGE,
            )
        else:
            self.state.game_state["fuse_tokens"] -= 1
            self.state.game_state["discard_pile"].append(card)
            self.state.add_observation(
                from_id=GAME_ID,
                to_id=-1,
                message=(
                    f"Player {acting_player_id} plays slot {slot_label}. It was {card} and the play fails. "
                    f"Lose 1 fuse token; {self.state.game_state['fuse_tokens']} remain."
                ),
                observation_type=ObservationType.GAME_MESSAGE,
            )

        self._refill_slot(player_id=acting_player_id, slot_index=slot_index)
        self._check_game_end()

    def _handle_discard(self, slot_index: int) -> None:
        acting_player_id = self.state.current_player_id
        card = self.state.game_state["player_hands"][acting_player_id][slot_index]
        if card is None:
            self._invalidate(
                reason="Tried to discard an empty slot.",
                message=f"Slot {SLOT_LABELS[slot_index]} is empty, so it cannot be discarded.",
            )
            return

        self.state.game_state["discard_pile"].append(card)
        slot_label = SLOT_LABELS[slot_index]
        message = f"Player {acting_player_id} discards slot {slot_label}. It was {card}."
        if self.state.game_state["info_tokens"] < self.max_info_tokens:
            self.state.game_state["info_tokens"] += 1
            message += " The team regains 1 info token."
        self.state.add_observation(
            from_id=GAME_ID,
            to_id=-1,
            message=message,
            observation_type=ObservationType.GAME_MESSAGE,
        )

        self._refill_slot(player_id=acting_player_id, slot_index=slot_index)
        self._check_game_end()

    def _handle_hint(self, hint_type: str, value: object) -> None:
        if self.state.game_state["info_tokens"] <= 0:
            self._invalidate(
                reason="Tried to hint with no info tokens.",
                message="You cannot give a hint when info tokens are at 0.",
            )
            return

        acting_player_id = self.state.current_player_id
        target_player_id = self._partner_id(acting_player_id)
        target_hand = self.state.game_state["player_hands"][target_player_id]
        target_knowledge = self.state.game_state["knowledge"][target_player_id]

        occupied_slots = [slot_index for slot_index, card in enumerate(target_hand) if card is not None]
        if hint_type == "color":
            touched_slots = [slot_index for slot_index in occupied_slots if target_hand[slot_index].color == value]
        else:
            touched_slots = [slot_index for slot_index in occupied_slots if target_hand[slot_index].rank == value]

        if not touched_slots:
            descriptor = f"{hint_type} {value}"
            self._invalidate(
                reason="Hint must touch at least one occupied partner slot.",
                message=f"Invalid hint: no partner card matches {descriptor}.",
            )
            return

        for slot_index in occupied_slots:
            knowledge = target_knowledge[slot_index]
            if slot_index in touched_slots:
                if hint_type == "color":
                    knowledge.known_color = value
                    knowledge.not_colors.discard(value)
                else:
                    knowledge.known_rank = value
                    knowledge.not_ranks.discard(value)
                knowledge.last_touched_turn = self.state.turn + 1
            else:
                if hint_type == "color":
                    knowledge.not_colors.add(value)
                else:
                    knowledge.not_ranks.add(value)

        self.state.game_state["info_tokens"] -= 1
        touched_labels = ", ".join(SLOT_LABELS[idx] for idx in touched_slots)
        if hint_type == "color":
            public_message = (
                f"Player {acting_player_id} hints Color {value} to Player {target_player_id}, touching slot"
                f"{'s' if len(touched_slots) != 1 else ''} {touched_labels}."
            )
        else:
            public_message = (
                f"Player {acting_player_id} hints Rank {value} to Player {target_player_id}, touching slot"
                f"{'s' if len(touched_slots) != 1 else ''} {touched_labels}."
            )
        self.state.add_observation(
            from_id=GAME_ID,
            to_id=-1,
            message=public_message,
            observation_type=ObservationType.GAME_MESSAGE,
        )
        self._check_game_end()

    def _refill_slot(self, player_id: int, slot_index: int) -> None:
        new_card = self._draw_card(self.state.game_state["deck"])
        self.state.game_state["player_hands"][player_id][slot_index] = new_card
        self.state.game_state["knowledge"][player_id][slot_index] = SlotKnowledge()
        self.state.game_state["deck_size"] = len(self.state.game_state["deck"])

    def _draw_card(self, deck: list[Card]) -> Optional[Card]:
        if not deck:
            return None
        return deck.pop()

    def _check_game_end(self) -> None:
        if self.state.game_state["fuse_tokens"] <= 0:
            self._finish_game(reason="The team ran out of fuse tokens.", cooperative_win=False)
            return

        if self._calculate_score() == 9:
            self._finish_game(reason="The team completed all fireworks.", cooperative_win=True)

    def _maybe_end_on_turn_cap(self) -> None:
        if not self.state.done and self.state.turn + 1 >= self.max_turns:
            self._finish_game(reason=f"Turn cap reached ({self.max_turns}).", cooperative_win=False)

    def _finish_game(self, reason: str, cooperative_win: bool) -> None:
        if self.state.done:
            return
        if cooperative_win:
            self.state.set_winners(player_ids=list(range(self.num_players)), reason=reason)
        else:
            self.state.set_draw(reason=reason)
        score = float(self._calculate_score())
        self.state.rewards = {pid: score for pid in range(self.num_players)}

    def _invalidate(self, reason: str, message: str) -> None:
        self.state.set_invalid_move(reason=reason)
        self.state.add_observation(
            from_id=GAME_ID,
            to_id=self.state.current_player_id,
            message=message,
            observation_type=ObservationType.GAME_MESSAGE,
        )

    def _set_step_info(self, acting_player_id: int, score_before: int) -> None:
        score_after = self._calculate_score()
        self.state.step_info["acting_player_id"] = int(acting_player_id)
        self.state.step_info["score_before"] = int(score_before)
        self.state.step_info["score_after"] = int(score_after)
        self.state.step_info["score_delta"] = int(score_after - score_before)

    def _calculate_score(self) -> int:
        return sum(int(value) for value in self.state.game_state["fireworks"].values())

    def _format_discard_pile(self) -> str:
        discard_pile = self.state.game_state["discard_pile"]
        if not discard_pile:
            return "(empty)"
        return ", ".join(card.short() for card in discard_pile)

    def _format_knowledge(self, card: Optional[Card], knowledge: SlotKnowledge) -> str:
        if card is None:
            return "empty"

        color_part = f"color={knowledge.known_color}" if knowledge.known_color else "color=unknown"
        rank_part = f"rank={knowledge.known_rank}" if knowledge.known_rank else "rank=unknown"

        not_color_part = "none"
        if not knowledge.known_color and knowledge.not_colors:
            not_color_part = "/".join(sorted(knowledge.not_colors))

        not_rank_part = "none"
        if not knowledge.known_rank and knowledge.not_ranks:
            not_rank_part = "/".join(str(rank) for rank in sorted(knowledge.not_ranks))

        touched_part = "never" if knowledge.last_touched_turn is None else f"t{knowledge.last_touched_turn}"
        return (
            f"occupied | {color_part} | {rank_part} | not-colors={not_color_part} | "
            f"not-ranks={not_rank_part} | last-touched={touched_part}"
        )

    def _partner_id(self, player_id: int) -> int:
        return 1 - player_id

    def _generate_deck(self) -> list[Card]:
        deck: list[Card] = []
        for color in COLORS:
            deck.extend([Card(color=color, rank=1), Card(color=color, rank=1), Card(color=color, rank=2), Card(color=color, rank=3)])
        return deck
