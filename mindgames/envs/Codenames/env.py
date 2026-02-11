import random
import re
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from mindgames.core import Env, Info, ObservationType
from mindgames.envs.Codenames.renderer import create_board_str
from mindgames.state import TeamMultiPlayerState

_CLUE_RE = re.compile(r"\[(\w+)\s+(\d+)\]")
_GUESS_RE = re.compile(r"\[(\w+)\]")


class CodenamesEnv(Env):
    _WORDLIST_CACHE: Dict[bool, list[str]] = {}

    def __init__(self, hardcore: bool = False, max_turns: int = 80):
        self.hardcore = bool(hardcore)
        self.max_turns = int(max_turns)
        if self.max_turns <= 0:
            raise ValueError(f"max_turns must be > 0, got {max_turns!r}")
        self._load_word_list(hardcore=self.hardcore)

    def _load_word_list(self, hardcore: bool = False) -> None:
        if hardcore in self._WORDLIST_CACHE:
            self.word_list = self._WORDLIST_CACHE[hardcore]
            return

        filename = "words_hardcore.txt" if hardcore else "words_basic.txt"
        words_path = Path(__file__).with_name("data") / filename
        if not words_path.exists():
            raise FileNotFoundError(
                f"Missing local codenames word list at {words_path}. "
                "Please regenerate it from textarena or restore the file."
            )

        word_list = [line.strip() for line in words_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        self._WORDLIST_CACHE[hardcore] = word_list
        self.word_list = word_list
        if len(self.word_list) < 25:
            raise ValueError("Codenames word list must contain at least 25 unique words.")

    def reset(self, num_players: int, seed: Optional[int] = None):
        if num_players != 4:
            raise ValueError(f"Codenames requires exactly 4 players, got {num_players}")

        self.state = TeamMultiPlayerState(num_players=num_players, max_turns=self.max_turns, seed=seed)

        assignments = ["R"] * 9 + ["B"] * 8 + ["N"] * 7 + ["A"]
        random.shuffle(assignments)
        self.board = {word: team for word, team in zip(random.sample(self.word_list, 25), assignments)}

        self.state.reset(
            game_state={
                "guessed_words": set(),
                "last_clue": None,
                "last_number": 0,
                "remaining_guesses": 0,
            },
            player_prompt_function=self._prompt,
            role_mapping={
                0: "Red Spymaster",
                1: "Red Operative",
                2: "Blue Spymaster",
                3: "Blue Operative",
            },
        )
        self._emit_board()

    def _prompt(self, player_id: int, game_state: Dict[str, Any]) -> str:
        prompt = (
            "You are playing Codenames, a 2v2 word deduction game.\n"
            "Rules:\n"
            "1) Spymaster gives one-word clue + number, format: '[clue 2]'.\n"
            "2) Operative guesses words in format '[word]' up to N+1 times or '[pass]'.\n"
            "3) Guessing Assassin ('A') loses instantly for your team.\n"
            "4) First team to reveal all own words wins.\n"
            "5) Clue may not be an exact match or substring of any board word, and vice versa.\n"
        )

        if player_id == 0:
            return prompt + "You are Player 0: Red Spymaster."
        if player_id == 1:
            return prompt + "You are Player 1: Red Operative."
        if player_id == 2:
            return prompt + "You are Player 2: Blue Spymaster."
        return prompt + "You are Player 3: Blue Operative."

    def _render_player_view(self) -> str:
        return create_board_str(
            board=self.board,
            guessed_words=self.state.game_state["guessed_words"],
            current_player_id=self.state.current_player_id,
            last_clue=self.state.game_state["last_clue"],
            last_number=int(self.state.game_state["last_number"]),
            remaining_guesses=int(self.state.game_state["remaining_guesses"]),
        )

    def get_board_str(self) -> str:
        return self._render_player_view()

    def _emit_board(self) -> None:
        self.state.add_observation(
            to_id=self.state.current_player_id,
            message=self._render_player_view(),
            observation_type=ObservationType.GAME_BOARD,
        )

    def _rotate_player_by_logic(self, done_guessing: bool = False, skip_guessing: bool = False) -> None:
        current = self.state.current_player_id
        if current == 0:
            next_player = 2 if skip_guessing else 1
        elif current == 2:
            next_player = 0 if skip_guessing else 3
        elif current == 1:
            next_player = 2 if done_guessing else 1
        else:
            next_player = 0 if done_guessing else 3
        self.state.manually_set_current_player_id(new_player_id=next_player)

    def _resolve_game(self) -> None:
        guessed_words = self.state.game_state["guessed_words"]
        red_correct = sum(1 for word, team in self.board.items() if team == "R" and word in guessed_words)
        blue_correct = sum(1 for word, team in self.board.items() if team == "B" and word in guessed_words)
        if red_correct > blue_correct:
            self.state.set_winners(
                player_ids=[0, 1],
                reason=f"Move limit reached ({self.max_turns}). Red revealed {red_correct} vs Blue {blue_correct}.",
            )
        elif blue_correct > red_correct:
            self.state.set_winners(
                player_ids=[2, 3],
                reason=f"Move limit reached ({self.max_turns}). Blue revealed {blue_correct} vs Red {red_correct}.",
            )
        else:
            self.state.set_draw(reason=f"Move limit reached ({self.max_turns}) with equal score.")

    def _finalize_step(self) -> Tuple[bool, Info]:
        if not self.state.done and self.state.turn + 1 >= self.max_turns:
            self._resolve_game()
        return self.state.step()

    def step(self, action: str) -> Tuple[bool, Info]:
        player_id = self.state.current_player_id
        current_team = "R" if player_id < 2 else "B"
        self.state.add_observation(
            from_id=player_id,
            to_id=self.state.current_player_id,
            message=action,
            observation_type=ObservationType.PLAYER_ACTION,
        )

        if player_id in {0, 2}:  # Spymaster turn
            clue_match = _CLUE_RE.search(action)
            if clue_match is None:
                self.state.add_observation(
                    message=(
                        f"Spymaster of {'Red' if current_team == 'R' else 'Blue'} team (Player {player_id}) "
                        "did not provide a valid clue. Team turn is skipped."
                    ),
                    observation_type=ObservationType.GAME_MESSAGE,
                )
                self._rotate_player_by_logic(skip_guessing=True)
                self._emit_board()
                return self._finalize_step()

            clue_word = clue_match.group(1).lower()
            clue_number = int(clue_match.group(2))

            if any(clue_word in board_word or board_word in clue_word for board_word in self.board.keys()):
                winner_ids = [0, 1] if current_team == "B" else [2, 3]
                self.state.set_winners(
                    player_ids=winner_ids,
                    reason=(
                        f"Player {player_id} used clue '{clue_word}' as exact/subset match with board word(s)."
                    ),
                )
                return self._finalize_step()

            self.state.game_state["last_clue"] = clue_word
            self.state.game_state["last_number"] = clue_number
            self.state.game_state["remaining_guesses"] = clue_number + 1
            self.state.add_observation(
                message=(
                    f"Spymaster of {'Red' if current_team == 'R' else 'Blue'} team "
                    f"(Player {player_id}) submitted [{clue_word} {clue_number}]."
                ),
                observation_type=ObservationType.GAME_ACTION_DESCRIPTION,
            )
            self._rotate_player_by_logic()
            self._emit_board()
            return self._finalize_step()

        # Operative turn
        guess_match = _GUESS_RE.search(action)
        if guess_match is None:
            self.state.add_observation(
                message=(
                    f"Operative of {'Red' if current_team == 'R' else 'Blue'} team (Player {player_id}) "
                    "did not provide a valid guess. Team turn is skipped."
                ),
                observation_type=ObservationType.GAME_MESSAGE,
            )
            self._rotate_player_by_logic(done_guessing=True)
            self._emit_board()
            return self._finalize_step()

        guessed_word = guess_match.group(1).lower()
        if guessed_word == "pass":
            self._rotate_player_by_logic(done_guessing=True)
            self._emit_board()
            return self._finalize_step()

        if guessed_word not in self.board or guessed_word in self.state.game_state["guessed_words"]:
            self.state.add_observation(
                message=(
                    f"Operative of {'Red' if current_team == 'R' else 'Blue'} team (Player {player_id}) "
                    "provided an invalid guess. Team turn is skipped."
                ),
                observation_type=ObservationType.GAME_MESSAGE,
            )
            self._rotate_player_by_logic(done_guessing=True)
            self._emit_board()
            return self._finalize_step()

        self.state.game_state["guessed_words"].add(guessed_word)
        guessed_team = self.board[guessed_word]

        if guessed_team == "A":
            winner_ids = [2, 3] if current_team == "R" else [0, 1]
            self.state.set_winners(player_ids=winner_ids, reason=f"Player {player_id} selected the assassin word.")
            return self._finalize_step()

        if guessed_team == current_team:
            if all(
                word in self.state.game_state["guessed_words"]
                for word, team in self.board.items()
                if team == current_team
            ):
                self.state.set_winners(
                    player_ids=[0, 1] if current_team == "R" else [2, 3],
                    reason=f"Player {player_id} guessed all words for their team.",
                )
                return self._finalize_step()

            self.state.add_observation(
                message=(
                    f"Operative of {'Red' if current_team == 'R' else 'Blue'} team "
                    f"(Player {player_id}) correctly guessed [{guessed_word}]."
                ),
                observation_type=ObservationType.GAME_ACTION_DESCRIPTION,
            )
            self.state.game_state["remaining_guesses"] -= 1
            if self.state.game_state["remaining_guesses"] <= 0:
                self.state.game_state["remaining_guesses"] = 0
                self._rotate_player_by_logic(done_guessing=True)
            self._emit_board()
            return self._finalize_step()

        opponent_team = "B" if current_team == "R" else "R"
        if all(
            word in self.state.game_state["guessed_words"]
            for word, team in self.board.items()
            if team == opponent_team
        ):
            self.state.set_winners(
                player_ids=[0, 1] if opponent_team == "R" else [2, 3],
                reason=f"Player {player_id} revealed the opponent team's final word.",
            )
            return self._finalize_step()

        if guessed_team == opponent_team:
            word_type = f"{'Red' if opponent_team == 'R' else 'Blue'} team"
        else:
            word_type = "neutral"

        self.state.add_observation(
            message=(
                f"Operative of {'Red' if current_team == 'R' else 'Blue'} team "
                f"(Player {player_id}) guessed [{guessed_word}] and it is a {word_type} word."
            ),
            observation_type=ObservationType.GAME_MESSAGE,
        )
        self.state.game_state["remaining_guesses"] = 0
        self._rotate_player_by_logic(done_guessing=True)
        self._emit_board()
        return self._finalize_step()
