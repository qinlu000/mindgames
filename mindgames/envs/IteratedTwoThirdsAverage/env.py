import re
from typing import Any, Dict, Optional, Tuple

from mindgames.core import Env, Info, ObservationType, GAME_ID
from mindgames.state import TwoPlayerState
from mindgames.envs.IteratedTwoThirdsAverage.renderer import create_board_str


_GUESS_RE = re.compile(r"\[\s*([-+]?(?:\d+(?:\.\d*)?|\.\d+))\s*\]")


class IteratedTwoThirdsAverageEnv(Env):
    def __init__(self, num_rounds: int = 5, min_guess: float = 0.0, max_guess: float = 100.0):
        num_rounds_int = int(num_rounds)
        min_guess_f = float(min_guess)
        max_guess_f = float(max_guess)

        if num_rounds_int <= 0:
            raise ValueError(f"num_rounds must be > 0, got {num_rounds!r}")
        if min_guess_f > max_guess_f:
            raise ValueError(f"min_guess must be <= max_guess, got {min_guess_f} > {max_guess_f}")

        self.num_rounds = num_rounds_int
        self.min_guess = min_guess_f
        self.max_guess = max_guess_f

    def get_board_str(self) -> str:
        return create_board_str(game_state=self.state.game_state)

    def reset(self, num_players: int, seed: Optional[int] = None):
        if num_players != 2:
            raise ValueError(f"IteratedTwoThirdsAverage is a 2-player game; got num_players={num_players}")
        self.state = TwoPlayerState(num_players=num_players, seed=seed, max_turns=self.num_rounds * 2)
        self.state.reset(
            game_state={
                "round": 1,
                "num_rounds": self.num_rounds,
                "min_guess": self.min_guess,
                "max_guess": self.max_guess,
                "points": {0: 0, 1: 0},
                "guesses": {},
                "history": [],
            },
            player_prompt_function=self._prompt,
        )
        self._emit_board()

    def _prompt(self, player_id: int, game_state: Dict[str, Any]) -> str:
        return (
            f"You are Player {player_id} in a {self.num_rounds}-round Iterated Two-Thirds Average game.\n"
            f"Each round, choose a number between {self.min_guess} and {self.max_guess}.\n"
            "After both players choose, the target is (2/3) × (average of both guesses).\n"
            "The player whose guess is closest to the target wins the round.\n"
            "Reply with your guess in the format '[<number>]'."
        )

    def _emit_board(self) -> None:
        self.state.add_observation(message=self.get_board_str(), observation_type=ObservationType.GAME_BOARD)

    def step(self, action: str) -> Tuple[bool, Info]:
        pid = self.state.current_player_id

        # Keep guesses private until the round resolves: record the raw action only for the acting player.
        self.state.add_observation(
            from_id=pid,
            to_id=pid,
            message=action,
            observation_type=ObservationType.PLAYER_ACTION,
        )

        match = _GUESS_RE.search(action)
        if not match:
            self.state.set_invalid_move(reason="Invalid format; submit your guess as '[<number>]'.")
            return self.state.step()

        guess = float(match.group(1))
        if not (self.min_guess <= guess <= self.max_guess):
            self.state.set_invalid_move(reason=f"Guess must be between {self.min_guess} and {self.max_guess}.")
            return self.state.step()

        self.state.game_state["guesses"][pid] = guess

        if len(self.state.game_state["guesses"]) == 2:
            round_idx = int(self.state.game_state["round"])
            guesses: Dict[int, float] = self.state.game_state["guesses"]
            avg = (guesses[0] + guesses[1]) / 2.0
            target = (2.0 / 3.0) * avg
            d0 = abs(guesses[0] - target)
            d1 = abs(guesses[1] - target)

            if d0 < d1:
                winner: Optional[int] = 0
            elif d1 < d0:
                winner = 1
            else:
                winner = None

            self.state.game_state["history"].append(
                {0: guesses[0], 1: guesses[1], "target": target, "winner": winner}
            )

            self.state.add_observation(
                from_id=GAME_ID,
                to_id=-1,
                message=(
                    f"Round {self.state.game_state['round']} results: "
                    f"Player 0 guessed {guesses[0]}; Player 1 guessed {guesses[1]}; "
                    f"target = {target:.2f}."
                ),
                observation_type=ObservationType.GAME_MESSAGE,
            )

            if winner is None:
                self.state.add_observation(
                    from_id=GAME_ID,
                    to_id=-1,
                    message="Round is a draw.",
                    observation_type=ObservationType.GAME_MESSAGE,
                )
            else:
                self.state.game_state["points"][winner] += 1
                self.state.add_observation(
                    from_id=GAME_ID,
                    to_id=-1,
                    message=f"Player {winner} wins the round!",
                    observation_type=ObservationType.GAME_MESSAGE,
                )

            self.state.game_state["guesses"].clear()

            if round_idx >= self.num_rounds:
                p0 = int(self.state.game_state["points"][0])
                p1 = int(self.state.game_state["points"][1])
                self.state.add_observation(
                    from_id=GAME_ID,
                    to_id=-1,
                    message=f"Final score: Player 0 = {p0}, Player 1 = {p1}.",
                    observation_type=ObservationType.GAME_MESSAGE,
                )
                if p0 > p1:
                    self.state.set_winner(player_id=0, reason="Player 0 won more rounds.")
                elif p1 > p0:
                    self.state.set_winner(player_id=1, reason="Player 1 won more rounds.")
                else:
                    self.state.set_draw(reason="Overall game is a draw.")
            else:
                self.state.game_state["round"] = round_idx + 1
            self._emit_board()

        return self.state.step()
