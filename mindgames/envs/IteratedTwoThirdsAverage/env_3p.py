import re
from typing import Any, Dict, Optional, Tuple

from mindgames.core import Env, Info, ObservationType, GAME_ID
from mindgames.state import FFAMultiPlayerState
from mindgames.envs.IteratedTwoThirdsAverage.renderer import create_board_str


_GUESS_RE = re.compile(r"\[\s*([-+]?(?:\d+(?:\.\d*)?|\.\d+))\s*\]")


class IteratedTwoThirdsAverage3PEnv(Env):
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
        if num_players != 3:
            raise ValueError(f"IteratedTwoThirdsAverage3P is a 3-player game; got num_players={num_players}")

        self.state = FFAMultiPlayerState(num_players=num_players, seed=seed, max_turns=self.num_rounds * num_players)
        self.state.reset(
            game_state={
                "round": 1,
                "num_rounds": self.num_rounds,
                "min_guess": self.min_guess,
                "max_guess": self.max_guess,
                "num_players": num_players,
                "points": {pid: 0 for pid in range(num_players)},
                "guesses": {},
                "history": [],
            },
            player_prompt_function=self._prompt,
        )
        self._emit_board()

    def _prompt(self, player_id: int, game_state: Dict[str, Any]) -> str:
        return (
            f"You are Player {player_id} in a {self.num_rounds}-round 3-player Iterated Two-Thirds Average game.\n"
            f"Each round, choose a number between {self.min_guess} and {self.max_guess}.\n"
            "After all players choose, the target is (2/3) × (average of all guesses).\n"
            "The player whose guess is closest to the target wins the round. Ties are draws.\n"
            "Reply with your guess in the format '[<number>]'."
        )

    def _emit_board(self) -> None:
        self.state.add_observation(message=self.get_board_str(), observation_type=ObservationType.GAME_BOARD)

    def step(self, action: str) -> Tuple[bool, Info]:
        pid = self.state.current_player_id

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

        num_players = int(self.state.game_state["num_players"])
        if len(self.state.game_state["guesses"]) == num_players:
            round_idx = int(self.state.game_state["round"])
            guesses: Dict[int, float] = self.state.game_state["guesses"]
            avg = sum(guesses.values()) / float(num_players)
            target = (2.0 / 3.0) * avg
            dists = {p: abs(g - target) for p, g in guesses.items()}
            min_dist = min(dists.values())
            winners = sorted([p for p, d in dists.items() if d == min_dist])
            winner = winners[0] if len(winners) == 1 else None

            hist: Dict[str | int, Any] = {p: guesses[p] for p in sorted(guesses.keys())}
            hist["target"] = target
            hist["winners"] = winners
            hist["winner"] = winner
            self.state.game_state["history"].append(hist)

            guess_str = "; ".join(f"P{p} guessed {guesses[p]}" for p in sorted(guesses.keys()))
            self.state.add_observation(
                from_id=GAME_ID,
                to_id=-1,
                message=f"Round {round_idx} results: {guess_str}; target = {target:.2f}.",
                observation_type=ObservationType.GAME_MESSAGE,
            )

            if winner is None:
                self.state.add_observation(
                    from_id=GAME_ID,
                    to_id=-1,
                    message=f"Round is a draw (tied players: {winners}).",
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
                points: Dict[int, int] = {int(k): int(v) for k, v in self.state.game_state["points"].items()}
                max_pts = max(points.values())
                winners_final = sorted([p for p, pts in points.items() if pts == max_pts])

                score_str = ", ".join(f"P{p}={points[p]}" for p in sorted(points.keys()))
                self.state.add_observation(
                    from_id=GAME_ID,
                    to_id=-1,
                    message=f"Final score: {score_str}.",
                    observation_type=ObservationType.GAME_MESSAGE,
                )

                if len(winners_final) == 1:
                    self.state.set_winners(player_ids=winners_final, reason="Player won more rounds.")
                else:
                    self.state.set_draw(reason="Overall game is a draw.")
            else:
                self.state.game_state["round"] = round_idx + 1

            self._emit_board()

        return self.state.step()

