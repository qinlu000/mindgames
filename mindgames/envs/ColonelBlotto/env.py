import copy
import re
import string
from typing import Any, Dict, Optional, Tuple

from mindgames.core import Env, GAME_ID, Info, ObservationType
from mindgames.envs.ColonelBlotto.renderer import create_board_str
from mindgames.state import TwoPlayerState


class ColonelBlottoEnv(Env):
    def __init__(self, num_fields: int = 3, num_total_units: int = 20, num_rounds: int = 10):
        num_fields_int = int(num_fields)
        num_total_units_int = int(num_total_units)
        num_rounds_int = int(num_rounds)

        if num_fields_int < 2:
            raise ValueError(f"num_fields must be >= 2, got {num_fields!r}")
        if num_fields_int > 26:
            raise ValueError(f"num_fields must be <= 26, got {num_fields!r}")
        if num_total_units_int < num_fields_int:
            raise ValueError(
                f"num_total_units must be >= num_fields so each field can be named and allocated sensibly; "
                f"got num_total_units={num_total_units_int}, num_fields={num_fields_int}"
            )
        if num_rounds_int <= 0:
            raise ValueError(f"num_rounds must be > 0, got {num_rounds!r}")

        self.num_fields = num_fields_int
        self.num_total_units = num_total_units_int
        self.num_rounds = num_rounds_int
        self.field_names = list(string.ascii_uppercase[: self.num_fields])
        self._player_state_template = {
            "units_remaining": self.num_total_units,
            "current_allocation": {field_name: 0 for field_name in self.field_names},
            "allocation_complete": False,
        }

    def get_board_str(self) -> str:
        return create_board_str(
            game_state=self.state.game_state,
            num_rounds=self.num_rounds,
            num_total_units=self.num_total_units,
        )

    def reset(self, num_players: int, seed: Optional[int] = None):
        if num_players != 2:
            raise ValueError(f"ColonelBlotto is a 2-player game; got num_players={num_players}")

        self.state = TwoPlayerState(num_players=num_players, seed=seed, max_turns=self.num_rounds * 2)
        game_state = {
            "fields": [
                {"name": field_name, "value": 1, "player_0_units": 0, "player_1_units": 0}
                for field_name in self.field_names
            ],
            "current_round": 1,
            "scores": {0: 0, 1: 0},
            "player_states": {
                0: self._fresh_player_state(),
                1: self._fresh_player_state(),
            },
        }
        self.state.reset(
            game_state=game_state,
            player_prompt_function=self._prompt,
            role_mapping={0: "Commander Alpha", 1: "Commander Beta"},
        )
        self._emit_board()

    def _fresh_player_state(self) -> Dict[str, Any]:
        return copy.deepcopy(self._player_state_template)

    def _prompt(self, player_id: int, game_state: Dict[str, Any]) -> str:
        del game_state
        role = "Commander Alpha" if player_id == 0 else "Commander Beta"
        example_allocation = self._example_allocation()
        return (
            f"You are {role} in a game of Colonel Blotto. "
            f"Each round, allocate exactly {self.num_total_units} units across fields "
            f"{', '.join(self.field_names)}.\n"
            f"Submit one allocation in the format '{example_allocation}'.\n"
            "You may omit a field to allocate 0 units to it.\n"
            "Allocations are hidden until both players have submitted for the round.\n"
            "Higher allocation wins a field; equal allocations tie that field.\n"
            "Win more fields than your opponent to win the round; if field wins are tied, the round is a tie.\n"
            f"The match ends after {self.num_rounds} rounds or as soon as a commander has already secured a majority of rounds."
        )

    def _emit_board(self) -> None:
        self.state.add_observation(message=self.get_board_str(), observation_type=ObservationType.GAME_BOARD)

    def step(self, action: str) -> Tuple[bool, Info]:
        if self.state.done:
            return True, {}

        self.state.add_observation(
            from_id=self.state.current_player_id,
            to_id=self.state.current_player_id,
            message=action,
            observation_type=ObservationType.PLAYER_ACTION,
        )
        self._execute_player_move(action)
        self._check_gameover()
        return self.state.step()

    def _execute_player_move(self, action: str) -> None:
        allocation_dict = self._parse_allocation_input(action)
        validation_result = self._validate_allocation(allocation_dict)
        if validation_result != "Allocation is good.":
            self.state.add_observation(
                from_id=GAME_ID,
                to_id=self.state.current_player_id,
                message=f"Invalid allocation: {validation_result}",
                observation_type=ObservationType.GAME_MESSAGE,
            )
            self.state.set_invalid_move(reason=validation_result)
            return

        player_id = self.state.current_player_id
        player_state = self.state.game_state["player_states"][player_id]
        for field in self.state.game_state["fields"]:
            units = int(allocation_dict[field["name"]])
            field[f"player_{player_id}_units"] = units
            player_state["current_allocation"][field["name"]] = units

        player_state["units_remaining"] = 0
        player_state["allocation_complete"] = True

        other_player = 1 - player_id
        if self.state.game_state["player_states"][other_player]["allocation_complete"]:
            self._resolve_battle()

    def _parse_allocation_input(self, action_string: str) -> Optional[Dict[str, int]]:
        if not action_string or not action_string.strip():
            return None

        raw = action_string.strip()
        bracket_match = re.search(r"\[([^\]]+)\]", raw)
        body = (bracket_match.group(1) if bracket_match else raw).strip()
        if not body:
            return None

        token_re = re.compile(r"([A-Za-z])\s*:?\s*(\d+)", re.IGNORECASE)
        matches = list(token_re.finditer(body))
        if not matches:
            return None

        allocations: Dict[str, int] = {}
        for match in matches:
            field = match.group(1).upper()
            if field in allocations:
                return None
            allocations[field] = int(match.group(2))

        leftovers = token_re.sub("", body)
        leftovers = re.sub(r"[\s,]+", "", leftovers)
        if leftovers:
            return None

        for field_name in self.field_names:
            allocations.setdefault(field_name, 0)
        return allocations

    def _example_allocation(self) -> str:
        remaining_units = self.num_total_units
        parts: list[str] = []
        for idx, field_name in enumerate(self.field_names):
            if idx == len(self.field_names) - 1:
                units = remaining_units
            else:
                units = 1
                remaining_units -= units
            parts.append(f"{field_name}{units}")
        return "[" + " ".join(parts) + "]"

    def _validate_allocation(self, allocation_dict: Optional[Dict[str, int]]) -> str:
        if allocation_dict is None:
            return "Invalid input format. Use allocations like '[A4 B2 C14]'."
        if any(field_name not in self.field_names for field_name in allocation_dict):
            return f"Invalid field name(s). Valid fields: {', '.join(self.field_names)}"
        if any((not isinstance(units, int)) or units < 0 for units in allocation_dict.values()):
            return "All allocations must be non-negative integers."

        total_units = sum(allocation_dict.values())
        if total_units != self.num_total_units:
            return (
                f"You must allocate exactly {self.num_total_units} units. "
                f"Current sum: {total_units}"
            )
        return "Allocation is good."

    def _resolve_battle(self) -> None:
        field_winners = []
        for field in self.state.game_state["fields"]:
            p0_units = int(field["player_0_units"])
            p1_units = int(field["player_1_units"])
            if p0_units > p1_units:
                field_winners.append(0)
            elif p1_units > p0_units:
                field_winners.append(1)
            else:
                field_winners.append(None)

        p0_wins = field_winners.count(0)
        p1_wins = field_winners.count(1)

        p0_allocations = ", ".join(
            f"{field['name']}: {int(field['player_0_units'])}" for field in self.state.game_state["fields"]
        )
        p1_allocations = ", ".join(
            f"{field['name']}: {int(field['player_1_units'])}" for field in self.state.game_state["fields"]
        )

        message = (
            f"Round {int(self.state.game_state['current_round'])}\n"
            f"Commander Alpha allocated: {p0_allocations}\n"
            f"Commander Beta allocated: {p1_allocations}\n"
        )
        if p0_wins > p1_wins:
            self.state.game_state["scores"][0] += 1
            message += "Winner: Commander Alpha"
        elif p1_wins > p0_wins:
            self.state.game_state["scores"][1] += 1
            message += "Winner: Commander Beta"
        else:
            message += "Tie!"

        self.state.add_observation(
            from_id=GAME_ID,
            to_id=-1,
            message=message,
            observation_type=ObservationType.GAME_MESSAGE,
        )

        self.state.game_state["current_round"] += 1
        for player_id in (0, 1):
            self.state.game_state["player_states"][player_id] = self._fresh_player_state()
        for field in self.state.game_state["fields"]:
            field["player_0_units"] = 0
            field["player_1_units"] = 0
        self._emit_board()

    def _check_gameover(self) -> None:
        current_round = int(self.state.game_state["current_round"])
        scores = self.state.game_state["scores"]

        if current_round > self.num_rounds:
            if scores[0] > scores[1]:
                self.state.set_winner(
                    player_id=0,
                    reason=f"Commander Alpha wins {scores[0]}-{scores[1]} after {self.num_rounds} rounds!",
                )
            elif scores[1] > scores[0]:
                self.state.set_winner(
                    player_id=1,
                    reason=f"Commander Beta wins {scores[1]}-{scores[0]} after {self.num_rounds} rounds!",
                )
            else:
                self.state.set_draw(
                    reason=f"Game ends in a {scores[0]}-{scores[1]} tie after {self.num_rounds} rounds!"
                )
            return

        rounds_needed_to_win = (self.num_rounds // 2) + 1
        if scores[0] >= rounds_needed_to_win:
            self.state.set_winner(
                player_id=0,
                reason=f"Commander Alpha wins {scores[0]}-{scores[1]} (majority achieved)!",
            )
        elif scores[1] >= rounds_needed_to_win:
            self.state.set_winner(
                player_id=1,
                reason=f"Commander Beta wins {scores[1]}-{scores[0]} (majority achieved)!",
            )
