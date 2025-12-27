import random
import re
from typing import Any, Dict, Optional, Tuple

from mindgames.core import Env, Info, ObservationType, GAME_ID
from mindgames.state import FFAMultiPlayerState
from mindgames.envs.LiarsDice.renderer import create_board_str


class LiarsDiceEnv(Env):
    def __init__(self, num_dice: int = 5):
        """
        Args:
            num_dice (int): Initial number of dice each player starts with.
        """
        self.num_dice = num_dice

    def get_board_str(self):
        return create_board_str(game_state=self.state.game_state)

    def reset(self, num_players: int, seed: Optional[int] = None):
        assert 2 <= num_players <= 15, f"The number of players has to be 2<=x<=15, received {num_players}"
        self.state = FFAMultiPlayerState(num_players=num_players, seed=seed)
        remaining_dice = {pid: self.num_dice for pid in range(self.state.num_players)}
        game_state = {
            "round": 0,
            "current_bid": {"quantity": 0, "face_value": 0},
            "last_bidder_id": None,
            "remaining_dice": remaining_dice,
            "dice_rolls": None,
            "round_actions": [],
        }
        self.state.reset(game_state=game_state, player_prompt_function=self._prompt)
        self._roll_new_dice()
        self._emit_turn_board()

    def _prompt(self, player_id: int, game_state: Dict[str, Any]) -> str:
        return (
            f"You are Player {player_id} in an {self.state.num_players}-player Liar's Dice game.\n"
            "Rules:\n"
            "- On your turn, you may either:\n"
            "  1) Make a new bid with a higher quantity or higher face (or both) than the current bid; i.e. '[Bid: 3, 4]',\n"
            "  2) Call the last bid by typing '[Call]'.\n\n"
            "Bid validity: if a current bid exists, your new bid must satisfy new_Q >= current_Q AND new_F >= current_F, and not both equal.\n"
            "If you call:\n"
            "  - If the actual count of that face value among all dice is less than the bid, the last bidder loses one die.\n"
            "  - Otherwise, the caller loses one die.\n"
            "A player who reaches 0 dice is eliminated. The last remaining player wins."
        )

    def _roll_new_dice(self):
        self.state.game_state["round"] = int(self.state.game_state.get("round", 0)) + 1
        self.state.game_state["current_bid"] = {"quantity": 0, "face_value": 0}
        self.state.game_state["last_bidder_id"] = None
        self.state.game_state["round_actions"] = []

        new_dice_rolls = {}
        for pid, count in self.state.game_state["remaining_dice"].items():
            new_dice_rolls[pid] = [random.randint(1, 6) for _ in range(count)]
        self.state.game_state["dice_rolls"] = new_dice_rolls

    def _emit_turn_board(self) -> None:
        if getattr(self.state, "done", False):
            return

        pid = self.state.current_player_id
        round_idx = int(self.state.game_state.get("round", 0))
        remaining_dice = self.state.game_state.get("remaining_dice", {})
        remaining_summary = "; ".join([f"Player {p}: {d}" for p, d in remaining_dice.items()])

        current_bid = self.state.game_state.get("current_bid", {"quantity": 0, "face_value": 0})
        q = int(current_bid.get("quantity", 0) or 0)
        f = int(current_bid.get("face_value", 0) or 0)
        last_bidder = self.state.game_state.get("last_bidder_id", None)

        if q <= 0 or f <= 0 or last_bidder is None:
            bid_line = "Current bid: none (you MUST start with [Bid: Q, F]; [Call] is invalid until a bid exists)"
            legal_line = "Legal action: output ONLY one action: [Bid: Q, F]"
        else:
            bid_line = f"Current bid: {q} × face {f} (by Player {last_bidder})"
            legal_line = "Legal action: output ONLY one action: [Bid: Q, F] or [Call]"

        actions = list(self.state.game_state.get("round_actions", []) or [])
        if actions:
            actions_block = "\n".join([f"- {a}" for a in actions])
        else:
            actions_block = "(none)"

        rolled = (self.state.game_state.get("dice_rolls", {}) or {}).get(pid, [])
        dice_line = ", ".join(map(str, rolled)) if rolled else "(no dice)"

        message = (
            f"\nRound {round_idx}\n"
            f"Remaining dice: {remaining_summary}\n"
            f"{bid_line}\n"
            "Round actions so far:\n"
            f"{actions_block}\n"
            f"Your dice: {dice_line}\n"
            "Bid rule reminder: if a bid exists, you may not decrease quantity or face value; at least one must increase. Face must be 1..6.\n"
            f"{legal_line}"
        )
        self.state.add_observation(to_id=pid, message=message, observation_type=ObservationType.GAME_BOARD)

    def step(self, action: str) -> Tuple[bool, Info]:
        self.state.add_observation(from_id=self.state.current_player_id, message=action, observation_type=ObservationType.PLAYER_ACTION)

        # 1) Call
        if re.compile(r"\[call\]", re.IGNORECASE).search(action):
            current_bid = self.state.game_state["current_bid"]
            last_bidder_id = self.state.game_state["last_bidder_id"]

            if last_bidder_id is None or current_bid["quantity"] == 0:
                self._handle_invalid_move(reason="Call made with no prior bid.")
                self.state.add_observation(
                    to_id=self.state.current_player_id,
                    message="No prior bid in this round. Submit a bid first, e.g. '[Bid: 1, 1]'.",
                    observation_type=ObservationType.GAME_ADMIN,
                )
                return self.state.step(rotate_player=False)

            total_face_count = 0
            for _, dice_list in self.state.game_state["dice_rolls"].items():
                total_face_count += dice_list.count(current_bid["face_value"])

            prior_actions = list(self.state.game_state.get("round_actions", []) or [])
            prior_actions.append(f"Player {self.state.current_player_id}: Call")
            actions_block = "\n".join([f"- {a}" for a in prior_actions])

            if total_face_count < current_bid["quantity"]:
                loser_id = last_bidder_id
                msg = (
                    f"Player {self.state.current_player_id} calls! The actual count of face {current_bid['face_value']} "
                    f"is {total_face_count}, which is LESS than {current_bid['quantity']}.\n"
                    f"Player {loser_id} (the last bidder) loses one die.\n"
                    f"Round actions:\n{actions_block}"
                )
            else:
                loser_id = self.state.current_player_id
                msg = (
                    f"Player {self.state.current_player_id} calls! The actual count of face {current_bid['face_value']} "
                    f"is {total_face_count}, which is >= {current_bid['quantity']}.\n"
                    f"Player {loser_id} (the caller) loses one die.\n"
                    f"Round actions:\n{actions_block}"
                )

            self._apply_die_loss(loser_id, msg)
            self._rotate_players()
            self._emit_turn_board()
            return self.state.step(rotate_player=False)

        # 2) Bid
        bid_match = re.compile(r"\[bid\s*:?\s*(\d+)[,\s]+(\d+)\]", re.IGNORECASE).search(action)
        if bid_match:
            new_quantity = int(bid_match.group(1))
            new_face_value = int(bid_match.group(2))
            is_valid, reason = self._is_valid_bid(new_quantity, new_face_value, self.state.game_state["current_bid"])
            if is_valid:
                self.state.game_state["current_bid"] = {"quantity": new_quantity, "face_value": new_face_value}
                self.state.game_state["last_bidder_id"] = self.state.current_player_id
                self.state.game_state["round_actions"].append(
                    f"Player {self.state.current_player_id}: Bid {new_quantity} × face {new_face_value}"
                )
                self.state.add_observation(
                    message=f"Player {self.state.current_player_id} bids {new_quantity} of face {new_face_value}.",
                    observation_type=ObservationType.GAME_ACTION_DESCRIPTION,
                )
                self._rotate_players()
                self._emit_turn_board()
            else:
                self._handle_invalid_move(reason=f"Invalid bid: {reason}")

            return self.state.step(rotate_player=False)

        # 3) Invalid
        self._handle_invalid_move(
            reason=f"Action not recognized as either a valid '[Bid: X, Y]' or '[Call]'. Submitted action: {action}"
        )
        return self.state.step(rotate_player=False)

    def _handle_invalid_move(self, reason: str):
        was_eliminated = self.state.set_invalid_move(reason=reason)
        if was_eliminated:
            self.state.add_observation(
                from_id=GAME_ID,
                to_id=-1,
                message=f"Player {self.state.current_player_id} was eliminated by invalid move.",
                observation_type=ObservationType.GAME_MESSAGE,
            )
            self.state.game_state["remaining_dice"][self.state.current_player_id] = 0
            self._roll_new_dice()
            self._rotate_players()
            self._emit_turn_board()

    def _rotate_players(self):
        next_pid = self.state.next_alive_player()
        if next_pid is None or len(self.state.elimination_order) >= (self.state.num_players - 1):
            self._set_outcome()
        else:
            self.state.manually_set_current_player_id(new_player_id=next_pid, force=True)

    def _apply_die_loss(self, loser_id: int, message: str):
        self.state.add_observation(from_id=GAME_ID, to_id=-1, message=message, observation_type=ObservationType.GAME_MESSAGE)
        self.state.game_state["remaining_dice"][loser_id] -= 1
        if self.state.game_state["remaining_dice"][loser_id] == 0:
            self.state.add_elimination(pid=loser_id)
        self._roll_new_dice()

    def _is_valid_bid(self, new_quantity: int, new_face_value: int, current_bid: Dict[str, int]) -> Tuple[bool, str]:
        if new_quantity < current_bid["quantity"]:
            return False, f"New quantity {new_quantity} is lower than current {current_bid['quantity']}."
        if new_face_value < current_bid["face_value"]:
            return False, f"New face value {new_face_value} is lower than current {current_bid['face_value']}."
        if new_quantity == current_bid["quantity"] and new_face_value == current_bid["face_value"]:
            return False, "Bid is identical to the current bid."
        if not (1 <= new_face_value <= 6):
            return False, "Face value must be between 1 and 6."
        return True, ""

    def _set_outcome(self):
        final_ranking = self.state.elimination_order + [
            pid for pid, count in self.state.game_state["remaining_dice"].items() if count > 0
        ]
        reward_dict: Dict[int, float] = {}
        for rank, pid in enumerate(final_ranking):
            reward = -1.0 + 2.0 * (rank / (self.state.num_players - 1))
            reward_dict[pid] = reward
        self.state.set_game_outcome(reward_dict=reward_dict, reason=f"Player {final_ranking[-1]} wins! Final ranking: {final_ranking}")
