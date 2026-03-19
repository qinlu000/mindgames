import copy
import random
import re
from typing import Any, Dict, Optional, Tuple

from mindgames.core import Env, Info, ObservationType
from mindgames.state import TwoPlayerState


class NegotiationEnv(Env):
    DEFAULT_BASE_VALUES: Dict[str, int] = {
        "Wheat": 5,
        "Wood": 10,
        "Sheep": 15,
        "Brick": 25,
        "Ore": 40,
    }

    OFFER_RE = re.compile(r"\[offer\s*:\s*(.*?)\s*->\s*(.*?)\]", re.IGNORECASE | re.DOTALL)
    ACCEPT_RE = re.compile(r"\[accept\]", re.IGNORECASE)
    DENY_RE = re.compile(r"\[deny\]", re.IGNORECASE)

    def __init__(
        self,
        max_turns: int = 20,
        min_resource_qty: int = 4,
        max_resource_qty: int = 12,
        starting_resources: Optional[Dict[int, Dict[str, int]]] = None,
        resource_values: Optional[Dict[int, Dict[str, int]]] = None,
    ):
        if max_turns <= 0:
            raise ValueError(f"max_turns must be > 0, got {max_turns!r}")
        if max_turns % 2 != 0:
            raise ValueError(f"max_turns must be even for symmetric alternating play, got {max_turns!r}")
        if min_resource_qty < 0:
            raise ValueError(f"min_resource_qty must be >= 0, got {min_resource_qty!r}")
        if max_resource_qty < min_resource_qty:
            raise ValueError(
                f"max_resource_qty must be >= min_resource_qty, got {max_resource_qty!r} < {min_resource_qty!r}"
            )

        self.max_turns = int(max_turns)
        self.min_resource_qty = int(min_resource_qty)
        self.max_resource_qty = int(max_resource_qty)
        self.resource_names = tuple(self.DEFAULT_BASE_VALUES.keys())
        self._resource_lookup = self._build_resource_lookup()
        self.starting_resources_override = self._normalize_player_table(
            starting_resources,
            field_name="starting_resources",
        )
        self.resource_values_override = self._normalize_player_table(
            resource_values,
            field_name="resource_values",
        )

    def get_board_str(self):
        return self._render_turn_state(player_id=self.state.current_player_id)

    def reset(self, num_players: int, seed: Optional[int] = None):
        self.state = TwoPlayerState(num_players=num_players, max_turns=self.max_turns, seed=seed)

        resources = (
            copy.deepcopy(self.starting_resources_override)
            if self.starting_resources_override is not None
            else self._generate_starting_resources()
        )
        resource_values = (
            copy.deepcopy(self.resource_values_override)
            if self.resource_values_override is not None
            else self._generate_resource_values()
        )
        initial_totals = {
            pid: self._inventory_total(resources=resources[pid], resource_values=resource_values[pid])
            for pid in range(2)
        }

        self.state.reset(
            game_state={
                "resources": resources,
                "resource_values": resource_values,
                "initial_totals": initial_totals,
                "pending_offer": None,
                "trade_history": [],
            },
            player_prompt_function=self._prompt,
            role_mapping={0: "Negotiator 0", 1: "Negotiator 1"},
        )
        self._emit_turn_state()

    def _prompt(self, player_id: int, game_state: Dict[str, Any]) -> str:
        public_inventories = self._format_public_inventories(game_state["resources"])
        resource_values = game_state["resource_values"][player_id]
        starting_total = game_state["initial_totals"][player_id]
        return (
            f"You are Player {player_id} in a 2-player private-value negotiation game.\n"
            f"The game lasts at most {self.max_turns} total turns across both players.\n"
            "Goal: maximize your own final portfolio value gain relative to your starting value.\n"
            "Information structure:\n"
            "- Both players can see both players' current inventories.\n"
            "- Your per-unit values are private to you.\n"
            "- All chat messages, offers, acceptances, and denials are public.\n"
            "Action rules:\n"
            "- If there is no pending offer to you, you may send a public message and/or include one offer using "
            "'[Offer: 2 Wheat, 1 Ore -> 3 Sheep]'.\n"
            "- If there is a pending offer to you, you must respond with '[Accept]' or '[Deny]'.\n"
            "- If you use '[Deny]', you may append one counteroffer using '[Offer: ...]'.\n"
            "- In '[Offer: A -> B]', the resources before '->' are what YOU give, and the resources after '->' are "
            "what YOU request.\n"
            "- Only one live offer can exist at a time.\n"
            "- When possible, place '[Offer: ...]', '[Accept]', or '[Deny]' at the start of your message.\n"
            f"Public starting inventories:\n{public_inventories}\n"
            f"Your private per-unit values: {self._format_values(resource_values)}.\n"
            f"Your starting portfolio value: {starting_total}."
        )

    def step(self, action: str) -> Tuple[bool, Info]:
        player_id = self.state.current_player_id
        action = str(action).strip()
        self.state.add_observation(
            from_id=player_id,
            message=action,
            observation_type=ObservationType.PLAYER_ACTION,
        )

        pending_offer = self.state.game_state["pending_offer"]
        has_pending_offer = isinstance(pending_offer, dict) and pending_offer.get("to_player") == player_id
        has_accept = bool(self.ACCEPT_RE.search(action))
        has_deny = bool(self.DENY_RE.search(action))
        parsed_offer, offer_error = self._extract_offer(action)
        if offer_error is not None:
            return self._handle_invalid_move(offer_error)

        if has_pending_offer:
            if has_accept and has_deny:
                return self._handle_invalid_move("Choose exactly one of [Accept] or [Deny] when an offer is pending.")
            if not has_accept and not has_deny:
                return self._handle_invalid_move(
                    "There is a pending offer to you. Respond with [Accept] or [Deny] before sending more chat."
                )
            if has_accept and parsed_offer is not None:
                return self._handle_invalid_move("Do not attach a new offer to an [Accept] action.")

            if has_accept:
                validation_error = self._validate_offer(pending_offer)
                if validation_error is not None:
                    return self._handle_invalid_move(validation_error)
                self._execute_trade(pending_offer)
                self.state.game_state["pending_offer"] = None
                self.state.game_state["trade_history"].append(
                    {
                        "turn": self.state.turn,
                        "status": "accepted",
                        "offer": copy.deepcopy(pending_offer),
                    }
                )
                self.state.add_observation(
                    message=(
                        f"Player {player_id} accepted the offer. "
                        f"{self._describe_offer(pending_offer)}"
                    ),
                    observation_type=ObservationType.GAME_ACTION_DESCRIPTION,
                )
                return self._advance_turn()

            counter_offer = None
            if parsed_offer is not None:
                counter_offer = self._build_offer(
                    from_player=player_id,
                    give_bundle=parsed_offer["give_bundle"],
                    request_bundle=parsed_offer["request_bundle"],
                )
                validation_error = self._validate_offer(counter_offer)
                if validation_error is not None:
                    return self._handle_invalid_move(validation_error)
            self.state.game_state["trade_history"].append(
                {
                    "turn": self.state.turn,
                    "status": "denied",
                    "offer": copy.deepcopy(pending_offer),
                }
            )
            self.state.game_state["pending_offer"] = None
            self.state.add_observation(
                message=f"Player {player_id} denied the pending offer.",
                observation_type=ObservationType.GAME_ACTION_DESCRIPTION,
            )
            if counter_offer is not None:
                self.state.game_state["pending_offer"] = counter_offer
                self.state.game_state["trade_history"].append(
                    {
                        "turn": self.state.turn,
                        "status": "offered",
                        "offer": copy.deepcopy(counter_offer),
                    }
                )
                self.state.add_observation(
                    message=self._describe_new_offer(counter_offer),
                    observation_type=ObservationType.GAME_ACTION_DESCRIPTION,
                )
            return self._advance_turn()

        if has_accept or has_deny:
            return self._handle_invalid_move("There is no pending offer to accept or deny.")

        if parsed_offer is not None:
            new_offer = self._build_offer(
                from_player=player_id,
                give_bundle=parsed_offer["give_bundle"],
                request_bundle=parsed_offer["request_bundle"],
            )
            validation_error = self._validate_offer(new_offer)
            if validation_error is not None:
                return self._handle_invalid_move(validation_error)
            self.state.game_state["pending_offer"] = new_offer
            self.state.game_state["trade_history"].append(
                {
                    "turn": self.state.turn,
                    "status": "offered",
                    "offer": copy.deepcopy(new_offer),
                }
            )
            self.state.add_observation(
                message=self._describe_new_offer(new_offer),
                observation_type=ObservationType.GAME_ACTION_DESCRIPTION,
            )

        return self._advance_turn()

    def _generate_starting_resources(self) -> Dict[int, Dict[str, int]]:
        resources: Dict[int, Dict[str, int]] = {}
        for pid in range(2):
            resources[pid] = {
                resource: random.randint(self.min_resource_qty, self.max_resource_qty)
                for resource in self.resource_names
            }
        return resources

    def _generate_resource_values(self) -> Dict[int, Dict[str, int]]:
        values: Dict[int, Dict[str, int]] = {}
        value_tiers = list(self.DEFAULT_BASE_VALUES.values())
        for pid in range(2):
            shuffled_tiers = random.sample(value_tiers, k=len(value_tiers))
            values[pid] = {
                resource: shuffled_tiers[idx]
                for idx, resource in enumerate(self.resource_names)
            }
        return values

    def _emit_turn_state(self) -> None:
        player_id = self.state.current_player_id
        self.state.add_observation(
            to_id=player_id,
            message=self._render_turn_state(player_id=player_id),
            observation_type=ObservationType.GAME_BOARD,
        )

    def _render_turn_state(self, player_id: int) -> str:
        resources = self.state.game_state["resources"]
        resource_values = self.state.game_state["resource_values"][player_id]
        current_total = self._inventory_total(resources=resources[player_id], resource_values=resource_values)
        initial_total = self.state.game_state["initial_totals"][player_id]
        gain = current_total - initial_total
        pending_offer = self.state.game_state["pending_offer"]
        turns_remaining = self.max_turns - self.state.turn

        lines = [
            f"Turn {self.state.turn + 1} of {self.max_turns}",
            f"Turns remaining including this turn: {turns_remaining}",
            "Public inventories:",
            f"- Player 0 inventory: {self._format_bundle(resources[0])}",
            f"- Player 1 inventory: {self._format_bundle(resources[1])}",
            f"Your private per-unit values: {self._format_values(resource_values)}",
            f"Your current portfolio value: {current_total} (gain {gain:+d} from start)",
        ]
        if isinstance(pending_offer, dict) and pending_offer.get("to_player") == player_id:
            lines.append(f"Pending offer you must resolve now: {self._describe_offer(pending_offer)}")
            lines.append("Required reply: start with [Accept] or [Deny]. You may append one counteroffer using [Offer: ...].")
        else:
            lines.append("No pending offer to you.")
            lines.append("You may send a public message and/or make one offer using [Offer: ...].")
        lines.append("All chat and all trade-control tags are public to both players.")
        return "\n".join(lines)

    def _advance_turn(self) -> Tuple[bool, Info]:
        done, info = self.state.step()
        if not done and self.state.check_turn_limit():
            self._resolve_game()
            info = dict(info)
            info["reason"] = self.state.game_info[0].get("reason")
            return True, info
        if not self.state.done:
            self._emit_turn_state()
        return self.state.done, info

    def _resolve_game(self) -> None:
        current_totals = {
            pid: self._inventory_total(
                resources=self.state.game_state["resources"][pid],
                resource_values=self.state.game_state["resource_values"][pid],
            )
            for pid in range(2)
        }
        gains = {
            pid: current_totals[pid] - self.state.game_state["initial_totals"][pid]
            for pid in range(2)
        }
        reason = (
            f"Turn limit reached ({self.max_turns}). "
            f"Value gains: Player 0 {gains[0]:+d}, Player 1 {gains[1]:+d}."
        )
        if gains[0] > gains[1]:
            self.state.set_winner(player_id=0, reason=reason)
        elif gains[1] > gains[0]:
            self.state.set_winner(player_id=1, reason=reason)
        else:
            self.state.set_draw(reason=reason)

    def _handle_invalid_move(self, reason: str) -> Tuple[bool, Info]:
        self.state.set_invalid_move(reason=reason)
        return self._advance_turn()

    def _extract_offer(self, action: str) -> Tuple[Optional[Dict[str, Dict[str, int]]], Optional[str]]:
        if "[offer" not in action.lower():
            return None, None

        matches = list(self.OFFER_RE.finditer(action))
        if not matches:
            return None, "Offers must use the format [Offer: 2 Wheat, 1 Ore -> 3 Sheep]."
        if len(matches) > 1:
            return None, "Submit at most one [Offer: ...] per turn."

        give_text = matches[0].group(1)
        request_text = matches[0].group(2)
        give_bundle, give_error = self._parse_bundle(give_text)
        if give_error is not None:
            return None, give_error
        request_bundle, request_error = self._parse_bundle(request_text)
        if request_error is not None:
            return None, request_error

        return {
            "give_bundle": give_bundle,
            "request_bundle": request_bundle,
        }, None

    def _parse_bundle(self, text: str) -> Tuple[Optional[Dict[str, int]], Optional[str]]:
        items = [item.strip() for item in text.split(",") if item.strip()]
        if not items:
            return None, "Offers must include at least one resource on each side of '->'."

        bundle: Dict[str, int] = {}
        for item in items:
            match = re.fullmatch(r"(\d+)\s+([A-Za-z][A-Za-z ]*)", item)
            if match is None:
                return None, f"Could not parse resource entry {item!r}."
            qty = int(match.group(1))
            if qty <= 0:
                return None, "Resource quantities must be positive integers."
            resource_name = self._normalize_resource_name(match.group(2))
            if resource_name is None:
                return None, f"Unknown resource {match.group(2)!r}."
            bundle[resource_name] = bundle.get(resource_name, 0) + qty
        return bundle, None

    def _build_offer(
        self,
        from_player: int,
        give_bundle: Dict[str, int],
        request_bundle: Dict[str, int],
    ) -> Dict[str, Any]:
        return {
            "from_player": from_player,
            "to_player": 1 - from_player,
            "give_bundle": dict(give_bundle),
            "request_bundle": dict(request_bundle),
        }

    def _validate_offer(self, offer: Dict[str, Any]) -> Optional[str]:
        from_player = int(offer["from_player"])
        to_player = int(offer["to_player"])
        proposer_resources = self.state.game_state["resources"][from_player]
        responder_resources = self.state.game_state["resources"][to_player]

        for resource, qty in offer["give_bundle"].items():
            if proposer_resources[resource] < qty:
                return (
                    f"Offer invalid: the public inventory shows Player {from_player} has only "
                    f"{proposer_resources[resource]} {resource}."
                )
        for resource, qty in offer["request_bundle"].items():
            if responder_resources[resource] < qty:
                return (
                    f"Offer invalid: the public inventory shows Player {to_player} has only "
                    f"{responder_resources[resource]} {resource}."
                )
        return None

    def _execute_trade(self, offer: Dict[str, Any]) -> None:
        from_player = int(offer["from_player"])
        to_player = int(offer["to_player"])
        resources = self.state.game_state["resources"]
        for resource, qty in offer["give_bundle"].items():
            resources[from_player][resource] -= qty
            resources[to_player][resource] += qty
        for resource, qty in offer["request_bundle"].items():
            resources[to_player][resource] -= qty
            resources[from_player][resource] += qty

    def _inventory_total(self, resources: Dict[str, int], resource_values: Dict[str, int]) -> int:
        return sum(int(resources[resource]) * int(resource_values[resource]) for resource in self.resource_names)

    def _describe_new_offer(self, offer: Dict[str, Any]) -> str:
        return f"Player {offer['from_player']} made an offer. {self._describe_offer(offer)}"

    def _describe_offer(self, offer: Dict[str, Any]) -> str:
        from_player = int(offer["from_player"])
        to_player = int(offer["to_player"])
        give_text = self._format_bundle(offer["give_bundle"])
        request_text = self._format_bundle(offer["request_bundle"])
        return (
            f"Player {from_player} gives {give_text} to Player {to_player} "
            f"in exchange for {request_text}."
        )

    def _format_public_inventories(self, resources: Dict[int, Dict[str, int]]) -> str:
        return "\n".join(
            [f"- Player {pid} inventory: {self._format_bundle(resources[pid])}" for pid in range(2)]
        )

    def _format_bundle(self, bundle: Dict[str, int]) -> str:
        parts = []
        for resource in self.resource_names:
            qty = int(bundle.get(resource, 0))
            if qty > 0:
                parts.append(f"{qty} {resource}")
        return ", ".join(parts) if parts else "nothing"

    def _format_values(self, resource_values: Dict[str, int]) -> str:
        return ", ".join(f"{resource}={int(resource_values[resource])}" for resource in self.resource_names)

    def _normalize_player_table(
        self,
        table: Optional[Dict[int, Dict[str, int]]],
        field_name: str,
    ) -> Optional[Dict[int, Dict[str, int]]]:
        if table is None:
            return None
        if sorted(table.keys()) != [0, 1]:
            raise ValueError(f"{field_name} must contain exactly player keys 0 and 1.")

        normalized: Dict[int, Dict[str, int]] = {}
        for pid in range(2):
            player_table = table[pid]
            if not isinstance(player_table, dict):
                raise ValueError(f"{field_name}[{pid}] must be a dict of resource -> quantity/value.")
            normalized_player = {resource: 0 for resource in self.resource_names}
            for raw_name, raw_value in player_table.items():
                resource_name = self._normalize_resource_name(str(raw_name))
                if resource_name is None:
                    raise ValueError(f"Unknown resource {raw_name!r} in {field_name}[{pid}].")
                value = int(raw_value)
                if value < 0:
                    raise ValueError(f"{field_name}[{pid}][{resource_name!r}] must be >= 0.")
                normalized_player[resource_name] = value
            normalized[pid] = normalized_player
        return normalized

    def _build_resource_lookup(self) -> Dict[str, str]:
        lookup: Dict[str, str] = {}
        for resource in self.resource_names:
            key = resource.lower()
            lookup[key] = resource
            if not key.endswith("s"):
                lookup[f"{key}s"] = resource
        return lookup

    def _normalize_resource_name(self, name: str) -> Optional[str]:
        normalized_name = " ".join(name.strip().lower().split())
        return self._resource_lookup.get(normalized_name)
