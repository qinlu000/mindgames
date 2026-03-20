from __future__ import annotations

import re
from typing import Any


def _extract_observation_section(
    observation: str,
    start_marker: str,
    end_marker: str | None = None,
) -> str:
    start_index = observation.find(start_marker)
    if start_index == -1:
        return ""

    section = observation[start_index + len(start_marker) :]
    if end_marker is not None:
        end_index = section.find(end_marker)
        if end_index != -1:
            section = section[:end_index]
    return section


def mini_hanabi_parse_available_actions(observation: str, env: Any | None = None) -> list[str]:
    del env
    valid_actions: list[str] = []
    slot_labels = ("A", "B")
    color_order = ("Red", "Blue", "Green")
    rank_order = (1, 2, 3)

    info_match = re.search(r"Resources:\s*info\s+(\d+)\s*/", observation)
    info_tokens = int(info_match.group(1)) if info_match else 0

    own_section = _extract_observation_section(
        observation,
        "Your hand knowledge:",
        "Valid actions:",
    )
    occupied_slots: list[str] = []
    for slot_label in slot_labels:
        match = re.search(rf"- Slot {slot_label}:\s*(.+)", own_section)
        if match is None:
            continue
        if not match.group(1).strip().lower().startswith("empty"):
            occupied_slots.append(slot_label)

    if not own_section.strip():
        occupied_slots = list(slot_labels)

    for slot_label in occupied_slots:
        valid_actions.append(f"[Play {slot_label}]")
        valid_actions.append(f"[Discard {slot_label}]")

    if info_tokens > 0:
        partner_section = _extract_observation_section(
            observation,
            "Visible partner hand",
            "Your hand knowledge:",
        )
        visible_cards = re.findall(r"- Slot [AB]:\s*([A-Za-z]+)\s+([123])", partner_section)
        seen_colors = {color.title() for color, _ in visible_cards}
        seen_ranks = {int(rank) for _, rank in visible_cards}

        if not partner_section.strip():
            seen_colors = set(color_order)
            seen_ranks = set(rank_order)

        for color in color_order:
            if color in seen_colors:
                valid_actions.append(f"[Hint Color {color}]")
        for rank in rank_order:
            if rank in seen_ranks:
                valid_actions.append(f"[Hint Rank {rank}]")

    if not valid_actions:
        return ["[Play A]", "[Play B]", "[Discard A]", "[Discard B]"]

    return valid_actions


def _enumerate_allocations(num_fields: int, num_total_units: int) -> list[tuple[int, ...]]:
    allocations: list[tuple[int, ...]] = []

    def _dfs(fields_left: int, units_left: int, prefix: list[int]) -> None:
        if fields_left == 1:
            allocations.append(tuple(prefix + [units_left]))
            return
        for units in range(units_left + 1):
            _dfs(fields_left - 1, units_left - units, prefix + [units])

    _dfs(num_fields, num_total_units, [])
    return allocations


def _allocation_to_action(allocation: tuple[int, ...], field_names: list[str]) -> str:
    parts = [f"{field_name}{int(units)}" for field_name, units in zip(field_names, allocation)]
    return "[" + " ".join(parts) + "]"


def colonel_blotto_parse_available_actions(
    observation: str, env: Any | None = None
) -> list[str]:
    field_names: list[str] = []
    num_total_units: int | None = None

    if env is not None:
        current = env
        while getattr(current, "env", None) is not None:
            current = current.env
        maybe_field_names = getattr(current, "field_names", None)
        maybe_total = getattr(current, "num_total_units", None)
        if isinstance(maybe_field_names, list) and maybe_field_names:
            field_names = [str(name) for name in maybe_field_names]
        if isinstance(maybe_total, int):
            num_total_units = maybe_total

    if not field_names:
        match = re.search(r"Available fields:\s*([A-Za-z,\s]+)", observation)
        if match:
            field_names = [token.strip() for token in match.group(1).split(",") if token.strip()]

    if num_total_units is None:
        match = re.search(r"Units to allocate:\s*(\d+)", observation)
        if match:
            num_total_units = int(match.group(1))

    if not field_names or num_total_units is None or num_total_units < 0:
        return []

    allocations = _enumerate_allocations(len(field_names), num_total_units)
    return [_allocation_to_action(allocation, field_names) for allocation in allocations]


def _parse_public_inventory_text(inventory_text: str) -> dict[str, int]:
    inventory: dict[str, int] = {}
    cleaned_text = inventory_text.strip()
    if not cleaned_text or cleaned_text.lower() == "nothing":
        return inventory

    for item in cleaned_text.split(","):
        entry = item.strip()
        match = re.fullmatch(r"(\d+)\s+([A-Za-z][A-Za-z ]*)", entry)
        if match is None:
            continue
        inventory[" ".join(match.group(2).split()).title()] = int(match.group(1))
    return inventory


def _build_negotiation_offer_candidates(
    my_inventory: dict[str, int],
    their_inventory: dict[str, int],
    prefix: str = "",
) -> list[str]:
    resource_order = ("Wheat", "Wood", "Sheep", "Brick", "Ore")
    candidates: list[str] = []
    give_resources = [name for name in resource_order if my_inventory.get(name, 0) > 0]
    request_resources = [name for name in resource_order if their_inventory.get(name, 0) > 0]

    for give_name in give_resources:
        for request_name in request_resources:
            if give_name == request_name and len(request_resources) > 1:
                continue
            give_qty = min(2, my_inventory[give_name])
            request_qty = min(2, their_inventory[request_name])
            candidates.append(
                f"{prefix}[Offer: {give_qty} {give_name} -> {request_qty} {request_name}]"
            )
            if len(candidates) >= 6:
                return candidates
    return candidates


def negotiation_parse_available_actions(observation: str, env: Any | None = None) -> list[str]:
    del env
    valid_actions: list[str] = []

    player_match = re.search(r"You are Player (\d+)", observation)
    if not player_match:
        return []

    player_id = int(player_match.group(1))
    other_player_id = 1 - player_id

    inventories: dict[int, dict[str, int]] = {}
    for match in re.finditer(r"- Player (\d+) inventory:\s*(.+)", observation):
        inventories[int(match.group(1))] = _parse_public_inventory_text(match.group(2))

    my_inventory = inventories.get(player_id, {})
    their_inventory = inventories.get(other_player_id, {})
    pending_offer_to_me = "Pending offer you must resolve now:" in observation

    if pending_offer_to_me:
        valid_actions.extend(
            [
                "[Accept]",
                "[Accept] Agreed.",
                "[Deny]",
                "[Deny] I need a different split.",
            ]
        )
        valid_actions.extend(
            _build_negotiation_offer_candidates(
                my_inventory=my_inventory,
                their_inventory=their_inventory,
                prefix="[Deny] ",
            )
        )
    else:
        valid_actions.extend(
            [
                "What trade helps you most?",
                "I'm open to a trade that improves both sides.",
            ]
        )
        valid_actions.extend(
            _build_negotiation_offer_candidates(
                my_inventory=my_inventory,
                their_inventory=their_inventory,
            )
        )

    deduped: list[str] = []
    seen: set[str] = set()
    for action in valid_actions:
        key = action.strip().lower()
        if not key or key in seen:
            continue
        seen.add(key)
        deduped.append(action)
    return deduped
