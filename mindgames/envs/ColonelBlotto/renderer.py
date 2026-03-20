from typing import Any, Dict, List


def create_board_str(game_state: Dict[str, Any], *, num_rounds: int, num_total_units: int) -> str:
    current_round = int(game_state.get("current_round", 1))
    scores = game_state.get("scores", {})
    fields: List[Dict[str, Any]] = list(game_state.get("fields", []) or [])
    field_names = [str(field.get("name", "?")) for field in fields]
    example_parts: List[str] = []
    remaining_units = num_total_units
    for idx, field_name in enumerate(field_names):
        if idx == len(field_names) - 1:
            units = remaining_units
        else:
            units = 1
            remaining_units -= units
        example_parts.append(f"{field_name}{units}")
    example_allocation = "[" + " ".join(example_parts) + "]"

    lines = [
        f"=== COLONEL BLOTTO - Round {current_round}/{num_rounds} ===",
        (
            "Rounds Won - "
            f"Commander Alpha: {int(scores.get(0, 0))}, "
            f"Commander Beta: {int(scores.get(1, 0))}"
        ),
        f"Available fields: {', '.join(field_names)}",
        f"Units to allocate: {num_total_units}",
        f"Format: '{example_allocation}'.",
    ]
    return "\n".join(lines)
