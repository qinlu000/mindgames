from typing import Any, Dict, List


def create_board_str(game_state: Dict[str, Any], *, num_rounds: int, num_total_units: int) -> str:
    current_round = int(game_state.get("current_round", 1))
    scores = game_state.get("scores", {})
    fields: List[Dict[str, Any]] = list(game_state.get("fields", []) or [])
    field_names = [str(field.get("name", "?")) for field in fields]

    lines = [
        f"=== COLONEL BLOTTO - Round {current_round}/{num_rounds} ===",
        (
            "Rounds Won - "
            f"Commander Alpha: {int(scores.get(0, 0))}, "
            f"Commander Beta: {int(scores.get(1, 0))}"
        ),
        f"Available fields: {', '.join(field_names)}",
        f"Units to allocate: {num_total_units}",
        "Format: '[A4 B2 C2]'.",
    ]
    return "\n".join(lines)
