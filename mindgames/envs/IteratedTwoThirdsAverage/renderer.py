from typing import Any, Dict


def create_board_str(game_state: Dict[str, Any]) -> str:
    round_idx = int(game_state.get("round", 1) or 1)
    num_rounds = int(game_state.get("num_rounds", 1) or 1)
    points = game_state.get("points", {0: 0, 1: 0}) or {0: 0, 1: 0}
    history = list(game_state.get("history", []) or [])

    lines: list[str] = []
    lines.append(f"Round {round_idx}/{num_rounds}")
    lines.append(f"Score: Player 0 = {points.get(0, 0)}, Player 1 = {points.get(1, 0)}")
    if history:
        lines.append("")
        lines.append("History (completed rounds):")
        for i, past in enumerate(history, start=1):
            g0 = past.get(0, "?")
            g1 = past.get(1, "?")
            target = past.get("target", None)
            winner = past.get("winner", None)
            target_str = f"{float(target):.2f}" if isinstance(target, (int, float)) else "?"
            if winner is None:
                outcome = "draw"
            else:
                outcome = f"Player {winner} won"
            lines.append(f"  Round {i}: P0→{g0}, P1→{g1}, target={target_str} ({outcome})")
    return "\n".join(lines)

