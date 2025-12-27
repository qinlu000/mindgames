from typing import Any, Dict


def create_board_str(game_state: Dict[str, Any]) -> str:
    round_idx = int(game_state.get("round", 1) or 1)
    num_rounds = int(game_state.get("num_rounds", 1) or 1)
    points = game_state.get("points", {0: 0, 1: 0}) or {0: 0, 1: 0}
    history = list(game_state.get("history", []) or [])

    lines: list[str] = []
    lines.append(f"Round {round_idx}/{num_rounds}")
    if isinstance(points, dict):
        score_parts: list[str] = []
        for pid in sorted([p for p in points.keys() if isinstance(p, int)]):
            score_parts.append(f"Player {pid} = {points.get(pid, 0)}")
        if score_parts:
            lines.append("Score: " + ", ".join(score_parts))
        else:
            lines.append("Score: (unknown)")
    else:
        lines.append("Score: (unknown)")
    if history:
        lines.append("")
        lines.append("History (completed rounds):")
        for i, past in enumerate(history, start=1):
            if not isinstance(past, dict):
                lines.append(f"  Round {i}: (invalid record)")
                continue

            guess_items = [(k, v) for k, v in past.items() if isinstance(k, int)]
            guess_items.sort(key=lambda kv: kv[0])
            guess_str = ", ".join(f"P{pid}→{guess}" for pid, guess in guess_items) if guess_items else "?"
            target = past.get("target", None)
            winner = past.get("winner", None)
            winners = past.get("winners", None)
            target_str = f"{float(target):.2f}" if isinstance(target, (int, float)) else "?"
            if isinstance(winners, list) and winners:
                outcome = "draw" if len(winners) != 1 else f"Player {winners[0]} won"
            elif winner is None:
                outcome = "draw"
            else:
                outcome = f"Player {winner} won"

            lines.append(f"  Round {i}: {guess_str}, target={target_str} ({outcome})")
    return "\n".join(lines)
