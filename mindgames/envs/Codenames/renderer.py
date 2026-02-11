from typing import Dict, Optional, Set


def create_board_str(
    board: Dict[str, str],
    guessed_words: Set[str],
    current_player_id: int,
    last_clue: Optional[str],
    last_number: int,
    remaining_guesses: int,
) -> str:
    is_spymaster_view = current_player_id in {0, 2}

    lines = ["Codenames Board:"]
    if last_clue is None:
        lines.append("Last clue: none")
    else:
        lines.append(f"Last clue: [{last_clue} {last_number}] | Remaining guesses: {remaining_guesses}")
    lines.append("")

    for index, word in enumerate(board.keys(), start=1):
        team = board[word]
        revealed = word in guessed_words
        if is_spymaster_view:
            visibility = "revealed" if revealed else "hidden"
            lines.append(f"{index:>2}. {word:<12} [{team}] {visibility}")
        else:
            visible_team = team if revealed else "?"
            visibility = "revealed" if revealed else ""
            lines.append(f"{index:>2}. {word:<12} [{visible_team}] {visibility}".rstrip())

    return "\n".join(lines)

