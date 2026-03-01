#!/usr/bin/env bash
set -euo pipefail

# Monitor a Hanabi batch run every 30 minutes and auto-generate a final report
# when all expected model summaries are present.
#
# Usage:
#   OUT_ROOT=outputs/hanabi_merged_4x100_YYYYMMDD_HHMMSS \
#   bash tools/rollout/watch_hanabi_batch_30m.sh
#
# Optional env vars:
#   EXPECT_MODELS=4
#   INTERVAL_SEC=1800
#   MONITOR_LOG=<default: ${OUT_ROOT}/monitor_30m.log>
#   REPORT_JSON=<default: ${OUT_ROOT}/final_report.json>
#   REPORT_MD=<default: ${OUT_ROOT}/final_report.md>

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

OUT_ROOT="${OUT_ROOT:-}"
EXPECT_MODELS="${EXPECT_MODELS:-4}"
INTERVAL_SEC="${INTERVAL_SEC:-1800}"

if [[ -z "$OUT_ROOT" ]]; then
  echo "OUT_ROOT is required." >&2
  exit 1
fi
if [[ ! -d "$OUT_ROOT" ]]; then
  echo "OUT_ROOT not found: $OUT_ROOT" >&2
  exit 1
fi

MONITOR_LOG="${MONITOR_LOG:-$OUT_ROOT/monitor_30m.log}"
REPORT_JSON="${REPORT_JSON:-$OUT_ROOT/final_report.json}"
REPORT_MD="${REPORT_MD:-$OUT_ROOT/final_report.md}"

log() {
  printf '[%s] %s\n' "$(date '+%F %T')" "$*" | tee -a "$MONITOR_LOG"
}

count_summaries() {
  find "$OUT_ROOT" -maxdepth 2 -type f -name 'summary.json' | wc -l
}

count_active_rollout_procs() {
  pgrep -af "tools/rollout/run_rollouts.py" | rg -F "$OUT_ROOT" | wc -l || true
}

log "monitor started: OUT_ROOT=$OUT_ROOT EXPECT_MODELS=$EXPECT_MODELS INTERVAL_SEC=$INTERVAL_SEC"

while true; do
  summaries="$(count_summaries)"
  active_rollouts="$(count_active_rollout_procs)"
  leaderboard="no"
  [[ -f "$OUT_ROOT/leaderboard.json" ]] && leaderboard="yes"

  log "status: summaries=${summaries}/${EXPECT_MODELS} active_rollouts=${active_rollouts} leaderboard=${leaderboard}"

  if [[ "$summaries" -ge "$EXPECT_MODELS" && -f "$OUT_ROOT/leaderboard.json" ]]; then
    log "all models completed. generating final report..."
    .venv/bin/python - "$OUT_ROOT" "$REPORT_JSON" "$REPORT_MD" <<'PY'
import json
import re
import sys
import time
from collections import Counter
from pathlib import Path

out_root = Path(sys.argv[1]).resolve()
report_json = Path(sys.argv[2]).resolve()
report_md = Path(sys.argv[3]).resolve()

pat_play = re.compile(r"^\s*\[\s*Play\s*\]\s*\d+\s*$", re.I)
pat_discard = re.compile(r"^\s*\[\s*Discard\s*\]\s*\d+\s*$", re.I)
pat_reveal_color = re.compile(
    r"^\s*\[\s*Reveal\s*\]\s*player\s+\d+\s+card\s+\d+\s+color\s+(white|yellow|green|blue|red)\s*$", re.I
)
pat_reveal_rank = re.compile(
    r"^\s*\[\s*Reveal\s*\]\s*player\s+\d+\s+card\s+\d+\s+rank\s+[1-5]\s*$", re.I
)

def valid_action(s: str) -> bool:
    if not isinstance(s, str):
        return False
    return bool(
        pat_play.match(s)
        or pat_discard.match(s)
        or pat_reveal_color.match(s)
        or pat_reveal_rank.match(s)
    )

def parse_summary(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    envs = data.get("envs") or {}
    env_id = next(iter(envs)) if envs else "UNKNOWN"
    env = envs.get(env_id) or {}
    bp = env.get("by_player") or {}
    p0 = bp.get("0") or {}
    p1 = bp.get("1") or {}
    return {
        "env_id": env_id,
        "episodes": env.get("episodes"),
        "avg_score_if_coop": env.get("avg_score_if_coop"),
        "perfect_score_rate_if_coop": env.get("perfect_score_rate_if_coop"),
        "p0_invalid_rate": p0.get("invalid_rate"),
        "p1_invalid_rate": p1.get("invalid_rate"),
        "end_reasons": env.get("end_reasons") or {},
    }

def parse_rollouts(path: Path) -> dict:
    steps = 0
    episodes = 0
    invalid_norm = 0
    raw_has_think = 0
    raw_unclosed_think = 0
    same_player_consecutive = 0
    invalid_move_episodes = 0
    bad_norm_counter = Counter()

    prev_pid_by_ep = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            tp = rec.get("type")
            if tp == "episode_end":
                episodes += 1
                gi = rec.get("game_info") or {}
                bad = False
                for key in ("0", "1", 0, 1):
                    info = gi.get(key)
                    if isinstance(info, dict) and info.get("invalid_move"):
                        bad = True
                if bad:
                    invalid_move_episodes += 1
                continue
            if tp != "step":
                continue

            steps += 1
            ep = rec.get("episode_id")
            pid = rec.get("player_id")
            if ep in prev_pid_by_ep and prev_pid_by_ep[ep] == pid:
                same_player_consecutive += 1
            prev_pid_by_ep[ep] = pid

            raw = rec.get("raw_action") if isinstance(rec.get("raw_action"), str) else rec.get("action", "")
            low = raw.lower() if isinstance(raw, str) else ""
            if "<think>" in low:
                raw_has_think += 1
                if "</think>" not in low:
                    raw_unclosed_think += 1

            norm = rec.get("normalized_action") if isinstance(rec.get("normalized_action"), str) else ""
            if not valid_action(norm):
                invalid_norm += 1
                bad_norm_counter[norm.strip()[:120]] += 1

    return {
        "steps": steps,
        "episodes_from_rollouts": episodes,
        "invalid_norm_count": invalid_norm,
        "invalid_norm_rate": (invalid_norm / steps) if steps else None,
        "raw_has_think_count": raw_has_think,
        "raw_has_think_rate": (raw_has_think / steps) if steps else None,
        "raw_unclosed_think_count": raw_unclosed_think,
        "raw_unclosed_think_rate": (raw_unclosed_think / steps) if steps else None,
        "same_player_consecutive_count": same_player_consecutive,
        "same_player_consecutive_rate": (same_player_consecutive / steps) if steps else None,
        "invalid_move_episodes": invalid_move_episodes,
        "invalid_move_episode_rate": (invalid_move_episodes / episodes) if episodes else None,
        "top_bad_normalized_action": [
            {"action": k, "count": v} for k, v in bad_norm_counter.most_common(10)
        ],
    }

model_dirs = sorted([p for p in out_root.glob("*-merged") if p.is_dir()])
rows = []
for d in model_dirs:
    s_path = d / "summary.json"
    r_path = d / "rollouts.jsonl"
    row = {"model": d.name}
    if s_path.exists():
        row["summary"] = parse_summary(s_path)
    else:
        row["summary"] = None
    if r_path.exists():
        row["quality"] = parse_rollouts(r_path)
    else:
        row["quality"] = None
    rows.append(row)

def score_key(row: dict) -> float:
    s = (row.get("summary") or {}).get("avg_score_if_coop")
    return float(s) if isinstance(s, (int, float)) else -1.0

rows_sorted = sorted(rows, key=lambda x: (-score_key(x), x["model"]))

report = {
    "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    "out_root": str(out_root),
    "num_models_found": len(model_dirs),
    "models": rows_sorted,
}
report_json.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

lines = []
lines.append("# Hanabi Batch Final Report")
lines.append("")
lines.append(f"- generated_at: {report['generated_at']}")
lines.append(f"- out_root: `{out_root}`")
lines.append(f"- models_found: {len(model_dirs)}")
lines.append("")
lines.append("## Score Ranking")
lines.append("")
lines.append("| model | episodes | avg_score_if_coop | perfect_rate | p0_invalid | p1_invalid |")
lines.append("|---|---:|---:|---:|---:|---:|")
for row in rows_sorted:
    s = row.get("summary") or {}
    lines.append(
        "| {model} | {episodes} | {avg:.3f} | {perf:.3f} | {p0:.3f} | {p1:.3f} |".format(
            model=row["model"],
            episodes=s.get("episodes", 0) or 0,
            avg=float(s.get("avg_score_if_coop") or 0.0),
            perf=float(s.get("perfect_score_rate_if_coop") or 0.0),
            p0=float(s.get("p0_invalid_rate") or 0.0),
            p1=float(s.get("p1_invalid_rate") or 0.0),
        )
    )
lines.append("")
lines.append("## Action Quality")
lines.append("")
lines.append("| model | steps | invalid_norm_rate | unclosed_think_rate | same_player_consecutive_rate | invalid_move_episode_rate |")
lines.append("|---|---:|---:|---:|---:|---:|")
for row in rows_sorted:
    q = row.get("quality") or {}
    lines.append(
        "| {model} | {steps} | {inv:.3f} | {unclosed:.3f} | {same:.3f} | {inv_ep:.3f} |".format(
            model=row["model"],
            steps=q.get("steps", 0) or 0,
            inv=float(q.get("invalid_norm_rate") or 0.0),
            unclosed=float(q.get("raw_unclosed_think_rate") or 0.0),
            same=float(q.get("same_player_consecutive_rate") or 0.0),
            inv_ep=float(q.get("invalid_move_episode_rate") or 0.0),
        )
    )
lines.append("")
lines.append("## End Reasons")
lines.append("")
for row in rows_sorted:
    lines.append(f"### {row['model']}")
    reasons = (row.get("summary") or {}).get("end_reasons") or {}
    if not reasons:
        lines.append("- (no data)")
        lines.append("")
        continue
    for k, v in reasons.items():
        lines.append(f"- {k}: {v}")
    lines.append("")

report_md.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
print(f"report_json={report_json}")
print(f"report_md={report_md}")
PY
    log "report generated: $REPORT_MD"
    log "report generated: $REPORT_JSON"
    break
  fi

  sleep "$INTERVAL_SEC"
done

log "monitor finished"

