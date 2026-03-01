#!/usr/bin/env bash
set -euo pipefail

cd /home/cql/projects/games/mindgames

WATCH_SESSION="${WATCH_SESSION:-hanabi_wait_torch28_then_sft}"
TRAIN_SESSION="${TRAIN_SESSION:-hanabi_sft_ep16_auto}"
CHECK_INTERVAL_SEC="${CHECK_INTERVAL_SEC:-900}" # 15 min

BASE_RUN="${BASE_RUN:-output/qwen3-8b-hanabi-ge9-think-lora-4gpu-ws1-maxlen8192-right/v1-20260225-011119}"
CKPT="${CKPT:-$BASE_RUN/checkpoint-765}"
RUN_ID="${RUN_ID:-auto16-$(date +%Y%m%d-%H%M%S)}"
TRAIN_LOG="${TRAIN_LOG:-output/logs/${TRAIN_SESSION}_${RUN_ID}.log}"
WATCH_LOG="${WATCH_LOG:-output/logs/${WATCH_SESSION}.log}"

mkdir -p output/logs

echo "[$(date '+%F %T')] watcher started, interval=${CHECK_INTERVAL_SEC}s" | tee -a "$WATCH_LOG"
echo "[$(date '+%F %T')] train_log=$TRAIN_LOG" | tee -a "$WATCH_LOG"

while true; do
  ts="$(date '+%F %T')"

  if tmux has-session -t "$TRAIN_SESSION" 2>/dev/null; then
    echo "[$ts] train session '$TRAIN_SESSION' already exists, watcher exits." | tee -a "$WATCH_LOG"
    exit 0
  fi

  if pgrep -af "swift sft .*qwen3-8b-hanabi-ge9-think-lora-4gpu-ws1-maxlen8192-right-ep16-resume" >/dev/null; then
    echo "[$ts] found existing matching swift sft process, watcher exits." | tee -a "$WATCH_LOG"
    exit 0
  fi

  status_line="$(
    .venv/bin/python - <<'PY' 2>/dev/null || true
try:
    import torch
    tver = torch.__version__
except Exception:
    tver = ''

fa_ok = False
if tver.startswith('2.8'):
    try:
        import flash_attn  # noqa: F401
        fa_ok = True
    except Exception:
        fa_ok = False

print(f"{tver}|{int(fa_ok)}")
PY
  )"
  torch_ver="${status_line%%|*}"
  fa_ok_flag="${status_line##*|}"

  if [[ "$torch_ver" == 2.8* && "$fa_ok_flag" == "1" ]]; then
    echo "[$ts] torch+flash_attn ready ($torch_ver), starting SFT." | tee -a "$WATCH_LOG"

    tmux new -d -s "$TRAIN_SESSION" "cd /home/cql/projects/games/mindgames && \
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy && \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:128 \
CUDA_VISIBLE_DEVICES=0,1,2,3 NPROC_PER_NODE=4 NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 \
WANDB_PROJECT=mindgames WANDB_MODE=offline \
.venv/bin/swift sft \
  --model Qwen/Qwen3-8B \
  --dataset data/hanabi_qwen3_235b_sft_score_ge9_think.jsonl \
  --template qwen3 \
  --train_type lora \
  --output_dir $BASE_RUN \
  --resume_from_checkpoint $CKPT \
  --num_train_epochs 16 \
  --max_steps -1 \
  --max_length 8192 \
  --truncation_strategy right \
  --per_device_train_batch_size 1 \
  --learning_rate 1e-4 \
  --weight_decay 0.1 \
  --lr_scheduler_type cosine \
  --save_steps 500 \
  --logging_steps 5 \
  --attn_impl flash_attn \
  --report_to wandb \
  --run_name qwen3-8b-hanabi-ge9-think-lora-4gpu-ws1-maxlen8192-right-ep16-resume \
  2>&1 | tee $TRAIN_LOG"

    echo "[$(date '+%F %T')] started tmux session '$TRAIN_SESSION'" | tee -a "$WATCH_LOG"
    exit 0
  fi

  echo "[$ts] not ready (torch='${torch_ver:-none}', flash_attn_ok='${fa_ok_flag:-0}'), sleep ${CHECK_INTERVAL_SEC}s." | tee -a "$WATCH_LOG"
  sleep "$CHECK_INTERVAL_SEC"
done
