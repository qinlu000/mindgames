#!/usr/bin/env bash

resolve_default_qwen3_8b_model() {
  if [ -d "/workspace/models/Qwen3-8B" ]; then
    printf '%s\n' "/workspace/models/Qwen3-8B"
  else
    printf '%s\n' "Qwen/Qwen3-8B"
  fi
}

resolve_swift_cmd() {
  local swift_bin="${1:-}"
  local -n out_ref="$2"
  if [ -n "$swift_bin" ]; then
    out_ref=("$swift_bin")
  elif [ -x ".venv-grpo/bin/swift" ]; then
    out_ref=(".venv-grpo/bin/swift")
  elif [ -x ".venv/bin/swift" ]; then
    out_ref=(".venv/bin/swift")
  elif command -v uv >/dev/null 2>&1; then
    out_ref=(uv run swift)
  elif command -v swift >/dev/null 2>&1; then
    out_ref=(swift)
  else
    echo "swift not found. Install ms-swift first." >&2
    return 1
  fi
}

resolve_python_cmd() {
  local python_bin="${1:-}"
  local -n out_ref="$2"
  if [ -n "$python_bin" ]; then
    out_ref=("$python_bin")
  elif [ -x ".venv-grpo/bin/python" ]; then
    out_ref=(".venv-grpo/bin/python")
  elif [ -x ".venv/bin/python" ]; then
    out_ref=(".venv/bin/python")
  elif command -v uv >/dev/null 2>&1; then
    out_ref=(uv run python)
  elif command -v python >/dev/null 2>&1; then
    out_ref=(python)
  elif command -v python3 >/dev/null 2>&1; then
    out_ref=(python3)
  else
    echo "python not found." >&2
    return 1
  fi
}

is_true() {
  case "${1:-}" in
    1|true|TRUE|yes|YES|on|ON) return 0 ;;
    *) return 1 ;;
  esac
}

is_pos_int() {
  [[ "${1:-}" =~ ^[1-9][0-9]*$ ]]
}

parse_list() {
  local raw="${1:-}"
  local -n out_ref="$2"
  raw="${raw//,/ }"
  out_ref=()
  if [ -n "${raw//[[:space:]]/}" ]; then
    # shellcheck disable=SC2206
    out_ref=($raw)
  fi
}

append_arg_if_set() {
  local -n cmd_ref="$1"
  local flag="$2"
  local value="${3:-}"
  if [ -n "$value" ]; then
    cmd_ref+=("$flag" "$value")
  fi
}

append_list_arg_if_any() {
  local -n cmd_ref="$1"
  local flag="$2"
  local -n values_ref="$3"
  if [ "${#values_ref[@]}" -gt 0 ]; then
    cmd_ref+=("$flag" "${values_ref[@]}")
  fi
}

count_csv_items() {
  local raw="${1:-}"
  if [ -z "$raw" ]; then
    echo 0
    return
  fi
  local item
  local count=0
  IFS=',' read -r -a _items <<< "$raw"
  for item in "${_items[@]}"; do
    if [ -n "${item//[[:space:]]/}" ]; then
      count=$((count + 1))
    fi
  done
  echo "$count"
}

gpu_count() {
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo 0
    return
  fi
  nvidia-smi -L 2>/dev/null | wc -l | tr -d ' '
}

build_range_csv() {
  local start="$1"
  local end="$2"
  local values=()
  local i
  for ((i = start; i <= end; i++)); do
    values+=("$i")
  done
  local out
  out="$(IFS=,; echo "${values[*]}")"
  echo "$out"
}
