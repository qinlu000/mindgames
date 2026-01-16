#!/usr/bin/env python3
"""
Phase-1 experiment registry tool for the `mindgames/` subproject.

Goals:
- Experiment config lives in `mindgames/experiments/**/experiment.yaml` (source of truth).
- Validate required fields and render reproducible `cmd.sh` with a stable `run_id`.
- DOES NOT execute training/eval (Phase 2 would add executors).

Usage (recommended with uv):
  cd mindgames
  uv run python tools/expctl.py init --template rollout_eval --name hanabi_baseline
  uv run python tools/expctl.py prepare experiments/hanabi_baseline/experiment.yaml
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shlex
import subprocess
import sys
import time
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping

import yaml
from jsonschema import Draft202012Validator


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"
SCHEMA_PATH = EXPERIMENTS_DIR / "_schema" / "experiment.schema.json"
TEMPLATES_DIR = EXPERIMENTS_DIR / "templates"


_TEMPLATE_TOKEN_RE = re.compile(r"{([^{}]+)}")


def _deep_get(d: Mapping[str, Any], dotted: str) -> Any:
    cur: Any = d
    for part in dotted.split("."):
        if not isinstance(cur, Mapping) or part not in cur:
            raise KeyError(dotted)
        cur = cur[part]
    return cur


def _render_template(s: str, ctx: Mapping[str, Any]) -> str:
    # Support {a.b.c} dotted access by rewriting fields into a flat map.
    # We keep this simple: find "{...}" tokens, resolve each token via dotted lookup.
    def _stringify(v: Any) -> str:
        if isinstance(v, (dict, list)):
            return json.dumps(v, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return str(v)

    def _is_empty(v: Any) -> bool:
        return v is None or v == "" or v == [] or v == {}

    def repl(m: re.Match[str]) -> str:
        expr = m.group(1).strip()
        if not expr:
            return m.group(0)

        if expr.startswith("optsh:"):
            # Syntax: {optsh:<flag> <dotted_path>}
            # Emits "<flag> <shell_quoted(value)>" only when value is non-empty.
            rest = expr[len("optsh:") :].strip()
            parts = rest.split(None, 1)
            if len(parts) != 2:
                raise KeyError(f"Invalid optsh expression: {expr}")
            flag, dotted = parts[0], parts[1].strip()
            try:
                val = _deep_get(ctx, dotted)
            except KeyError:
                return ""
            if _is_empty(val):
                return ""
            if isinstance(val, bool):
                return f"{flag} {shlex.quote('true' if val else 'false')}"
            return f"{flag} {shlex.quote(_stringify(val))}"
        if expr.startswith("opt:"):
            # Syntax: {opt:<flag> <dotted_path>}
            # Emits "<flag> <value>" only when value is non-empty.
            rest = expr[len("opt:") :].strip()
            parts = rest.split(None, 1)
            if len(parts) != 2:
                raise KeyError(f"Invalid opt expression: {expr}")
            flag, dotted = parts[0], parts[1].strip()
            try:
                val = _deep_get(ctx, dotted)
            except KeyError:
                return ""
            if _is_empty(val):
                return ""
            if isinstance(val, bool):
                return f"{flag} {'true' if val else 'false'}"
            return f"{flag} {_stringify(val)}"
        if expr.startswith("flag:"):
            # Syntax: {flag:<flag> <dotted_path>}
            # Emits "<flag>" only when value is truthy.
            rest = expr[len("flag:") :].strip()
            parts = rest.split(None, 1)
            if len(parts) != 2:
                raise KeyError(f"Invalid flag expression: {expr}")
            flag, dotted = parts[0], parts[1].strip()
            try:
                val = _deep_get(ctx, dotted)
            except KeyError:
                return ""
            return flag if bool(val) else ""
        if expr.startswith("sh:"):
            val = _deep_get(ctx, expr[len("sh:") :].strip())
            return shlex.quote(_stringify(val))
        val = _deep_get(ctx, expr)
        return _stringify(val)

    return _TEMPLATE_TOKEN_RE.sub(repl, s)


def _load_yaml(path: Path) -> Dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise SystemExit(f"Invalid YAML (expected mapping): {path}")
    return data


@lru_cache(maxsize=1)
def _load_schema() -> Draft202012Validator:
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    return Draft202012Validator(schema)


def _validate(exp: Dict[str, Any]) -> None:
    v = _load_schema()
    errs = sorted(v.iter_errors(exp), key=lambda e: list(e.path))
    if errs:
        lines = []
        for e in errs[:50]:
            loc = ".".join(str(x) for x in e.path) or "<root>"
            lines.append(f"- {loc}: {e.message}")
        raise SystemExit("experiment.yaml schema validation failed:\n" + "\n".join(lines))

    # Additional semantic checks (fast and useful)
    name = exp.get("name", "")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", str(name)):
        raise SystemExit("experiment.name must match [A-Za-z0-9_.-]+ (safe for paths)")

    if exp.get("kind") == "rollout_eval":
        spec = exp.get("spec") or {}
        if not isinstance(spec, dict):
            raise SystemExit("spec must be a mapping")
        for k in ["env_id", "num_players", "episodes", "agents"]:
            if k not in spec:
                raise SystemExit(f"rollout_eval requires spec.{k}")
        if not isinstance(spec.get("agents"), list) or not spec["agents"]:
            raise SystemExit("spec.agents must be a non-empty list")


def _agent_flags(spec: Mapping[str, Any]) -> str:
    agents = spec.get("agents") or []
    if not isinstance(agents, list):
        return ""
    agent_gen = spec.get("agent_gen") or []
    if agent_gen and not isinstance(agent_gen, list):
        raise SystemExit("spec.agent_gen must be a list when provided")
    parts: list[str] = []
    for i, a in enumerate(agents):
        a = str(a).strip()
        if not a:
            continue
        parts.append("--agent")
        parts.append(shlex.quote(a))
        if agent_gen:
            if i >= len(agent_gen):
                raise SystemExit(f"spec.agent_gen has {len(agent_gen)} entries but spec.agents has {len(agents)}")
            g = agent_gen[i]
            if g is None:
                continue
            if not isinstance(g, dict):
                raise SystemExit(f"spec.agent_gen[{i}] must be a mapping (dict) or null")
            parts.append("--agent-gen")
            parts.append(shlex.quote(json.dumps(g, ensure_ascii=False, sort_keys=True, separators=(',', ':'))))
    return " ".join(parts)


def _compute_run_id(resolved: Dict[str, Any]) -> str:
    # Hash only stable fields; exclude human notes and derived paths.
    stable = {
        "schema_version": resolved.get("schema_version"),
        "kind": resolved.get("kind"),
        "repro": resolved.get("repro"),
        "spec": resolved.get("spec"),
        "commands": resolved.get("_resolved_commands"),
        "outputs": resolved.get("outputs"),
    }
    payload = json.dumps(stable, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:12]


def _render_commands(exp: Dict[str, Any], run_id: str, run_dir: Path) -> list[str]:
    ctx: Dict[str, Any] = {
        "run_id": run_id,
        "run_dir": str(run_dir),
        "schema_version": exp.get("schema_version"),
        "name": exp.get("name"),
        "kind": exp.get("kind"),
        "repro": exp.get("repro") or {},
        "spec": exp.get("spec") or {},
        "agent_flags": _agent_flags(exp.get("spec") or {}),
    }
    rendered: list[str] = []
    for idx, c in enumerate(exp.get("commands", []), start=1):
        try:
            rendered.append(_render_template(str(c), ctx))
        except KeyError as e:
            key = str(e).strip("'")
            raise SystemExit(f"Template render failed in commands[{idx}]: missing key {key!r}") from e
    return rendered


def _write_cmd_sh(path: Path, commands: Iterable[str]) -> None:
    lines = ["#!/usr/bin/env bash", "set -euo pipefail", ""]
    lines.extend(commands)
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")
    os.chmod(path, 0o755)


def _inject_resume_flag(cmd: str) -> str:
    if "run_rollouts.py" not in cmd or "--resume" in cmd:
        return cmd
    return cmd + " --resume"


def _load_cmds_from_run_dir(run_dir: Path) -> list[str]:
    resolved_path = run_dir / "resolved.yaml"
    if resolved_path.exists():
        resolved = _load_yaml(resolved_path)
        cmds = resolved.get("_resolved_commands")
        if isinstance(cmds, list) and all(isinstance(c, str) for c in cmds):
            return list(cmds)

    cmd_sh = run_dir / "cmd.sh"
    if not cmd_sh.exists():
        raise SystemExit(f"Missing cmd.sh in {run_dir}")
    lines = cmd_sh.read_text(encoding="utf-8").splitlines()
    # Drop shebang + set -euo + blank lines
    return [ln for ln in lines if ln.strip() and not ln.startswith("#!") and not ln.startswith("set -")]


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _maybe_git_info(path: Path) -> Dict[str, Any]:
    try:
        # Suppress stderr to avoid noisy "not a git repository" messages when
        # running outside a work tree.
        commit = subprocess.check_output(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        dirty = bool(
            subprocess.check_output(
                ["git", "-C", str(path), "status", "--porcelain"],
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
        )
        return {"commit": commit, "dirty": dirty}
    except Exception:
        return {}


def _collect_env_fingerprint() -> Dict[str, Any]:
    fp: Dict[str, Any] = {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "cwd": os.getcwd(),
        "python": {"executable": sys.executable, "version": sys.version},
        "env": {
            "CUDA_VISIBLE_DEVICES": os.getenv("CUDA_VISIBLE_DEVICES"),
            "OPENAI_BASE_URL": os.getenv("OPENAI_BASE_URL"),
            "OPENAI_API_KEY_set": bool(os.getenv("OPENAI_API_KEY")),
            "HF_HOME": os.getenv("HF_HOME"),
            "HF_HUB_CACHE": os.getenv("HF_HUB_CACHE"),
        },
    }

    uv_lock = PROJECT_ROOT / "uv.lock"
    if uv_lock.exists():
        fp["uv_lock_sha256"] = _sha256_file(uv_lock)

    fp["git"] = _maybe_git_info(PROJECT_ROOT)

    try:
        import importlib.metadata as md

        pkgs = {}
        for name in ["vllm", "torch", "transformers", "openai"]:
            try:
                pkgs[name] = md.version(name)
            except Exception:
                continue
        if pkgs:
            fp["packages"] = pkgs
    except Exception:
        pass

    try:
        import torch

        torch_info: Dict[str, Any] = {
            "torch": getattr(torch, "__version__", None),
            "cuda": getattr(torch.version, "cuda", None),
            "cuda_available": bool(torch.cuda.is_available()),
            "device_count": int(torch.cuda.device_count()),
        }
        if torch_info["cuda_available"] and torch_info["device_count"] > 0:
            try:
                torch_info["device_0"] = torch.cuda.get_device_name(0)
            except Exception:
                pass
        fp["torch"] = torch_info
    except Exception:
        pass

    return fp


def _redact_secrets(obj: Any) -> Any:
    # Best-effort redaction to avoid accidentally logging API keys to W&B.
    redacted_keys = {
        "openai_api_key",
        "openrouter_api_key",
        "api_key",
        "token",
        "access_token",
        "secret",
        "password",
    }
    if isinstance(obj, dict):
        out: Dict[str, Any] = {}
        for k, v in obj.items():
            k_norm = str(k).lower()
            if k_norm in redacted_keys or k_norm.endswith("_api_key"):
                out[k] = "<redacted>"
            else:
                out[k] = _redact_secrets(v)
        return out
    if isinstance(obj, list):
        return [_redact_secrets(v) for v in obj]
    return obj


def _maybe_log_wandb(exp: Dict[str, Any], run_dir: Path, meta: Dict[str, Any]) -> None:
    wandb_cfg = exp.get("wandb")
    if not isinstance(wandb_cfg, dict):
        return
    if not wandb_cfg.get("project"):
        return
    if os.getenv("WANDB_MODE", "").lower() == "disabled":
        return

    try:
        import wandb
    except Exception:
        print("WARN: wandb not installed; skipping W&B logging.", file=sys.stderr)
        return

    resolved_path = run_dir / "resolved.yaml"
    resolved = _load_yaml(resolved_path) if resolved_path.exists() else {}
    run_id = str(resolved.get("run_id") or run_dir.name)

    project = str(wandb_cfg.get("project"))
    entity = wandb_cfg.get("entity") or os.getenv("WANDB_ENTITY")
    job_type = wandb_cfg.get("job_type") or exp.get("kind") or "run"
    group = wandb_cfg.get("group") or str(exp.get("name") or "")
    tags = wandb_cfg.get("tags") or exp.get("tags") or []
    if isinstance(tags, str):
        tags = [tags]
    if not isinstance(tags, list):
        tags = []

    run_name = wandb_cfg.get("name") or run_id
    config = _redact_secrets(
        {
            "experiment": {
                "name": exp.get("name"),
                "kind": exp.get("kind"),
                "description": exp.get("description"),
                "tags": exp.get("tags"),
                "path": str(meta.get("experiment_path") or ""),
                "run_id": run_id,
            },
            "repro": resolved.get("repro") or exp.get("repro"),
            "spec": resolved.get("spec") or exp.get("spec"),
        }
    )

    settings = wandb.Settings(save_code=False, disable_git=True)
    with wandb.init(
        project=project,
        entity=entity,
        job_type=job_type,
        group=(group or None),
        name=str(run_name),
        tags=tags,
        config=config,
        dir=str(run_dir),
        settings=settings,
    ) as run:
        run.summary.update(
            {
                "run_id": run_id,
                "exit_code": meta.get("exit_code"),
                "elapsed_s": meta.get("elapsed_s"),
            }
        )

        summary_path = run_dir / "summary.json"
        if summary_path.exists():
            try:
                summary = json.loads(summary_path.read_text(encoding="utf-8"))
                if isinstance(summary, dict):
                    metrics = summary.get("metrics")
                    if isinstance(metrics, dict):
                        run.summary.update(metrics)
                    run.summary.update({k: v for k, v in summary.items() if k != "metrics"})
            except Exception:
                pass

        artifact = wandb.Artifact(name=f"{exp.get('name','experiment')}-{run_id}", type="eval_results")
        for p in ["resolved.yaml", "cmd.sh", "run.log", "run_meta.json", "env_fingerprint.json", "summary.json"]:
            fp = run_dir / p
            if fp.exists():
                artifact.add_file(str(fp), name=p)

        outputs = (exp.get("outputs") or {}).get("files") if isinstance(exp.get("outputs"), dict) else None
        if isinstance(outputs, list):
            for pat in outputs:
                resolved_p = _render_template(str(pat), {"run_dir": str(run_dir), "run_id": run_id})
                fp = Path(resolved_p)
                if fp.exists() and fp.is_file():
                    artifact.add_file(str(fp), name=fp.name)

        run.log_artifact(artifact, aliases=["latest", run_id])


def _prepare_run(exp: Dict[str, Any], exp_path: Path, overwrite: bool) -> Path:
    _validate(exp)

    exp_dir = exp_path.parent
    runs_dir = exp_dir / "runs"
    if not runs_dir.exists():
        raise SystemExit(f"Missing runs/ directory: {runs_dir} (use expctl init or create it)")

    # First pass: provisional run_id to build run_dir, then render commands, then compute final run_id.
    provisional_id = "PROVISIONAL"
    provisional_run_dir = runs_dir / provisional_id
    resolved_cmds = _render_commands(exp, provisional_id, provisional_run_dir)

    resolved: Dict[str, Any] = dict(exp)
    resolved["_resolved_commands"] = resolved_cmds
    run_id = _compute_run_id(resolved)

    run_dir = runs_dir / run_id
    if run_dir.exists() and not overwrite:
        raise SystemExit(f"Run already exists: {run_dir} (use --overwrite to re-render)")
    run_dir.mkdir(parents=True, exist_ok=True)

    # Render again with final id/dir.
    final_cmds = _render_commands(exp, run_id, run_dir)
    resolved["_resolved_commands"] = final_cmds
    resolved["run_id"] = run_id
    resolved["run_dir"] = str(run_dir)

    (run_dir / "resolved.yaml").write_text(
        yaml.safe_dump(resolved, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    _write_cmd_sh(run_dir / "cmd.sh", final_cmds)

    # W&B export is Phase-1: generate JSON payload + artifact manifest, but do not upload.
    wandb_payload = {
        "run_id": run_id,
        "experiment_path": str(exp_path),
        "config": {k: v for k, v in resolved.items() if k not in {"commands", "_resolved_commands"}},
    }
    (run_dir / "wandb_config.json").write_text(
        json.dumps(wandb_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    outputs = (exp.get("outputs") or {}).get("files") if isinstance(exp.get("outputs"), dict) else None
    if isinstance(outputs, list):
        resolved_outputs = [_render_template(str(p), {"run_dir": str(run_dir), "run_id": run_id}) for p in outputs]
    else:
        resolved_outputs = []
    (run_dir / "artifacts.json").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "outputs": resolved_outputs,
                "suggested_wandb_artifacts": [{"type": "eval_results", "path": p} for p in resolved_outputs],
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    return run_dir


def cmd_init(args: argparse.Namespace) -> int:
    template_path = TEMPLATES_DIR / f"{args.template}.yaml"
    if not template_path.exists():
        raise SystemExit(f"Unknown template: {args.template} (expected {template_path})")

    exp_dir = EXPERIMENTS_DIR / args.name
    exp_dir.mkdir(parents=True, exist_ok=False)
    (exp_dir / "runs").mkdir()

    content = template_path.read_text(encoding="utf-8").replace("PLACEHOLDER_NAME", args.name)
    (exp_dir / "experiment.yaml").write_text(content, encoding="utf-8")
    print(str(exp_dir / "experiment.yaml"))
    return 0


def cmd_validate(args: argparse.Namespace) -> int:
    exp = _load_yaml(Path(args.path))
    _validate(exp)
    print("OK")
    return 0


def cmd_prepare(args: argparse.Namespace) -> int:
    exp_path = Path(args.path)
    exp = _load_yaml(exp_path)
    if getattr(args, "run_tag", None):
        spec = exp.get("spec")
        if not isinstance(spec, dict):
            spec = {}
            exp["spec"] = spec
        run_tag = str(args.run_tag)
        if run_tag in {"auto", "now"}:
            run_tag = time.strftime("%Y%m%d_%H%M%S")
        spec["run_tag"] = run_tag
    run_dir = _prepare_run(exp, exp_path, overwrite=bool(args.overwrite))
    print(str(run_dir))
    return 0


def cmd_print(args: argparse.Namespace) -> int:
    exp_path = Path(args.path)
    exp = _load_yaml(exp_path)
    if getattr(args, "run_tag", None):
        spec = exp.get("spec")
        if not isinstance(spec, dict):
            spec = {}
            exp["spec"] = spec
        run_tag = str(args.run_tag)
        if run_tag in {"auto", "now"}:
            run_tag = time.strftime("%Y%m%d_%H%M%S")
        spec["run_tag"] = run_tag
    _validate(exp)
    run_id = args.run_id or "DRYRUN"
    run_dir = Path(args.run_dir) if args.run_dir else (exp_path.parent / "runs" / run_id)
    for c in _render_commands(exp, run_id, run_dir):
        print(c)
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    exp_path = Path(args.path)
    exp = _load_yaml(exp_path)
    if getattr(args, "run_tag", None):
        spec = exp.get("spec")
        if not isinstance(spec, dict):
            spec = {}
            exp["spec"] = spec
        run_tag = str(args.run_tag)
        if run_tag in {"auto", "now"}:
            run_tag = time.strftime("%Y%m%d_%H%M%S")
        spec["run_tag"] = run_tag
    if args.resume:
        if args.run_dir:
            run_dir = Path(args.run_dir)
        elif args.run_id:
            run_dir = exp_path.parent / "runs" / args.run_id
        else:
            raise SystemExit("--resume requires --run-id or --run-dir")
        if not run_dir.exists():
            raise SystemExit(f"Run directory does not exist: {run_dir}")
    else:
        run_dir = _prepare_run(exp, exp_path, overwrite=bool(args.overwrite))

    (run_dir / "env_fingerprint.json").write_text(
        json.dumps(_collect_env_fingerprint(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    cmd_sh = run_dir / "cmd.sh"
    if args.resume:
        resume_cmds = [_inject_resume_flag(c) for c in _load_cmds_from_run_dir(run_dir)]
        cmd_sh = run_dir / "cmd_resume.sh"
        _write_cmd_sh(cmd_sh, resume_cmds)
    log_path = run_dir / "run.log"
    meta_path = run_dir / "run_meta.json"

    start = time.time()
    meta: Dict[str, Any] = {
        "experiment_path": str(exp_path),
        "run_dir": str(run_dir),
        "cmd_sh": str(cmd_sh),
        "argv": sys.argv,
        "started_at_s": start,
    }
    if args.resume:
        meta["resumed"] = True

    log_mode = "a" if args.resume and log_path.exists() else "w"
    with log_path.open(log_mode, encoding="utf-8") as logf:
        logf.write(f"# expctl run: {time.strftime('%Y-%m-%dT%H:%M:%S%z')}\n")
        logf.write(f"# experiment: {exp_path}\n")
        logf.write(f"# run_dir: {run_dir}\n")
        if args.resume:
            logf.write("# resume: true\n")
        logf.write("\n")
        logf.flush()

        proc = subprocess.run(
            ["bash", str(cmd_sh)],
            cwd=str(PROJECT_ROOT),
            stdout=logf,
            stderr=subprocess.STDOUT,
            text=True,
        )

    end = time.time()
    meta.update(
        {
            "ended_at_s": end,
            "elapsed_s": end - start,
            "exit_code": int(proc.returncode),
            "log_path": str(log_path),
        }
    )
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    if not getattr(args, "no_wandb", False):
        _maybe_log_wandb(exp, run_dir, meta)

    print(str(run_dir))
    return int(proc.returncode)


def main() -> int:
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)

    ap_init = sp.add_parser("init", help="Create experiments/<name>/ from a template")
    ap_init.add_argument("--template", default="rollout_eval", help="Template name (without .yaml)")
    ap_init.add_argument("--name", required=True, help="Experiment folder name (and experiment.name)")
    ap_init.set_defaults(func=cmd_init)

    ap_val = sp.add_parser("validate", help="Validate experiment.yaml against schema + semantic checks")
    ap_val.add_argument("path", help="Path to experiment.yaml")
    ap_val.set_defaults(func=cmd_validate)

    ap_prep = sp.add_parser("prepare", help="Create runs/<run_id>/ with cmd.sh + resolved.yaml + W&B export JSON")
    ap_prep.add_argument("path", help="Path to experiment.yaml")
    ap_prep.add_argument("--overwrite", action="store_true", help="Overwrite existing run files")
    ap_prep.add_argument(
        "--run-tag",
        default=None,
        help="Optional. If set, inject spec.run_tag to force a new run_id (use 'auto'/'now' for a timestamp).",
    )
    ap_prep.set_defaults(func=cmd_prepare)

    ap_print = sp.add_parser("print-cmd", help="Print rendered commands without creating a run directory")
    ap_print.add_argument("path", help="Path to experiment.yaml")
    ap_print.add_argument("--run-id", default=None)
    ap_print.add_argument("--run-dir", default=None)
    ap_print.add_argument(
        "--run-tag",
        default=None,
        help="Optional. Inject spec.run_tag while rendering (use 'auto'/'now' for a timestamp).",
    )
    ap_print.set_defaults(func=cmd_print)

    ap_run = sp.add_parser("run", help="Prepare + execute cmd.sh, capturing run.log + env_fingerprint.json")
    ap_run.add_argument("path", help="Path to experiment.yaml")
    ap_run.add_argument("--overwrite", action="store_true", help="Overwrite existing run files")
    ap_run.add_argument("--no-wandb", action="store_true", help="Disable W&B logging (even if experiment has wandb config)")
    ap_run.add_argument(
        "--run-tag",
        default=None,
        help="Optional. If set, inject spec.run_tag to force a new run_id (use 'auto'/'now' for a timestamp).",
    )
    ap_run.add_argument("--resume", action="store_true", help="Resume an existing run_dir (append to rollouts.jsonl).")
    ap_run.add_argument("--run-id", default=None, help="Run id to resume (uses experiments/<name>/runs/<run_id>).")
    ap_run.add_argument("--run-dir", default=None, help="Explicit run directory to resume.")
    ap_run.set_defaults(func=cmd_run)

    args = ap.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
