from __future__ import annotations

import argparse
import json
import os
import re
import threading
import time
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, List, Optional, Set
from urllib.parse import urlparse

import mindgames as mg


DEFAULT_SYSTEM_PROMPT = (
    "You are an expert Hanabi teammate.\n"
    "Output EXACTLY ONE valid action and nothing else (no reasoning).\n\n"
    "Valid formats:\n"
    "- [Play] X\n"
    "- [Discard] X\n"
    "- [Reveal] player N card X color C\n"
    "- [Reveal] player N card X rank R\n\n"
    "Rules (non-standard Hanabi here):\n"
    "- Reveal must target exactly ONE specific card index in another player's hand.\n"
    "- Reveal must be truthful for that specific card.\n"
    "- Do not reveal about yourself.\n"
    "- Use exactly one hint type: color OR rank.\n"
    "- Fireworks are independent; you may play the next required rank of any color.\n\n"
    "Strategy priority:\n"
    "1) If you know a card is playable, [Play] it.\n"
    "2) Else if a teammate has a clearly playable card and info_tokens>0, reveal that exact card.\n"
    "3) Else discard the least useful / most uncertain card.\n"
    "4) Avoid repeating the same Reveal on the same card unless it adds new info."
)

_REVEAL_COLOR_RE = re.compile(
    r"^\[Reveal\]\s+player\s+(?P<player>\d+)\s+card\s+(?P<card>\d+)\s+color\s+(?P<color>[a-zA-Z]+)\s*$",
    re.IGNORECASE,
)
_REVEAL_RANK_RE = re.compile(
    r"^\[Reveal\]\s+player\s+(?P<player>\d+)\s+card\s+(?P<card>\d+)\s+rank\s+(?P<rank>\d+)\s*$",
    re.IGNORECASE,
)
_OBS_REVEAL_COLOR_RE = re.compile(
    r"^Card\s+(?P<card>\d+)\s+from\s+player\s+(?P<player>\d+)\s+is\s+(?P<color>[a-zA-Z]+)\.\s*$",
    re.IGNORECASE,
)
_OBS_REVEAL_RANK_RE = re.compile(
    r"^Card\s+(?P<card>\d+)\s+from\s+player\s+(?P<player>\d+)\s+has\s+rank\s+(?P<rank>\d+)\.\s*$",
    re.IGNORECASE,
)


HTML_PAGE = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Hanabi Human + AI GUI</title>
  <style>
    :root {
      --bg: #f2efe8;
      --panel: #fff9ee;
      --ink: #1f2a30;
      --muted: #5f6a73;
      --line: #d8ccb5;
      --accent: #1d6f5f;
      --accent-2: #d6653f;
      --warn: #b12424;
      --card: #fffdf6;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: "Space Grotesk", "Avenir Next", "Segoe UI", sans-serif;
      color: var(--ink);
      background:
        radial-gradient(60rem 60rem at -20% -10%, #f7d69a66, transparent 55%),
        radial-gradient(50rem 50rem at 120% 20%, #8cc8bf66, transparent 50%),
        linear-gradient(150deg, #ece8de, #f8f5ec 45%, #efe9db);
      min-height: 100vh;
    }
    .wrap {
      max-width: 1200px;
      margin: 0 auto;
      padding: 20px;
      display: grid;
      grid-template-columns: 1.2fr 0.8fr;
      gap: 16px;
    }
    .panel {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 14px;
      box-shadow: 0 4px 14px #0000000d;
      padding: 14px;
    }
    h1 {
      margin: 0 0 10px;
      font-size: 1.3rem;
      letter-spacing: 0.01em;
    }
    h2 {
      margin: 0 0 10px;
      font-size: 1rem;
      color: var(--muted);
      text-transform: uppercase;
      letter-spacing: 0.05em;
    }
    .badge {
      display: inline-block;
      border: 1px solid var(--line);
      border-radius: 999px;
      padding: 3px 10px;
      margin-right: 8px;
      font-size: 0.85rem;
      background: var(--card);
    }
    .turn-human { color: var(--accent); font-weight: 700; }
    .turn-ai { color: var(--accent-2); font-weight: 700; }
    .warn { color: var(--warn); font-weight: 700; }
    .mono {
      font-family: "JetBrains Mono", "SFMono-Regular", Consolas, monospace;
      white-space: pre-wrap;
      word-break: break-word;
      margin: 0;
      background: var(--card);
      border: 1px solid var(--line);
      border-radius: 10px;
      padding: 10px;
      max-height: 300px;
      overflow: auto;
      font-size: 0.86rem;
      line-height: 1.35;
    }
    .grid {
      display: grid;
      gap: 10px;
      grid-template-columns: repeat(2, minmax(0, 1fr));
    }
    label {
      display: block;
      font-size: 0.86rem;
      color: var(--muted);
      margin-bottom: 4px;
    }
    select, button {
      width: 100%;
      border: 1px solid var(--line);
      border-radius: 10px;
      padding: 9px 10px;
      background: #fffef9;
      color: var(--ink);
      font: inherit;
    }
    button {
      cursor: pointer;
      background: linear-gradient(160deg, #fef2d5, #fae2b6);
    }
    button:hover { filter: brightness(0.98); }
    button:disabled {
      cursor: not-allowed;
      opacity: 0.55;
    }
    .quick {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
    }
    .quick button {
      width: auto;
      padding: 6px 10px;
    }
    .log {
      max-height: 280px;
      overflow: auto;
      border: 1px solid var(--line);
      border-radius: 10px;
      background: var(--card);
      padding: 8px;
      font-size: 0.84rem;
      font-family: "JetBrains Mono", "SFMono-Regular", Consolas, monospace;
    }
    .line { padding: 3px 0; border-bottom: 1px dashed #d7cfbf; }
    .line:last-child { border-bottom: 0; }
    .full { grid-column: 1 / -1; }
    @media (max-width: 980px) {
      .wrap { grid-template-columns: 1fr; }
    }
  </style>
</head>
<body>
  <div class="wrap">
    <section class="panel">
      <h1>Hanabi Human + AI</h1>
      <div id="status"></div>
      <div id="errors" class="warn" style="margin-top: 8px;"></div>

      <h2 style="margin-top: 14px;">Board</h2>
      <pre id="board" class="mono"></pre>

      <h2 style="margin-top: 14px;">Your Observation</h2>
      <pre id="observation" class="mono"></pre>

      <h2 style="margin-top: 14px;">Recent Steps</h2>
      <div id="steps" class="log"></div>
    </section>

    <section class="panel">
      <h2>Submit Action</h2>
      <div class="grid">
        <div class="full">
          <label for="actionType">Action Type</label>
          <select id="actionType">
            <option value="play">Play</option>
            <option value="discard">Discard</option>
            <option value="reveal_color">Reveal Color</option>
            <option value="reveal_rank">Reveal Rank</option>
          </select>
        </div>

        <div>
          <label for="cardIndex">Card Index</label>
          <select id="cardIndex"></select>
        </div>
        <div id="targetPlayerWrap">
          <label for="targetPlayer">Target Player</label>
          <select id="targetPlayer"></select>
        </div>

        <div id="targetCardWrap">
          <label for="targetCardIndex">Target Card Index</label>
          <select id="targetCardIndex"></select>
        </div>
        <div id="hintWrap">
          <label for="hintValue">Hint Value</label>
          <select id="hintValue"></select>
        </div>

        <div class="full">
          <button id="submitBtn" type="button">Submit Action</button>
        </div>
      </div>

      <h2 style="margin-top: 16px;">Quick Actions</h2>
      <div class="quick" id="quickActions"></div>

      <h2 style="margin-top: 16px;">Controls</h2>
      <button id="newGameBtn" type="button">New Game (Same Seed)</button>
      <button id="refreshBtn" type="button" style="margin-top: 8px;">Refresh Now</button>
    </section>
  </div>

  <script>
    let latestState = null;
    let refreshTimer = null;

    const statusEl = document.getElementById("status");
    const errorsEl = document.getElementById("errors");
    const boardEl = document.getElementById("board");
    const obsEl = document.getElementById("observation");
    const stepsEl = document.getElementById("steps");
    const actionTypeEl = document.getElementById("actionType");
    const cardIndexEl = document.getElementById("cardIndex");
    const targetPlayerEl = document.getElementById("targetPlayer");
    const targetCardIndexEl = document.getElementById("targetCardIndex");
    const hintValueEl = document.getElementById("hintValue");
    const submitBtn = document.getElementById("submitBtn");
    const newGameBtn = document.getElementById("newGameBtn");
    const refreshBtn = document.getElementById("refreshBtn");
    const quickActionsEl = document.getElementById("quickActions");
    const targetPlayerWrap = document.getElementById("targetPlayerWrap");
    const targetCardWrap = document.getElementById("targetCardWrap");
    const hintWrap = document.getElementById("hintWrap");

    function setError(msg) {
      errorsEl.textContent = msg || "";
    }

    function addOption(selectEl, value, label) {
      const opt = document.createElement("option");
      opt.value = String(value);
      opt.textContent = label;
      selectEl.appendChild(opt);
    }

    function clearOptions(selectEl) {
      while (selectEl.firstChild) selectEl.removeChild(selectEl.firstChild);
    }

    function renderStatus(state) {
      const turnClass = state.is_human_turn ? "turn-human" : "turn-ai";
      const turnText = state.done
        ? "Game Finished"
        : `Player ${state.current_player_id} (${state.is_human_turn ? "Human" : "AI"})`;
      const g = state.game_state || {};

      statusEl.innerHTML = `
        <span class="badge">Env: ${state.env_id}</span>
        <span class="badge">Seed: ${state.seed}</span>
        <span class="badge">Info: ${g.info_tokens ?? "-"}</span>
        <span class="badge">Fuse: ${g.fuse_tokens ?? "-"}</span>
        <span class="badge">Deck: ${g.deck_size ?? "-"}</span>
        <span class="badge ${turnClass}">Turn: ${turnText}</span>
      `;
    }

    function getRevealTargets(state) {
      const opts = state.action_options || {};
      return opts.reveal_targets || [];
    }

    function currentActionType() {
      return actionTypeEl.value;
    }

    function renderActionControls(state) {
      const opts = state.action_options || {};
      const playIdx = opts.play_indices || [];
      const discardIdx = opts.discard_indices || [];
      const revealTargets = getRevealTargets(state);
      const isHumanTurn = !!state.is_human_turn && !state.done;

      clearOptions(cardIndexEl);
      for (const idx of (currentActionType() === "discard" ? discardIdx : playIdx)) {
        addOption(cardIndexEl, idx, String(idx));
      }
      if (!cardIndexEl.options.length) {
        addOption(cardIndexEl, "", "-");
      }

      clearOptions(targetPlayerEl);
      for (const t of revealTargets) {
        addOption(targetPlayerEl, t.player_id, `Player ${t.player_id}`);
      }
      if (!targetPlayerEl.options.length) {
        addOption(targetPlayerEl, "", "-");
      }

      refreshRevealCardAndHintOptions();
      refreshActionVisibility();

      submitBtn.disabled = !isHumanTurn;
      actionTypeEl.disabled = !isHumanTurn;
      cardIndexEl.disabled = !isHumanTurn;
      targetPlayerEl.disabled = !isHumanTurn;
      targetCardIndexEl.disabled = !isHumanTurn;
      hintValueEl.disabled = !isHumanTurn;
    }

    function refreshActionVisibility() {
      const t = currentActionType();
      const isReveal = t.startsWith("reveal_");
      targetPlayerWrap.style.display = isReveal ? "block" : "none";
      targetCardWrap.style.display = isReveal ? "block" : "none";
      hintWrap.style.display = isReveal ? "block" : "none";
    }

    function refreshRevealCardAndHintOptions() {
      if (!latestState) return;
      const t = currentActionType();
      const isReveal = t.startsWith("reveal_");
      const targets = getRevealTargets(latestState);
      const selectedPlayer = Number(targetPlayerEl.value);
      const targetObj = targets.find((x) => x.player_id === selectedPlayer) || targets[0];

      clearOptions(targetCardIndexEl);
      clearOptions(hintValueEl);

      if (!isReveal || !targetObj) {
        addOption(targetCardIndexEl, "", "-");
        addOption(hintValueEl, "", "-");
        return;
      }

      for (const card of targetObj.cards || []) {
        addOption(targetCardIndexEl, card.index, `${card.index} (${card.label})`);
      }
      if (!targetCardIndexEl.options.length) {
        addOption(targetCardIndexEl, "", "-");
        addOption(hintValueEl, "", "-");
        return;
      }

      const selectedCardIdx = Number(targetCardIndexEl.value || targetCardIndexEl.options[0].value);
      const cardObj = (targetObj.cards || []).find((c) => c.index === selectedCardIdx) || targetObj.cards[0];
      if (!cardObj) {
        addOption(hintValueEl, "", "-");
        return;
      }

      if (t === "reveal_color") {
        addOption(hintValueEl, cardObj.color, cardObj.color);
      } else {
        addOption(hintValueEl, cardObj.rank, String(cardObj.rank));
      }
    }

    function renderQuickActions(state) {
      quickActionsEl.innerHTML = "";
      const opts = state.action_options || {};
      const isHumanTurn = !!state.is_human_turn && !state.done;
      const allButtons = [];

      for (const idx of (opts.play_indices || [])) {
        allButtons.push({ label: `Play ${idx}`, payload: { type: "play", card_index: idx } });
      }
      for (const idx of (opts.discard_indices || [])) {
        allButtons.push({ label: `Discard ${idx}`, payload: { type: "discard", card_index: idx } });
      }

      if (!allButtons.length) {
        const muted = document.createElement("div");
        muted.textContent = "No quick actions available.";
        muted.style.color = "#6f7a81";
        quickActionsEl.appendChild(muted);
        return;
      }

      for (const item of allButtons) {
        const btn = document.createElement("button");
        btn.type = "button";
        btn.textContent = item.label;
        btn.disabled = !isHumanTurn;
        btn.addEventListener("click", async () => {
          await sendAction(item.payload);
        });
        quickActionsEl.appendChild(btn);
      }
    }

    function renderSteps(state) {
      stepsEl.innerHTML = "";
      const steps = state.recent_steps || [];
      if (!steps.length) {
        const empty = document.createElement("div");
        empty.className = "line";
        empty.textContent = "No steps yet.";
        stepsEl.appendChild(empty);
        return;
      }
      for (const rec of steps) {
        const row = document.createElement("div");
        row.className = "line";
        const actor = rec.actor || "unknown";
        const norm = rec.normalized_action || rec.action || "";
        row.textContent = `#${rec.step} | p${rec.player_id} | ${actor} | ${norm}`;
        stepsEl.appendChild(row);
      }
    }

    function renderState(state) {
      latestState = state;
      renderStatus(state);
      boardEl.textContent = state.board || "";
      obsEl.textContent = state.observation || "";
      renderActionControls(state);
      renderQuickActions(state);
      renderSteps(state);
      setError(state.last_error || "");
    }

    async function fetchState() {
      try {
        const resp = await fetch("/state");
        const data = await resp.json();
        renderState(data);
      } catch (err) {
        setError(`Failed to fetch state: ${err}`);
      }
    }

    function buildActionPayload() {
      const t = currentActionType();
      if (t === "play") {
        return { type: "play", card_index: Number(cardIndexEl.value) };
      }
      if (t === "discard") {
        return { type: "discard", card_index: Number(cardIndexEl.value) };
      }
      if (t === "reveal_color" || t === "reveal_rank") {
        return {
          type: t,
          target_player: Number(targetPlayerEl.value),
          card_index: Number(targetCardIndexEl.value),
          hint_value: String(hintValueEl.value),
        };
      }
      return null;
    }

    async function sendAction(payload) {
      try {
        const resp = await fetch("/action", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(payload),
        });
        const data = await resp.json();
        if (!resp.ok || !data.ok) {
          setError(data.error || "Action rejected.");
          await fetchState();
          return;
        }
        setError("");
        renderState(data.state);
      } catch (err) {
        setError(`Failed to submit action: ${err}`);
      }
    }

    actionTypeEl.addEventListener("change", () => {
      refreshActionVisibility();
      renderActionControls(latestState || {});
    });
    targetPlayerEl.addEventListener("change", refreshRevealCardAndHintOptions);
    targetCardIndexEl.addEventListener("change", refreshRevealCardAndHintOptions);

    submitBtn.addEventListener("click", async () => {
      const payload = buildActionPayload();
      if (!payload) {
        setError("Unsupported action type.");
        return;
      }
      await sendAction(payload);
    });

    newGameBtn.addEventListener("click", async () => {
      try {
        const resp = await fetch("/new_game", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({}),
        });
        const data = await resp.json();
        if (!resp.ok) {
          setError(data.error || "Failed to start new game.");
          return;
        }
        setError("");
        renderState(data.state);
      } catch (err) {
        setError(`Failed to start new game: ${err}`);
      }
    });

    refreshBtn.addEventListener("click", fetchState);

    async function boot() {
      await fetchState();
      refreshTimer = setInterval(fetchState, 1200);
    }
    boot();
  </script>
</body>
</html>
"""


@dataclass
class AgentSpec:
    kind: str
    model: Optional[str] = None


def _parse_agent_spec(raw: str) -> AgentSpec:
    value = (raw or "").strip()
    if not value:
        raise ValueError("Agent spec is empty.")
    if ":" in value:
        kind, model = value.split(":", 1)
        kind = kind.strip().lower()
        model = model.strip()
        if not model:
            raise ValueError(f"Invalid agent spec (missing model): {raw!r}")
        return AgentSpec(kind=kind, model=model)
    return AgentSpec(kind=value.lower(), model=None)


def _parse_player_ids(raw: str, *, num_players: int) -> List[int]:
    ids: List[int] = []
    for token in raw.split(","):
        tok = token.strip()
        if not tok:
            continue
        try:
            pid = int(tok)
        except Exception as exc:
            raise ValueError(f"Invalid player id in --human-players: {tok!r}") from exc
        if pid < 0 or pid >= num_players:
            raise ValueError(f"Player id out of range: {pid} (expected 0..{num_players - 1})")
        if pid not in ids:
            ids.append(pid)
    return ids


def _build_agent(
    spec: AgentSpec,
    *,
    system_prompt: str,
    openai_api_key: Optional[str],
    openai_base_url: Optional[str],
    timeout_s: float,
    max_retries: int,
    retry_initial_delay_s: float,
    gen_kwargs: Dict[str, Any],
) -> mg.Agent:
    class _ScriptedAgent(mg.Agent):
        def __init__(self, action: str):
            self._action = action

        def __call__(self, observation: str) -> str:  # noqa: ARG002
            return self._action

    def _openai_like_kwargs() -> Dict[str, Any]:
        extra_body: Dict[str, Any] = dict(gen_kwargs.get("extra_body") or {})
        if gen_kwargs.get("chat_template_kwargs") is not None:
            extra_body["chat_template_kwargs"] = dict(gen_kwargs["chat_template_kwargs"])
        if gen_kwargs.get("top_k") is not None:
            extra_body["top_k"] = int(gen_kwargs["top_k"])

        kwargs: Dict[str, Any] = {}
        if gen_kwargs.get("temperature") is not None:
            kwargs["temperature"] = float(gen_kwargs["temperature"])
        if gen_kwargs.get("top_p") is not None:
            kwargs["top_p"] = float(gen_kwargs["top_p"])
        if gen_kwargs.get("max_tokens") is not None:
            kwargs["max_tokens"] = int(gen_kwargs["max_tokens"])
        if extra_body:
            kwargs["extra_body"] = extra_body
        if timeout_s and timeout_s > 0:
            kwargs["timeout"] = float(timeout_s)
        return kwargs

    if spec.kind == "scripted":
        if not spec.model:
            raise ValueError("scripted agent requires scripted:<name>.")
        name = spec.model.strip().lower()
        if name in {"hanabi_discard0", "hanabi_discard", "discard0", "discard"}:
            return _ScriptedAgent("[Discard] 0")
        if name.startswith("const="):
            return _ScriptedAgent(spec.model[len("const=") :])
        raise ValueError("Unknown scripted agent. Supported: hanabi_discard0 | const=<action>")

    if spec.kind == "openai":
        return mg.agents.OpenAIAgent(
            model_name=spec.model,  # type: ignore[arg-type]
            system_prompt=system_prompt,
            api_key=(openai_api_key or os.getenv("OPENAI_API_KEY")),
            base_url=(openai_base_url or os.getenv("OPENAI_BASE_URL")),
            max_retries=int(max_retries),
            retry_initial_delay_s=float(retry_initial_delay_s),
            **_openai_like_kwargs(),
        )

    if spec.kind == "qwen":
        return mg.agents.QwenAgent(
            model_name=spec.model,  # type: ignore[arg-type]
            system_prompt=system_prompt,
            api_key=(openai_api_key or os.getenv("OPENAI_API_KEY")),
            base_url=(openai_base_url or os.getenv("OPENAI_BASE_URL")),
            max_retries=int(max_retries),
            retry_initial_delay_s=float(retry_initial_delay_s),
            **_openai_like_kwargs(),
        )

    if spec.kind == "openrouter":
        kwargs: Dict[str, Any] = {
            "model_name": spec.model,  # type: ignore[arg-type]
            "system_prompt": system_prompt,
            "max_retries": int(max_retries),
            "retry_initial_delay_s": float(retry_initial_delay_s),
        }
        if gen_kwargs.get("temperature") is not None:
            kwargs["temperature"] = float(gen_kwargs["temperature"])
        if gen_kwargs.get("top_p") is not None:
            kwargs["top_p"] = float(gen_kwargs["top_p"])
        if gen_kwargs.get("max_tokens") is not None:
            kwargs["max_tokens"] = int(gen_kwargs["max_tokens"])
        return mg.agents.OpenRouterAgent(**kwargs)

    if spec.kind == "gemini":
        generation_config: Dict[str, Any] = {}
        if gen_kwargs.get("temperature") is not None:
            generation_config["temperature"] = float(gen_kwargs["temperature"])
        if gen_kwargs.get("top_p") is not None:
            generation_config["top_p"] = float(gen_kwargs["top_p"])
        if gen_kwargs.get("top_k") is not None:
            generation_config["top_k"] = int(gen_kwargs["top_k"])
        if gen_kwargs.get("max_tokens") is not None:
            generation_config["max_output_tokens"] = int(gen_kwargs["max_tokens"])
        return mg.agents.GeminiAgent(
            model_name=spec.model,  # type: ignore[arg-type]
            system_prompt=system_prompt,
            generation_config=generation_config,
        )

    if spec.kind == "ollama":
        kwargs = {
            "model_name": spec.model,  # type: ignore[arg-type]
            "system_prompt": system_prompt,
            "max_retries": int(max_retries),
            "retry_initial_delay_s": float(retry_initial_delay_s),
        }
        if gen_kwargs.get("temperature") is not None:
            kwargs["temperature"] = float(gen_kwargs["temperature"])
        if gen_kwargs.get("top_p") is not None:
            kwargs["top_p"] = float(gen_kwargs["top_p"])
        if gen_kwargs.get("max_tokens") is not None:
            kwargs["max_tokens"] = int(gen_kwargs["max_tokens"])
        return mg.agents.OllamaAgent(**kwargs)

    if spec.kind == "hf":
        agent = mg.agents.HFLocalAgent(model_name=spec.model)  # type: ignore[arg-type]
        agent.system_prompt = system_prompt
        return agent

    raise ValueError(
        f"Unsupported llm agent kind: {spec.kind}. "
        "Supported: scripted, openai, qwen, openrouter, gemini, ollama, hf"
    )


def _unwrap_env(env: mg.Env) -> mg.Env:
    current = env
    while isinstance(current, mg.Wrapper):
        current = current.env
    return current


def _normalize_action(env: mg.Env, action: str) -> str:
    normalized = action
    current = env
    while isinstance(current, mg.Wrapper):
        if isinstance(current, mg.ActionWrapper):
            normalized = current.action(normalized)
        current = current.env
    return normalized


def _parse_reveal_action(action: str) -> Optional[Dict[str, Any]]:
    txt = (action or "").strip()

    m = _REVEAL_COLOR_RE.match(txt)
    if m is not None:
        return {
            "target_player": int(m.group("player")),
            "card_index": int(m.group("card")),
            "hint_type": "color",
            "hint_value": m.group("color").lower(),
        }

    m = _REVEAL_RANK_RE.match(txt)
    if m is not None:
        return {
            "target_player": int(m.group("player")),
            "card_index": int(m.group("card")),
            "hint_type": "rank",
            "hint_value": int(m.group("rank")),
        }

    return None


def _observation_contains_reveal_message(
    observation_text: str,
    *,
    target_player: int,
    card_index: int,
    hint_type: str,
    hint_value: Any,
) -> bool:
    lines = (observation_text or "").splitlines()
    for raw in lines:
        line = raw.strip()

        if hint_type == "color":
            m = _OBS_REVEAL_COLOR_RE.match(line)
            if m is None:
                continue
            if int(m.group("player")) != int(target_player):
                continue
            if int(m.group("card")) != int(card_index):
                continue
            if m.group("color").lower() != str(hint_value).lower():
                continue
            return True

        elif hint_type == "rank":
            m = _OBS_REVEAL_RANK_RE.match(line)
            if m is None:
                continue
            if int(m.group("player")) != int(target_player):
                continue
            if int(m.group("card")) != int(card_index):
                continue
            if int(m.group("rank")) != int(hint_value):
                continue
            return True

    return False


class HanabiHumanAIGame:
    def __init__(
        self,
        *,
        env_id: str,
        env_kwargs: Optional[Dict[str, Any]],
        num_players: int,
        human_players: str,
        llm_agent: str,
        seed: int,
        system_prompt: str,
        openai_api_key: Optional[str],
        openai_base_url: Optional[str],
        timeout_s: float,
        max_retries: int,
        retry_initial_delay_s: float,
        temperature: Optional[float],
        top_p: Optional[float],
        top_k: Optional[int],
        max_tokens: Optional[int],
        disable_thinking: bool,
        max_auto_steps_per_tick: int = 200,
    ):
        self.env_id = env_id
        self.env_kwargs = dict(env_kwargs or {})
        self.num_players = int(num_players)
        self.seed = int(seed)
        self.system_prompt = system_prompt
        self.openai_api_key = openai_api_key
        self.openai_base_url = openai_base_url
        self.timeout_s = float(timeout_s)
        self.max_retries = int(max_retries)
        self.retry_initial_delay_s = float(retry_initial_delay_s)
        self.max_auto_steps_per_tick = int(max_auto_steps_per_tick)

        human_ids = _parse_player_ids(human_players, num_players=self.num_players)
        if not human_ids:
            raise ValueError("human_players must contain at least one seat id.")
        if len(human_ids) >= self.num_players:
            raise ValueError("human_players must leave at least one non-human seat.")
        self.human_players: Set[int] = set(human_ids)

        llm_spec = _parse_agent_spec(llm_agent)
        if llm_spec.kind == "human":
            raise ValueError("llm_agent cannot be human.")

        chat_template_kwargs = {"enable_thinking": False} if disable_thinking else None
        self.gen_kwargs: Dict[str, Any] = {
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "max_tokens": max_tokens,
            "chat_template_kwargs": chat_template_kwargs,
        }

        self.agents: Dict[int, Optional[mg.Agent]] = {}
        for pid in range(self.num_players):
            if pid in self.human_players:
                self.agents[pid] = None
            else:
                self.agents[pid] = _build_agent(
                    llm_spec,
                    system_prompt=self.system_prompt,
                    openai_api_key=self.openai_api_key,
                    openai_base_url=self.openai_base_url,
                    timeout_s=self.timeout_s,
                    max_retries=self.max_retries,
                    retry_initial_delay_s=self.retry_initial_delay_s,
                    gen_kwargs=self.gen_kwargs,
                )

        self.lock = threading.RLock()
        self.env: Optional[mg.Env] = None
        self.done = False
        self.rewards: Optional[Dict[int, float]] = None
        self.game_info: Optional[Dict[str, Any]] = None
        self.step_history: List[Dict[str, Any]] = []
        self._current_player_id: Optional[int] = None
        self._current_observation: Optional[str] = None
        self.last_error: Optional[str] = None

    def start_new_game(self, seed: Optional[int] = None) -> Dict[str, Any]:
        with self.lock:
            if seed is not None:
                self.seed = int(seed)
            self.env = mg.make(env_id=self.env_id, **self.env_kwargs)
            self.env.reset(num_players=self.num_players, seed=self.seed)
            self.done = False
            self.rewards = None
            self.game_info = None
            self.step_history = []
            self._current_player_id = None
            self._current_observation = None
            self.last_error = None
            self._advance_until_human_turn()
            return self._state_unlocked()

    def _ensure_observation_unlocked(self) -> None:
        if self.done or self.env is None:
            return
        if self._current_player_id is not None:
            return
        player_id, observation = self.env.get_observation()
        self._current_player_id = int(player_id)
        if isinstance(observation, str):
            self._current_observation = observation
        else:
            self._current_observation = str(observation)

    def _record_step_unlocked(
        self,
        *,
        player_id: int,
        actor: str,
        observation: str,
        action: str,
        infer_ms: int,
        reasoning: Optional[str],
        step_info: Dict[str, Any],
        done: bool,
    ) -> None:
        if self.env is None:
            return
        normalized = _normalize_action(self.env, action)
        rec = {
            "step": len(self.step_history),
            "player_id": int(player_id),
            "actor": actor,
            "observation": observation,
            "action": action,
            "normalized_action": normalized,
            "infer_ms": int(infer_ms),
            "reasoning": reasoning,
            "step_info": step_info,
            "done": bool(done),
        }
        self.step_history.append(rec)
        if len(self.step_history) > 400:
            self.step_history = self.step_history[-400:]

    def _take_action_unlocked(
        self,
        *,
        player_id: int,
        actor: str,
        action: str,
        infer_ms: int,
        reasoning: Optional[str],
    ) -> None:
        if self.env is None:
            raise RuntimeError("Environment is not initialized.")
        observation = self._current_observation or ""
        done, step_info = self.env.step(action=action)
        self._record_step_unlocked(
            player_id=player_id,
            actor=actor,
            observation=observation,
            action=action,
            infer_ms=infer_ms,
            reasoning=reasoning,
            step_info=step_info,
            done=done,
        )

        self._current_player_id = None
        self._current_observation = None
        self.done = bool(done)
        if self.done:
            rewards, game_info = self.env.close()
            self.rewards = rewards
            self.game_info = game_info

    def _advance_until_human_turn(self) -> None:
        if self.env is None or self.done:
            return
        auto_steps = 0
        while not self.done:
            self._ensure_observation_unlocked()
            if self._current_player_id is None:
                return

            if self._current_player_id in self.human_players:
                return

            agent = self.agents.get(self._current_player_id)
            if agent is None:
                self.last_error = f"Missing AI agent for player {self._current_player_id}."
                return

            t0 = time.time()
            action = agent(self._current_observation or "")
            infer_ms = int((time.time() - t0) * 1000)
            _, reasoning = agent.get_last_content_reasoning()
            self._take_action_unlocked(
                player_id=self._current_player_id,
                actor="ai",
                action=action,
                infer_ms=infer_ms,
                reasoning=reasoning,
            )
            auto_steps += 1
            if auto_steps >= self.max_auto_steps_per_tick and not self.done:
                self.last_error = (
                    f"Auto-play step limit reached ({self.max_auto_steps_per_tick}) before returning to a human turn."
                )
                return

    def _action_options_for_player_unlocked(self, player_id: int) -> Dict[str, Any]:
        if self.env is None:
            return {"play_indices": [], "discard_indices": [], "reveal_targets": []}
        base = _unwrap_env(self.env)
        state = getattr(base, "state", None)
        game_state = getattr(state, "game_state", None)
        if not isinstance(game_state, dict):
            return {"play_indices": [], "discard_indices": [], "reveal_targets": []}

        hands = game_state.get("player_hands") or {}
        hand = list(hands.get(player_id) or [])
        reveal_targets: List[Dict[str, Any]] = []
        for pid in range(self.num_players):
            if pid == player_id:
                continue
            cards = list(hands.get(pid) or [])
            serialized = []
            for idx, card in enumerate(cards):
                suit = getattr(card, "suit", None)
                color = getattr(suit, "value", None)
                rank = getattr(card, "rank", None)
                serialized.append(
                    {
                        "index": idx,
                        "label": str(card),
                        "color": color,
                        "rank": rank,
                    }
                )
            reveal_targets.append({"player_id": pid, "cards": serialized})

        return {
            "play_indices": list(range(len(hand))),
            "discard_indices": list(range(len(hand))),
            "reveal_targets": reveal_targets,
            "can_reveal": int(game_state.get("info_tokens", 0)) > 0,
        }

    def _build_action_from_payload_unlocked(self, payload: Dict[str, Any], player_id: int) -> str:
        raw = payload.get("raw_action")
        if isinstance(raw, str) and raw.strip():
            return raw.strip()

        action_type = str(payload.get("type", "")).strip().lower()
        options = self._action_options_for_player_unlocked(player_id)
        play_or_discard = set(int(x) for x in options.get("play_indices", []))
        reveal_targets = options.get("reveal_targets", [])

        def _to_int(name: str) -> int:
            value = payload.get(name)
            if value is None:
                raise ValueError(f"Missing field: {name}")
            try:
                return int(value)
            except Exception as exc:
                raise ValueError(f"Invalid integer field: {name}={value!r}") from exc

        if action_type == "play":
            idx = _to_int("card_index")
            if idx not in play_or_discard:
                raise ValueError(f"Invalid play card index: {idx}")
            return f"[Play] {idx}"

        if action_type == "discard":
            idx = _to_int("card_index")
            if idx not in play_or_discard:
                raise ValueError(f"Invalid discard card index: {idx}")
            return f"[Discard] {idx}"

        if action_type in {"reveal_color", "reveal_rank"}:
            if not bool(options.get("can_reveal")):
                raise ValueError("Reveal is invalid when info_tokens is 0.")
            target_player = _to_int("target_player")
            card_index = _to_int("card_index")
            hint_value = str(payload.get("hint_value", "")).strip().lower()
            target = None
            for item in reveal_targets:
                if int(item.get("player_id")) == target_player:
                    target = item
                    break
            if target is None:
                raise ValueError(f"Invalid target player: {target_player}")
            card = None
            for item in target.get("cards", []):
                if int(item.get("index")) == card_index:
                    card = item
                    break
            if card is None:
                raise ValueError(f"Invalid target card index: {card_index}")

            if action_type == "reveal_color":
                color = str(card.get("color", "")).strip().lower()
                if not color:
                    raise ValueError("Target card has no color info.")
                if hint_value and hint_value != color:
                    raise ValueError(f"Untruthful reveal color: {hint_value} (must be {color})")
                return f"[Reveal] player {target_player} card {card_index} color {color}"

            rank = int(card.get("rank"))
            if hint_value and str(rank) != hint_value:
                raise ValueError(f"Untruthful reveal rank: {hint_value} (must be {rank})")
            return f"[Reveal] player {target_player} card {card_index} rank {rank}"

        raise ValueError(f"Unsupported action type: {action_type!r}")

    def _previous_turn_hint_for_player_unlocked(self, player_id: int, observation_text: str) -> Optional[str]:
        if not self.step_history:
            return None

        last = self.step_history[-1]
        raw_actor = last.get("player_id")
        try:
            actor_id = int(raw_actor)
        except Exception:
            return None
        if actor_id == int(player_id):
            return None

        action = str(last.get("normalized_action") or last.get("action") or "").strip()
        parsed = _parse_reveal_action(action)
        if parsed is None:
            return None
        if int(parsed["target_player"]) != int(player_id):
            return None

        card_index = int(parsed["card_index"])
        hint_type = str(parsed["hint_type"])
        hint_value = parsed["hint_value"]
        if not _observation_contains_reveal_message(
            observation_text,
            target_player=int(player_id),
            card_index=card_index,
            hint_type=hint_type,
            hint_value=hint_value,
        ):
            return None

        if hint_type == "color":
            return f"Previous turn hint: Player {actor_id} revealed your card {card_index} color {hint_value}."
        return f"Previous turn hint: Player {actor_id} revealed your card {card_index} rank {hint_value}."

    def submit_action(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        with self.lock:
            if self.env is None:
                return {"ok": False, "error": "Game is not initialized."}
            if self.done:
                return {"ok": False, "error": "Game already ended.", "state": self._state_unlocked()}

            self._ensure_observation_unlocked()
            if self._current_player_id is None:
                return {"ok": False, "error": "No active player turn."}
            if self._current_player_id not in self.human_players:
                return {"ok": False, "error": f"It is not a human turn (current player={self._current_player_id})."}

            try:
                action = self._build_action_from_payload_unlocked(payload, self._current_player_id)
            except ValueError as exc:
                return {"ok": False, "error": str(exc), "state": self._state_unlocked()}

            self._take_action_unlocked(
                player_id=self._current_player_id,
                actor="human",
                action=action,
                infer_ms=0,
                reasoning=None,
            )
            self._advance_until_human_turn()
            return {"ok": True, "action": action, "state": self._state_unlocked()}

    def _state_unlocked(self) -> Dict[str, Any]:
        if self.env is None:
            return {"ok": False, "error": "Game is not initialized."}

        self._ensure_observation_unlocked()
        base = _unwrap_env(self.env)
        state = getattr(base, "state", None)
        game_state = getattr(state, "game_state", {}) if state is not None else {}

        current_player_id = self._current_player_id
        is_human_turn = (not self.done) and (current_player_id in self.human_players)
        observation_text = self._current_observation or ""
        previous_turn_hint: Optional[str] = None
        if is_human_turn and current_player_id is not None:
            previous_turn_hint = self._previous_turn_hint_for_player_unlocked(current_player_id, observation_text)
            if previous_turn_hint:
                observation_text = observation_text.rstrip()
                if observation_text:
                    observation_text = f"{observation_text}\n\n{previous_turn_hint}"
                else:
                    observation_text = previous_turn_hint

        fireworks_raw = game_state.get("fireworks", {})
        fireworks: Dict[str, int] = {}
        if isinstance(fireworks_raw, dict):
            for key, val in fireworks_raw.items():
                k = getattr(key, "value", str(key))
                fireworks[str(k)] = int(val)

        board = ""
        try:
            board = base.get_board_str()
        except Exception:
            board = ""

        action_options = (
            self._action_options_for_player_unlocked(current_player_id) if (current_player_id is not None and is_human_turn) else None
        )

        payload = {
            "ok": True,
            "env_id": self.env_id,
            "seed": self.seed,
            "num_players": self.num_players,
            "human_players": sorted(self.human_players),
            "done": self.done,
            "current_player_id": current_player_id,
            "is_human_turn": is_human_turn,
            "observation": observation_text,
            "previous_turn_hint": previous_turn_hint,
            "board": board,
            "game_state": {
                "info_tokens": int(game_state.get("info_tokens", 0)),
                "fuse_tokens": int(game_state.get("fuse_tokens", 0)),
                "deck_size": int(game_state.get("deck_size", 0)),
                "step_count": int(game_state.get("step_count", 0)),
                "discard_count": len(game_state.get("discard_pile", []) or []),
                "fireworks": fireworks,
            },
            "action_options": action_options,
            "recent_steps": self.step_history[-60:],
            "rewards": self.rewards,
            "game_info": self.game_info,
            "last_error": self.last_error,
        }
        return payload

    def get_public_state(self) -> Dict[str, Any]:
        with self.lock:
            return self._state_unlocked()


class HanabiGUIHandler(BaseHTTPRequestHandler):
    server_version = "HanabiHumanAI/0.1"

    def log_message(self, format: str, *args: object) -> None:
        return

    def _send_json(self, code: int, obj: Dict[str, Any]) -> None:
        raw = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def _read_json(self) -> Dict[str, Any]:
        length = int(self.headers.get("Content-Length") or "0")
        if length <= 0:
            return {}
        raw = self.rfile.read(length)
        if not raw:
            return {}
        try:
            data = json.loads(raw.decode("utf-8"))
        except Exception as exc:
            raise ValueError(f"Invalid JSON body: {exc}") from exc
        if not isinstance(data, dict):
            raise ValueError("JSON body must be an object.")
        return data

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/":
            raw = HTML_PAGE.encode("utf-8")
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)
            return

        if parsed.path == "/state":
            app: HanabiHumanAIGame = self.server.app  # type: ignore[attr-defined]
            self._send_json(HTTPStatus.OK, app.get_public_state())
            return

        self._send_json(HTTPStatus.NOT_FOUND, {"ok": False, "error": "Not found."})

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        app: HanabiHumanAIGame = self.server.app  # type: ignore[attr-defined]

        if parsed.path == "/action":
            try:
                payload = self._read_json()
            except ValueError as exc:
                self._send_json(HTTPStatus.BAD_REQUEST, {"ok": False, "error": str(exc)})
                return
            result = app.submit_action(payload)
            code = HTTPStatus.OK if result.get("ok") else HTTPStatus.BAD_REQUEST
            self._send_json(code, result)
            return

        if parsed.path == "/new_game":
            try:
                payload = self._read_json()
            except ValueError as exc:
                self._send_json(HTTPStatus.BAD_REQUEST, {"ok": False, "error": str(exc)})
                return
            seed = payload.get("seed")
            try:
                state = app.start_new_game(seed=None if seed is None else int(seed))
            except Exception as exc:
                self._send_json(HTTPStatus.BAD_REQUEST, {"ok": False, "error": str(exc)})
                return
            self._send_json(HTTPStatus.OK, {"ok": True, "state": state})
            return

        self._send_json(HTTPStatus.NOT_FOUND, {"ok": False, "error": "Not found."})


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Hanabi Human + AI web GUI")
    ap.add_argument("--env-id", default="Hanabi-v0-train")
    ap.add_argument("--env-kwargs", default=None, help="Optional JSON dict passed to env constructor.")
    ap.add_argument("--num-players", type=int, default=2)
    ap.add_argument("--human-players", default="0", help="Comma-separated human seat ids, e.g. 0 or 0,2")
    ap.add_argument(
        "--llm-agent",
        default="scripted:hanabi_discard0",
        help="Agent spec for all non-human seats. E.g. scripted:hanabi_discard0 or openai:gpt-4.1-mini",
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--system-prompt", default=DEFAULT_SYSTEM_PROMPT)
    ap.add_argument("--openai-base-url", default=None)
    ap.add_argument("--openai-api-key", default=None)
    ap.add_argument("--timeout", type=float, default=120.0)
    ap.add_argument("--max-retries", type=int, default=10)
    ap.add_argument("--retry-initial-delay", type=float, default=0.0)
    ap.add_argument("--temperature", type=float, default=None)
    ap.add_argument("--top-p", type=float, default=None)
    ap.add_argument("--top-k", type=int, default=None)
    ap.add_argument("--max-tokens", type=int, default=None)
    ap.add_argument(
        "--disable-thinking",
        action="store_true",
        help="Set chat_template_kwargs.enable_thinking=false for OpenAI-compatible backends that support it.",
    )
    ap.add_argument(
        "--max-auto-steps-per-tick",
        type=int,
        default=200,
        help="Safety guard: max consecutive AI steps processed in one server tick.",
    )
    return ap


def main(argv: Optional[List[str]] = None) -> int:
    ap = build_arg_parser()
    args = ap.parse_args(argv)

    if args.env_kwargs is not None:
        try:
            env_kwargs = json.loads(args.env_kwargs)
        except Exception as exc:
            raise SystemExit(f"Invalid --env-kwargs JSON: {exc}") from exc
        if not isinstance(env_kwargs, dict):
            raise SystemExit("--env-kwargs must be a JSON object.")
    else:
        env_kwargs = {}

    game = HanabiHumanAIGame(
        env_id=args.env_id,
        env_kwargs=env_kwargs,
        num_players=args.num_players,
        human_players=args.human_players,
        llm_agent=args.llm_agent,
        seed=args.seed,
        system_prompt=args.system_prompt,
        openai_api_key=args.openai_api_key,
        openai_base_url=args.openai_base_url,
        timeout_s=args.timeout,
        max_retries=args.max_retries,
        retry_initial_delay_s=args.retry_initial_delay,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        max_tokens=args.max_tokens,
        disable_thinking=bool(args.disable_thinking),
        max_auto_steps_per_tick=args.max_auto_steps_per_tick,
    )
    game.start_new_game(seed=args.seed)

    server = ThreadingHTTPServer((args.host, args.port), HanabiGUIHandler)
    server.app = game  # type: ignore[attr-defined]

    print(f"Hanabi GUI is running at http://{args.host}:{args.port}")
    print(
        "Human seats: "
        + ",".join(str(x) for x in sorted(game.human_players))
        + f" | Non-human seats: llm={args.llm_agent}"
    )
    print("Press Ctrl+C to stop.")

    try:
        server.serve_forever(poll_interval=0.5)
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
