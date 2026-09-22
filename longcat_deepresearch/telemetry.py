"""Low-overhead, content-free usage telemetry for harness runs.

This module is deliberately observational: callers pass completed request and
response metadata after the model/tool decision has already been made.  It
never mutates prompts, messages, tool policy, budgets, or model responses.
"""

from __future__ import annotations

import json
import os
import threading
import time
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
from typing import Any, Iterator


_CONTEXT: ContextVar[dict[str, Any]] = ContextVar(
    "longcat_deepresearch_usage_context", default={}
)
_LOCK = threading.Lock()


def enabled() -> bool:
    return (os.getenv("DR_USAGE_TELEMETRY", "off") or "off").lower() not in {
        "off",
        "0",
        "false",
    }


@contextmanager
def scope(**fields: Any) -> Iterator[None]:
    current = dict(_CONTEXT.get())
    current.update({key: value for key, value in fields.items() if value is not None})
    token = _CONTEXT.set(current)
    try:
        yield
    finally:
        _CONTEXT.reset(token)


def _stage(call_type: str) -> str:
    if call_type.startswith("planning_"):
        return "planning"
    if call_type.startswith("subsection_researcher"):
        return "research"
    if call_type in {"global_editorial_plan", "local_editor"}:
        return "editing"
    return "unknown"


def _normalized_usage(response: Any) -> dict[str, int]:
    usage = response.get("usage") if isinstance(response, dict) else None
    usage = usage if isinstance(usage, dict) else {}
    details = usage.get("completion_tokens_details")
    details = details if isinstance(details, dict) else {}

    def integer(*keys: str) -> int:
        for key in keys:
            value = usage.get(key)
            if isinstance(value, (int, float)):
                return int(value)
        return 0

    prompt = integer("prompt_tokens", "input_tokens")
    completion = integer("completion_tokens", "output_tokens")
    total = integer("total_tokens") or prompt + completion
    reasoning = details.get("reasoning_tokens", usage.get("reasoning_tokens", 0))
    return {
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "reasoning_tokens": int(reasoning) if isinstance(reasoning, (int, float)) else 0,
        "total_tokens": total,
    }


def _append(event: dict[str, Any]) -> None:
    if not enabled():
        return
    context = dict(_CONTEXT.get())
    run_dir = context.get("run_dir")
    if not run_dir:
        return
    call_type = str(context.get("call_type") or "unknown")
    value = {
        "ts": time.time(),
        "stage": str(context.get("stage") or _stage(call_type)),
        "call_type": call_type,
        "actor": str(context.get("actor") or "unknown"),
        "trajectory_id": str(context.get("trajectory_id") or ""),
        "turn_index": context.get("turn_index"),
        "terminal_retry": bool(context.get("terminal_retry")),
        **event,
    }
    path = Path(run_dir) / "telemetry" / "events.jsonl"
    try:
        with _LOCK:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(value, ensure_ascii=False) + "\n")
    except Exception:
        # Telemetry is strictly best-effort and must never change pipeline flow.
        return


def record_llm(
    *,
    response: Any,
    max_tokens: int,
    tools_enabled: bool,
    attempts: int,
    elapsed_s: float,
    status: str,
    forced_terminal: bool | None = None,
) -> None:
    tool_names: list[str] = []
    if isinstance(response, dict):
        choices = response.get("choices") or []
        message = choices[0].get("message") if choices and isinstance(choices[0], dict) else {}
        for call in (message or {}).get("tool_calls") or []:
            if isinstance(call, dict):
                tool_names.append(str((call.get("function") or {}).get("name") or "unknown"))
    _append(
        {
            "event": "llm_call",
            "status": status,
            "tools_enabled": bool(tools_enabled),
            "forced_terminal": bool(
                _CONTEXT.get().get("forced_terminal")
                if forced_terminal is None
                else forced_terminal
            ),
            "max_output_tokens": int(max_tokens),
            "attempts": int(attempts),
            "elapsed_s": round(float(elapsed_s), 3),
            "usage": _normalized_usage(response),
            "requested_tool_calls": tool_names,
        }
    )


def record_llm_http_attempt(
    *,
    attempt: int,
    status: str,
    http_status: int | None,
    elapsed_s: float,
    affinity_session_id: str,
    biz_query_id: str,
    affinity_rotated: bool,
    error_type: str = "",
) -> None:
    """Record one content-free HTTP attempt for retry and affinity diagnosis."""
    _append(
        {
            "event": "llm_http_attempt",
            "attempt": int(attempt),
            "status": status,
            "http_status": http_status,
            "elapsed_s": round(float(elapsed_s), 3),
            "affinity_session_id": affinity_session_id,
            "biz_query_id": biz_query_id,
            "affinity_rotated": bool(affinity_rotated),
            "error_type": error_type,
        }
    )


def record_tool_batch(
    calls: list[dict[str, Any]], *, allowed_call_ids: set[str]
) -> None:
    for call in calls:
        if not isinstance(call, dict):
            continue
        call_id = str(call.get("id") or "missing-tool-id")
        name = str((call.get("function") or {}).get("name") or "unknown")
        _append(
            {
                "event": "tool_call",
                "tool": name,
                "call_id": call_id,
                "executed": call_id in allowed_call_ids,
            }
        )


def write_summary(run_dir: str | Path) -> dict[str, Any]:
    """Aggregate the append-only ledger without requiring model payloads."""
    run_dir = Path(run_dir)
    events_path = run_dir / "telemetry" / "events.jsonl"
    events: list[dict[str, Any]] = []
    if events_path.exists():
        for line in events_path.read_text(encoding="utf-8").splitlines():
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    groups: dict[tuple[str, str, str], dict[str, Any]] = {}
    for event in events:
        key = (event.get("stage", "unknown"), event.get("call_type", "unknown"), event.get("actor", "unknown"))
        group = groups.setdefault(
            key,
            {
                "stage": key[0],
                "call_type": key[1],
                "actor": key[2],
                "llm_calls": 0,
                "max_output_tokens": 0,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "reasoning_tokens": 0,
                "total_tokens": 0,
                "search_calls": 0,
                "fetch_calls": 0,
                "rejected_tool_calls": 0,
                "forced_terminal_calls": 0,
                "terminal_retry_calls": 0,
            },
        )
        if event.get("event") == "llm_call":
            group["llm_calls"] += 1
            group["max_output_tokens"] += int(event.get("max_output_tokens") or 0)
            usage = event.get("usage") or {}
            for name in ("prompt_tokens", "completion_tokens", "reasoning_tokens", "total_tokens"):
                group[name] += int(usage.get(name) or 0)
            group["forced_terminal_calls"] += int(bool(event.get("forced_terminal")))
            group["terminal_retry_calls"] += int(bool(event.get("terminal_retry")))
        elif event.get("event") == "tool_call":
            if not event.get("executed"):
                group["rejected_tool_calls"] += 1
            elif event.get("tool") == "web_search":
                group["search_calls"] += 1
            elif event.get("tool") == "web_fetch":
                group["fetch_calls"] += 1

    rows = sorted(groups.values(), key=lambda row: (row["stage"], row["call_type"], row["actor"]))
    metric_names = (
        "llm_calls",
        "max_output_tokens",
        "prompt_tokens",
        "completion_tokens",
        "reasoning_tokens",
        "total_tokens",
        "search_calls",
        "fetch_calls",
        "rejected_tool_calls",
        "forced_terminal_calls",
        "terminal_retry_calls",
    )
    totals = {name: sum(int(row[name]) for row in rows) for name in metric_names}
    stages: dict[str, dict[str, Any]] = {}
    for row in rows:
        stage = stages.setdefault(
            row["stage"], {"stage": row["stage"], **{name: 0 for name in metric_names}}
        )
        for name in metric_names:
            stage[name] += int(row[name])
    summary = {
        "schema_version": 1,
        "totals": totals,
        "by_stage": sorted(stages.values(), key=lambda row: row["stage"]),
        "by_actor": rows,
    }
    out_dir = run_dir / "telemetry"
    out_dir.mkdir(parents=True, exist_ok=True)
    temporary = out_dir / "usage_summary.json.tmp"
    temporary.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(out_dir / "usage_summary.json")

    columns = ["Stage", "Call type", "Actor", "LLM", "Search", "Fetch", "Input tok", "Output tok", "Reasoning tok", "Total tok", "Output budget", "Forced final", "64K retry"]
    stage_columns = ["Stage", "LLM", "Search", "Fetch", "Input tok", "Output tok", "Reasoning tok", "Total tok", "Output budget", "Forced final", "64K retry"]
    lines = [
        "# LongCat-DeepResearch Usage Summary", "", "## By Stage", "",
        "| " + " | ".join(stage_columns) + " |",
        "|" + "|".join(["---"] + ["---:"] * 10) + "|",
    ]
    for row in summary["by_stage"]:
        lines.append(
            "| " + " | ".join(
                str(row[name])
                for name in (
                    "stage", "llm_calls", "search_calls", "fetch_calls", "prompt_tokens",
                    "completion_tokens", "reasoning_tokens", "total_tokens",
                    "max_output_tokens", "forced_terminal_calls", "terminal_retry_calls",
                )
            ) + " |"
        )
    lines.extend([
        "", "## By Agent", "",
        "| " + " | ".join(columns) + " |",
        "|" + "|".join(["---"] * 3 + ["---:"] * 10) + "|",
    ])
    for row in rows:
        lines.append(
            "| " + " | ".join(
                str(row[name])
                for name in (
                    "stage", "call_type", "actor", "llm_calls", "search_calls", "fetch_calls",
                    "prompt_tokens", "completion_tokens", "reasoning_tokens", "total_tokens",
                    "max_output_tokens", "forced_terminal_calls", "terminal_retry_calls",
                )
            ) + " |"
        )
    lines.extend(["", "## Totals", "", "```json", json.dumps(totals, ensure_ascii=False, indent=2), "```", ""])
    (out_dir / "usage_summary.md").write_text("\n".join(lines), encoding="utf-8")
    return summary
