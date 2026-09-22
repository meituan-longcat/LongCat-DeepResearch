#!/usr/bin/env python3
"""Markdown-first research harness: one question to a refined report.

Input:
    one research question

Planning output:
    candidate_a/
    ├── candidate_writers/{agent_a,agent_b,agent_c}.md
    ├── research_spec.judge.md
    ├── research_spec.critique.md
    └── research_spec.md

Research semantics stay in Markdown. JSON is used only where the HTTP tool-call
protocol requires it; no JSON planning artifact is written to disk.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
import hashlib
import json
import os
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextvars import ContextVar, copy_context
from pathlib import Path
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from .backend import current_backend
from . import execution_context as run_context
from . import telemetry
from .policy import current_harness_config


MODEL = os.getenv("DR_MODEL", os.getenv("DR_PLANNING_MODEL", "default"))
DEFAULT_RUN_ROOT = Path(
    os.getenv("DR_RUN_ROOT", "~/.cache/longcat-deepresearch/runs")
).expanduser()
WEB_FETCH_CACHE_TTL_SECONDS = int(os.getenv("DR_WEB_FETCH_CACHE_TTL_SECONDS", "86400"))


@dataclass
class _ToolLoopState:
    """Model-invisible hard limits for one Agent tool loop."""

    max_search_calls: int = 8
    max_fetch_calls: int = 5
    search_calls: int = 0
    fetch_calls: int = 0
    finalize_reason: str = ""

    @property
    def search_remaining(self) -> int:
        return max(0, self.max_search_calls - self.search_calls)

    @property
    def fetch_remaining(self) -> int:
        return max(0, self.max_fetch_calls - self.fetch_calls)

    @property
    def exhausted(self) -> bool:
        return self.search_remaining == 0 and self.fetch_remaining == 0


def _env_flag(name: str, default: str = "off") -> bool:
    return (os.getenv(name, default) or default).strip().lower() in {
        "1", "true", "yes", "on"
    }


TERMINAL_ENVIRONMENT_MESSAGE = (
    "Tool access is now closed by the environment. Do not request or describe "
    "another tool call. Use only the evidence already in the conversation."
)


def _markdown_protocol_mode() -> str:
    """Select one model-independent terminal protocol.

    ``legacy`` preserves the original tool-free Markdown request byte for byte.
    ``auto`` uses that same request for every model and enables deterministic
    recovery only after invalid output.
    """
    mode = os.getenv("DR_MARKDOWN_PROTOCOL_MODE", "auto").strip().lower()
    if mode not in {"auto", "legacy"}:
        raise ValueError("DR_MARKDOWN_PROTOCOL_MODE must be one of: auto, legacy")
    return mode


def _uses_auto_markdown_recovery() -> bool:
    return _markdown_protocol_mode() == "auto"


PLANNING_FILE_MARKER_RE = re.compile(
    r"<!--\s*FILE:\s*(research_spec(?:\.write)?\.md)\s*-->"
    r"(.*?)"
    r"<!--\s*END FILE\s*-->",
    re.IGNORECASE | re.DOTALL,
)
SECTION_RE = re.compile(r"^##\s+(S\d+)\s*[｜|:]", re.MULTILINE)
SUBSECTION_RE = re.compile(r"^###\s+(S\d+\.\d+)\s*[｜|:]", re.MULTILINE)
FACT_RE = re.compile(r"^##\s+(F\d+)\s*[｜|:]", re.MULTILINE)

_WEB_FETCH_CACHE_DIR: Path | None = None
_WEB_FETCH_CACHE_CONTEXT: ContextVar[Path | None] = ContextVar(
    "longcat_deepresearch_web_fetch_cache_dir", default=None
)
_WEB_FETCH_CACHE_GUARD = threading.Lock()
_WEB_FETCH_CACHE_LOCKS: dict[str, threading.Lock] = {}



TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web for authoritative sources and supporting evidence.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "top_k": {"type": "integer", "minimum": 1, "maximum": 10},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "web_fetch",
            "description": "Fetch the readable text of one or more web pages.",
            "parameters": {
                "type": "object",
                "properties": {
                    "urls": {
                        "type": "array",
                        "items": {"type": "string"},
                        "minItems": 1,
                        "maxItems": 10,
                    }
                },
                "required": ["urls"],
            },
        },
    },
]


def _chat(
    messages: list[dict[str, Any]],
    *,
    tools: bool | list[dict[str, Any]],
    max_tokens: int = 12000,
    tool_choice: str | dict[str, Any] | None = None,
    reject_truncation: bool = False,
) -> dict[str, Any]:
    """Call the user-provided backend with an OpenAI function-calling request."""

    selected_tools = TOOLS if tools is True else (tools or [])
    request: dict[str, Any] = {
        "model": MODEL,
        "messages": copy.deepcopy(messages),
        "stream": False,
        "max_tokens": max_tokens,
    }
    if selected_tools:
        request["tools"] = copy.deepcopy(selected_tools)
        request["tool_choice"] = tool_choice or "auto"

    started = time.monotonic()
    try:
        payload = current_backend().llm(request)
    except Exception:
        telemetry.record_llm(
            response={},
            max_tokens=max_tokens,
            tools_enabled=bool(selected_tools),
            attempts=1,
            elapsed_s=time.monotonic() - started,
            status="failed",
        )
        raise
    elapsed = time.monotonic() - started
    if not isinstance(payload, dict):
        raise TypeError("backend.llm() must return a JSON object")

    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        raise ValueError("backend.llm() response must contain choices[0]")
    message = choices[0].get("message")
    if not isinstance(message, dict):
        raise ValueError("backend.llm() response choices[0].message must be an object")
    if message.get("role") not in {None, "assistant"}:
        raise ValueError("backend.llm() message role must be assistant")
    if not isinstance(message.get("content"), (str, type(None))):
        raise ValueError("backend.llm() message content must be a string or null")
    if message.get("tool_calls") is not None and not isinstance(
        message.get("tool_calls"), list
    ):
        raise ValueError("backend.llm() message tool_calls must be a list")
    for call in message.get("tool_calls") or []:
        if not isinstance(call, dict) or not isinstance(call.get("id"), str):
            raise ValueError("each LLM tool call must be an object with a string id")
        if call.get("type", "function") != "function":
            raise ValueError("only function tool calls are supported")
        function = call.get("function")
        if not isinstance(function, dict) or not isinstance(function.get("name"), str):
            raise ValueError("each LLM tool call must contain a function name")
        if not isinstance(function.get("arguments"), str):
            raise ValueError("each LLM tool call must contain JSON arguments as a string")

    truncation_error = _truncation_error(payload, max_tokens) if reject_truncation else None
    telemetry.record_llm(
        response=payload,
        max_tokens=max_tokens,
        tools_enabled=bool(selected_tools),
        attempts=1,
        elapsed_s=elapsed,
        status="failed" if truncation_error else "success",
    )
    if truncation_error:
        raise RuntimeError(truncation_error)
    clean = {
        "role": "assistant",
        "content": message.get("content"),
    }
    if message.get("tool_calls"):
        clean["tool_calls"] = message["tool_calls"]
    return clean


def _search_web(query: str, top_k: int = 8) -> str:
    query = str(query or "").strip()
    if not query:
        return "Search rejected: empty query."
    top_k = max(1, min(int(top_k or 8), 10))
    payload = current_backend().web_search({"query": query, "top_k": top_k})
    if not isinstance(payload, dict) or not isinstance(payload.get("results"), list):
        raise ValueError("backend.web_search() must return an object with a results list")

    lines = [f"# Search results: {query}"]
    for index, item in enumerate(payload["results"][:top_k], start=1):
        if not isinstance(item, dict):
            raise ValueError("each web_search result must be an object")
        title = item.get("title")
        url = item.get("url")
        snippet = item.get("snippet", "")
        published_at = item.get("published_at")
        if not isinstance(title, str) or not isinstance(url, str):
            raise ValueError("web_search result title and url must be strings")
        if not isinstance(snippet, str):
            raise ValueError("web_search result snippet must be a string")
        if published_at is not None and not isinstance(published_at, str):
            raise ValueError("web_search result published_at must be a string or null")
        parsed = urlsplit(url)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            raise ValueError("web_search result url must be an HTTP(S) URL")
        lines.append(f"\n## Result {index}: {title.strip() or 'Untitled'}\nURL: {url}")
        if published_at:
            lines.append(f"Date: {published_at}")
        if snippet.strip():
            normalized_snippet = re.sub(r"\s+", " ", snippet).strip()
            lines.append(f"Snippet: {normalized_snippet[:1200]}")
    return "\n".join(lines)[:14000]


_FETCH_IMAGE_RE = re.compile(r"!\[([^\]]*)\]\([^)]*\s*[\"']?[\"']?\s*\)")


def _web_fetch_cache_key(urls: list[str]) -> str:
    material = json.dumps(urls, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def _validate_fetch_urls(urls: Any) -> list[str]:
    if not isinstance(urls, list) or not urls:
        raise ValueError("urls must be a non-empty list")
    if len(urls) > 10:
        raise ValueError("urls must contain at most 10 items")
    normalized: list[str] = []
    for value in urls:
        if not isinstance(value, str):
            raise ValueError("each URL must be a string")
        url = value.strip()
        if not url or any(
            character.isspace() or ord(character) < 32 or ord(character) == 127
            for character in url
        ):
            raise ValueError("each URL must be a valid HTTP(S) URL")
        try:
            parsed = urlsplit(url)
            parsed.port
        except ValueError as exc:
            raise ValueError("each URL must be a valid HTTP(S) URL") from exc
        if parsed.scheme.lower() not in {"http", "https"} or not parsed.hostname:
            raise ValueError("each URL must be a valid HTTP(S) URL")
        normalized.append(url)
    return normalized


def _validate_fetch_response(
    payload: Any, requested_urls: list[str]
) -> list[dict[str, str | None]]:
    if not isinstance(payload, dict) or not isinstance(payload.get("pages"), list):
        raise ValueError("backend.web_fetch() must return an object with a pages list")
    pages = payload["pages"]
    if len(pages) != len(requested_urls):
        raise ValueError("web_fetch pages must match the requested URL count")

    cleaned: list[dict[str, str | None]] = []
    for expected_url, page in zip(requested_urls, pages):
        if not isinstance(page, dict):
            raise ValueError("each web_fetch page must be an object")
        url = page.get("url")
        title = page.get("title")
        text = page.get("text")
        error = page.get("error")
        if url != expected_url:
            raise ValueError("web_fetch pages must preserve requested URL order")
        if not isinstance(title, str) or not isinstance(text, str):
            raise ValueError("web_fetch page title and text must be strings")
        if error is not None and not isinstance(error, str):
            raise ValueError("web_fetch page error must be a string or null")
        normalized_text = _FETCH_IMAGE_RE.sub("![](IMG_URL)", text)
        if len(normalized_text) > 8000:
            normalized_text = normalized_text[:8000] + "...[CONTENT TRUNCATED]"
        cleaned.append(
            {
                "url": url,
                "title": title,
                "text": normalized_text,
                "error": error,
            }
        )
    return cleaned


def _web_fetch(urls: Any) -> str:
    try:
        requested_urls = _validate_fetch_urls(urls)
    except ValueError as exc:
        return f"Web fetch rejected: {exc}."

    key = _web_fetch_cache_key(requested_urls)
    cache_dir = _WEB_FETCH_CACHE_CONTEXT.get() or _WEB_FETCH_CACHE_DIR
    cache_path = cache_dir / f"{key}.json" if cache_dir else None
    with _WEB_FETCH_CACHE_GUARD:
        batch_lock = _WEB_FETCH_CACHE_LOCKS.setdefault(key, threading.Lock())
    with batch_lock:
        if (
            cache_path
            and cache_path.is_file()
            and WEB_FETCH_CACHE_TTL_SECONDS > 0
            and time.time() - cache_path.stat().st_mtime <= WEB_FETCH_CACHE_TTL_SECONDS
        ):
            try:
                cached = json.loads(cache_path.read_text(encoding="utf-8"))
                pages = _validate_fetch_response(
                    {"pages": cached}, requested_urls
                )
                return json.dumps({"pages": pages}, ensure_ascii=False)
            except (OSError, ValueError, TypeError, json.JSONDecodeError):
                pass

        pages = _validate_fetch_response(
            current_backend().web_fetch({"urls": requested_urls}),
            requested_urls,
        )
        rendered = json.dumps({"pages": pages}, ensure_ascii=False)
        if cache_path:
            try:
                _atomic_write(
                    cache_path,
                    json.dumps(pages, ensure_ascii=False),
                )
            except (OSError, UnicodeError):
                pass
        return rendered


def _assistant_message(message: dict[str, Any]) -> dict[str, Any]:
    clean: dict[str, Any] = {
        "role": "assistant",
        "content": message.get("content"),
    }
    if message.get("tool_calls"):
        clean["tool_calls"] = message["tool_calls"]
    return clean


def _run_tool_call(call: dict[str, Any]) -> tuple[str, str]:
    call_id = str(call.get("id") or "missing-tool-id")
    function = call.get("function") or {}
    name = str(function.get("name") or "")
    try:
        arguments = json.loads(function.get("arguments") or "{}")
        if name == "web_search":
            result = _search_web(
                arguments.get("query", ""),
                arguments.get("top_k", 8),
            )
        elif name == "web_fetch":
            result = _web_fetch(arguments.get("urls", []))
        else:
            result = f"Unknown tool: {name}"
    except Exception as exc:
        result = f"Tool {name} failed: {type(exc).__name__}: {exc}"
    return call_id, result


def _run_scoped_tool_call(
    call: dict[str, Any],
    *,
    call_type: str,
    actor: str,
    trajectory_id: str,
    turn_index: int,
    execution_fields: dict[str, Any] | None = None,
) -> tuple[str, str]:
    call_id = str(call.get("id") or "missing-tool-id")
    fields = dict(execution_fields or {})
    fields.update(
        {
            "call_type": call_type,
            "actor": actor,
            "trajectory_id": trajectory_id,
            "turn_index": turn_index,
            "tool_call_id": call_id,
        }
    )
    with run_context.scope(**fields):
        return _run_tool_call(call)


def _extract_terminal_text(message: dict[str, Any]) -> str:
    """Return only provider-visible final content, never private reasoning."""
    content = message.get("content")
    if isinstance(content, str) and content.strip():
        return content.strip()
    return ""


def _truncation_error(payload: dict[str, Any], max_tokens: int) -> str | None:
    choices = payload.get("choices") or []
    choice = choices[0] if choices and isinstance(choices[0], dict) else {}
    if choice.get("finish_reason") == "length":
        return "LLM response was truncated with finish_reason=length"
    usage = payload.get("usage") or {}
    for field_name in ("output_tokens", "completion_tokens", "output_token_num"):
        value = usage.get(field_name)
        if (
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and value >= max_tokens
        ):
            return (
                "LLM response reached the requested output token limit: "
                f"{value} >= {max_tokens}"
            )
    return None


def _append_tool_free_retry(
    messages: list[dict[str, Any]],
    message: dict[str, Any],
    instruction: str,
) -> None:
    """Close provider-emitted tool calls before a tool-free retry."""
    messages.append(_assistant_message(message))
    for call in message.get("tool_calls") or []:
        messages.append(
            {
                "role": "tool",
                "tool_call_id": str(call.get("id") or "missing-tool-id"),
                "content": "Tools are closed. Use the evidence already in context and finalize.",
            }
        )
    messages.append({"role": "user", "content": instruction})


def _stable_tool_prefix_terminal_enabled() -> bool:
    """Keep raw-Serving terminal retries on the prior tool-prefixed prompt."""
    default = (
        "on"
        if current_harness_config().terminal_tool_schema == "preserve"
        else "off"
    )
    return _env_flag("DR_STABLE_TOOL_PREFIX_TERMINAL", default)


def _chat_forced_terminal(
    messages: list[dict[str, Any]], *, max_tokens: int
) -> dict[str, Any]:
    """Request terminal Markdown without invalidating the raw prompt prefix.

    Raw LongCat Serving renders tool declarations before the system message.  A
    tools=True -> tools=False transition therefore changes the prompt from its
    first token and defeats prefix-cache reuse.  Other providers retain the
    established tool-free request unless the raw adapter explicitly enables
    this behavior.
    """
    if _stable_tool_prefix_terminal_enabled():
        return _chat(
            messages,
            tools=True,
            max_tokens=max_tokens,
            tool_choice="none",
        )
    return _chat(messages, tools=False, max_tokens=max_tokens)


def _controller_terminal(
    messages: list[dict[str, Any]],
    *,
    final_instruction: str,
    call_type: str,
    actor: str,
    trajectory_id: str,
    turn_index: int,
    reason: str,
    chat_fn: Any | None = None,
    max_tokens: int = 32000,
    reject_truncation: bool = False,
) -> tuple[str, dict[str, Any], int]:
    """Use one shared, KV-stable terminal protocol for every tool agent."""
    messages.append(
        {
            "role": "user",
            "content": TERMINAL_ENVIRONMENT_MESSAGE + "\n\n" + final_instruction,
        }
    )
    for attempt in range(2):
        with run_context.scope(
            call_type=call_type,
            actor=actor,
            trajectory_id=trajectory_id,
            turn_index=turn_index + attempt,
            controller_terminal=True,
            forced_terminal=True,
            terminal_retry=bool(attempt),
            terminal_reason=reason,
            degraded=True,
        ):
            request_kwargs: dict[str, Any] = {
                "tools": True,
                "max_tokens": max_tokens,
                "tool_choice": "none",
            }
            if reject_truncation:
                request_kwargs["reject_truncation"] = True
            message = (chat_fn or _chat)(messages, **request_kwargs)
        terminal = _extract_terminal_text(message)
        if terminal and not message.get("tool_calls"):
            return terminal, message, attempt
    raise RuntimeError(f"{actor} returned an empty controller terminal answer")


class ResearchAgent:
    """One agent implementation with one isolated message history per instance.

    Writer, Judge, Critic, and Reviser differ only in their role prompt, input,
    and tool policy. They deliberately do not share messages with one another.
    """

    def __init__(
        self,
        *,
        system_prompt: str,
        label: str,
        call_type: str,
        actor: str,
        trajectory_id: str,
    ) -> None:
        self.label = label
        self.call_type = call_type
        self.actor = actor
        self.trajectory_id = trajectory_id
        self.messages: list[dict[str, Any]] = [
            {"role": "system", "content": system_prompt}
        ]

    def complete(
        self,
        user_prompt: str,
        *,
        max_tokens: int = 12000,
        empty_error: str | None = None,
        turn_index: int = 1,
    ) -> str:
        """Run a tool-free turn through the same isolated agent session."""
        self.messages.append({"role": "user", "content": user_prompt})
        budgets = [max_tokens, 64000] if _uses_auto_markdown_recovery() else [max_tokens]
        for attempt, budget in enumerate(budgets):
            with run_context.scope(
                call_type=self.call_type,
                actor=self.actor,
                trajectory_id=self.trajectory_id,
                turn_index=turn_index + attempt,
                terminal_retry=bool(attempt),
            ):
                message = _chat(self.messages, tools=False, max_tokens=budget)
            terminal = _extract_terminal_text(message)
            if terminal and not message.get("tool_calls"):
                return terminal
            if attempt + 1 < len(budgets):
                _append_tool_free_retry(
                    self.messages,
                    message,
                    "Tools are unavailable. Return the complete required Markdown artifact now; "
                    "do not explain your reasoning or request another tool.",
                )
        raise RuntimeError(empty_error or f"{self.label} returned empty content")

    def research(
        self,
        user_prompt: str,
        *,
        max_tool_rounds: int,
        min_search_calls: int = 3,
        min_fetch_calls: int = 0,
        final_instruction: str,
        max_tokens: int = 12000,
        terminal_max_tokens: int = 32000,
        reject_truncation: bool = False,
    ) -> tuple[str, list[str]]:
        """Run the shared web_search/web_fetch/message-memory loop."""
        self.messages.append({"role": "user", "content": user_prompt})
        trace: list[str] = [f"# {self.label} Trace", "", f"Model: `{MODEL}`", ""]
        search_count = 0
        fetch_count = 0
        max_search_calls = 8
        max_fetch_calls = 5
        state = _ToolLoopState(
            max_search_calls=max_search_calls,
            max_fetch_calls=max_fetch_calls,
        )

        def request_terminal(turn_index: int) -> tuple[str, list[str]]:
            reason = state.finalize_reason or "round_limit"
            terminal, _, attempt = _controller_terminal(
                self.messages,
                final_instruction=final_instruction,
                call_type=self.call_type,
                actor=self.actor,
                trajectory_id=self.trajectory_id,
                turn_index=turn_index,
                reason=reason,
                max_tokens=terminal_max_tokens,
                reject_truncation=reject_truncation,
            )
            trace.append(
                f"Degraded forced terminal answer after {search_count} "
                f"search calls and {fetch_count} fetch calls ({reason})"
                f"{' with one identical protocol retry' if attempt else ''}."
            )
            return terminal, trace

        for round_index in range(1, max_tool_rounds + 1):
            if state.finalize_reason:
                return request_terminal(round_index)
            with run_context.scope(
                call_type=self.call_type,
                actor=self.actor,
                trajectory_id=self.trajectory_id,
                turn_index=round_index,
            ):
                # Quality baseline (d837e92): preserve this request exactly.
                request_kwargs: dict[str, Any] = {}
                if max_tokens != 12000:
                    request_kwargs["max_tokens"] = max_tokens
                if reject_truncation:
                    request_kwargs["reject_truncation"] = True
                message = _chat(self.messages, tools=True, **request_kwargs)
            calls = message.get("tool_calls") or []
            if calls:
                print(
                    f"[{self.label}] tool round {round_index}: {len(calls)} call(s)",
                    file=sys.stderr,
                    flush=True,
                )
                trace.append(f"## Tool round {round_index}")
                results: dict[str, str] = {}
                allowed_calls: list[dict[str, Any]] = []
                rejected_calls: list[dict[str, Any]] = []
                proposed_search_calls = 0
                proposed_fetch_calls = 0
                for call in calls:
                    call_id = str(call.get("id") or "missing-tool-id")
                    name = str((call.get("function") or {}).get("name") or "unknown")
                    if name == "web_search":
                        if search_count + proposed_search_calls >= max_search_calls:
                            rejected_calls.append(call)
                            continue
                        proposed_search_calls += 1
                    elif name == "web_fetch":
                        if fetch_count + proposed_fetch_calls >= max_fetch_calls:
                            rejected_calls.append(call)
                            continue
                        proposed_fetch_calls += 1
                    allowed_calls.append(call)
                if rejected_calls:
                    with telemetry.scope(
                        call_type=self.call_type,
                        actor=self.actor,
                        trajectory_id=self.trajectory_id,
                        turn_index=round_index,
                    ):
                        telemetry.record_tool_batch(calls, allowed_call_ids=set())
                    trace.extend(
                        [
                            f"- Controller atomically rejected a {len(calls)}-call action "
                            f"containing {len(rejected_calls)} over-budget request(s); no "
                            "tool was executed and no synthetic observation was added.",
                            "",
                        ]
                    )
                    state.finalize_reason = "tool_action_exceeds_remaining_budget"
                    continue
                search_count += proposed_search_calls
                fetch_count += proposed_fetch_calls
                state.search_calls += proposed_search_calls
                state.fetch_calls += proposed_fetch_calls
                assistant = _assistant_message(message)
                self.messages.append(assistant)
                with ThreadPoolExecutor(
                    max_workers=max(1, min(4, len(allowed_calls)))
                ) as executor:
                    execution_fields = run_context.context_fields()
                    futures = {}
                    for call in allowed_calls:
                        context = copy_context()
                        future = executor.submit(
                            context.run,
                            _run_scoped_tool_call,
                            call,
                            call_type=self.call_type,
                            actor=self.actor,
                            trajectory_id=self.trajectory_id,
                            turn_index=round_index,
                            execution_fields=execution_fields,
                        )
                        futures[future] = call
                    for future in as_completed(futures):
                        call_id, result = future.result()
                        results[call_id] = result
                with telemetry.scope(
                    call_type=self.call_type,
                    actor=self.actor,
                    trajectory_id=self.trajectory_id,
                    turn_index=round_index,
                ):
                    telemetry.record_tool_batch(
                        calls,
                        allowed_call_ids={
                            str(call.get("id") or "missing-tool-id")
                            for call in allowed_calls
                        },
                    )
                for call in allowed_calls:
                    call_id = str(call.get("id") or "missing-tool-id")
                    name = str((call.get("function") or {}).get("name") or "unknown")
                    trace.append(f"- `{name}`: `{call_id}`")
                    self.messages.append(
                        {"role": "tool", "tool_call_id": call_id, "content": results[call_id]}
                    )
                trace.append("")
                if state.exhausted:
                    state.finalize_reason = "tool_budget_exhausted"
                continue

            terminal = _extract_terminal_text(message)
            if search_count < min_search_calls or fetch_count < min_fetch_calls:
                self.messages.append(_assistant_message(message))
                self.messages.append(
                    {
                        "role": "user",
                        "content": (
                            f"Evidence collection is incomplete ({search_count} web_search, "
                            f"{fetch_count} web_fetch). Continue and use at least "
                            f"{min_search_calls} complementary web searches and {min_fetch_calls} "
                            "authoritative page fetches before finalizing."
                        ),
                    }
                )
                continue
            if terminal:
                trace.append(f"Terminal answer produced after {search_count} search calls.")
                print(
                    f"[{self.label}] complete after {search_count} search calls",
                    file=sys.stderr,
                    flush=True,
                )
                return terminal, trace
            state.finalize_reason = "empty_model_turn"
            continue

        state.finalize_reason = state.finalize_reason or "round_limit"
        return request_terminal(max_tool_rounds + 1)



def _canonical_url(url: str) -> str:
    parts = urlsplit(url.strip())
    query = [
        (key, value)
        for key, value in parse_qsl(parts.query, keep_blank_values=True)
        if not key.lower().startswith("utm_") and key.lower() not in {"ref", "source"}
    ]
    path = parts.path.rstrip("/") or "/"
    return urlunsplit((parts.scheme.lower(), parts.netloc.lower(), path, urlencode(query), ""))


def _activate_web_fetch_cache(run_dir: Path) -> Path:
    global _WEB_FETCH_CACHE_DIR
    cache_dir = run_dir / "web_fetch_cache"
    _WEB_FETCH_CACHE_DIR = cache_dir
    cache_dir.mkdir(parents=True, exist_ok=True)
    _WEB_FETCH_CACHE_CONTEXT.set(cache_dir)
    return cache_dir



def _atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)
