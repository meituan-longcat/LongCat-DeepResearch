#!/usr/bin/env python3
"""Make one small request to each configured backend capability."""

from __future__ import annotations

import os
import time
from collections.abc import Callable
from typing import Any
from urllib.parse import urlsplit

from longcat_deepresearch.backend import load_backend


def _timed(name: str, call: Callable[[], dict[str, Any]]) -> dict[str, Any]:
    started = time.monotonic()
    value = call()
    if not isinstance(value, dict):
        raise TypeError(f"{name} must return a dictionary")
    print(f"{name}: ok ({time.monotonic() - started:.2f}s)")
    return value


def main() -> int:
    model = os.getenv("DR_MODEL", "").strip()
    if not model:
        raise RuntimeError("DR_MODEL is required")

    backend = load_backend()
    llm_response = _timed(
        "llm",
        lambda: backend.llm(
            {
                "model": model,
                "messages": [
                    {
                        "role": "user",
                        "content": "Reply with exactly: backend smoke passed",
                    }
                ],
                "stream": False,
                "max_tokens": 256,
            }
        ),
    )
    choices = llm_response.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        raise ValueError("llm response must contain choices[0]")
    message = choices[0].get("message")
    if not isinstance(message, dict) or not str(message.get("content") or "").strip():
        raise ValueError("llm response must contain non-empty assistant content")

    query = os.getenv("DR_SMOKE_QUERY", "official Python documentation")
    search_response = _timed(
        "web_search",
        lambda: backend.web_search({"query": query, "top_k": 1}),
    )
    results = search_response.get("results")
    if not isinstance(results, list) or not results or not isinstance(results[0], dict):
        raise ValueError("web_search response must contain at least one result")
    url = results[0].get("url")
    parsed = urlsplit(url) if isinstance(url, str) else None
    if parsed is None or parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError("web_search result must contain an HTTP(S) URL")

    fetch_response = _timed(
        "web_fetch",
        lambda: backend.web_fetch({"urls": [url]}),
    )
    pages = fetch_response.get("pages")
    if not isinstance(pages, list) or len(pages) != 1 or not isinstance(pages[0], dict):
        raise ValueError("web_fetch response must contain one page")
    page = pages[0]
    if page.get("url") != url:
        raise ValueError("web_fetch must preserve the requested URL")
    if page.get("error") or not str(page.get("text") or "").strip():
        raise ValueError("web_fetch returned no readable page content")

    print("backend smoke passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
