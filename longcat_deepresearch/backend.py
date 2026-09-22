"""Dependency-injected interfaces for model and web access.

The harness owns orchestration and validation. A backend implementation owns
networking, authentication, retries, rate limits, and provider-specific fields.
"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
import importlib
import os
import threading
from typing import Any, Iterator, Protocol, runtime_checkable


JsonObject = dict[str, Any]


@runtime_checkable
class HarnessBackend(Protocol):
    """External capabilities required by the harness.

    llm accepts and returns OpenAI Chat Completions-compatible objects.
    web_search and web_fetch use the normalized JSON contracts documented
    in docs/BACKEND_INTERFACE.md.
    """

    def llm(self, request: JsonObject) -> JsonObject:
        """Return a Chat Completions-compatible response."""

    def web_search(self, request: JsonObject) -> JsonObject:
        """Return normalized search results."""

    def web_fetch(self, request: JsonObject) -> JsonObject:
        """Return normalized page content in request order."""


_CURRENT_BACKEND: ContextVar[HarnessBackend | None] = ContextVar(
    "longcat_deepresearch_backend", default=None
)
_DEFAULT_BACKEND: HarnessBackend | None = None
_DEFAULT_BACKEND_LOCK = threading.Lock()


def _validate_backend(value: object, source: str) -> HarnessBackend:
    missing = [
        method
        for method in ("llm", "web_search", "web_fetch")
        if not callable(getattr(value, method, None))
    ]
    if missing:
        raise TypeError(
            f"{source} does not implement required methods: {', '.join(missing)}"
        )
    return value  # type: ignore[return-value]


def load_backend(module_name: str | None = None) -> HarnessBackend:
    """Load create_backend() or BACKEND from a user-owned module."""

    selected = (module_name or os.getenv("DR_BACKEND_MODULE") or "local_backend").strip()
    if not selected:
        raise RuntimeError("DR_BACKEND_MODULE must not be empty")
    try:
        module = importlib.import_module(selected)
    except ModuleNotFoundError as exc:
        if exc.name != selected:
            raise
        raise RuntimeError(
            f"backend module {selected!r} was not found; implement HarnessBackend "
            "and set DR_BACKEND_MODULE"
        ) from exc

    factory = getattr(module, "create_backend", None)
    if callable(factory):
        return _validate_backend(factory(), f"{selected}.create_backend()")
    if hasattr(module, "BACKEND"):
        return _validate_backend(module.BACKEND, f"{selected}.BACKEND")
    raise RuntimeError(
        f"backend module {selected!r} must export create_backend() or BACKEND"
    )


def current_backend() -> HarnessBackend:
    selected = _CURRENT_BACKEND.get()
    if selected is not None:
        return selected

    global _DEFAULT_BACKEND
    if _DEFAULT_BACKEND is None:
        with _DEFAULT_BACKEND_LOCK:
            if _DEFAULT_BACKEND is None:
                _DEFAULT_BACKEND = load_backend()
    return _DEFAULT_BACKEND


@contextmanager
def backend_scope(backend: HarnessBackend | None) -> Iterator[None]:
    """Bind one backend to a run and propagate it through copied contexts."""

    if backend is None:
        yield
        return
    selected = _validate_backend(backend, "backend")
    token = _CURRENT_BACKEND.set(selected)
    try:
        yield
    finally:
        _CURRENT_BACKEND.reset(token)


def reset_default_backend() -> None:
    """Clear the lazy module backend; intended for tests and host reloads."""

    global _DEFAULT_BACKEND
    _DEFAULT_BACKEND = None
