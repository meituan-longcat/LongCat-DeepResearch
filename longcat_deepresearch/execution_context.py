"""Request correlation context without prompt or response capture."""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Iterator

from . import telemetry


_CONTEXT: ContextVar[dict[str, Any]] = ContextVar(
    "longcat_deepresearch_execution_context", default={}
)


def context_fields() -> dict[str, Any]:
    return dict(_CONTEXT.get())


@contextmanager
def scope(**fields: Any) -> Iterator[None]:
    current = dict(_CONTEXT.get())
    current.update({key: value for key, value in fields.items() if value is not None})
    token = _CONTEXT.set(current)
    try:
        with telemetry.scope(**fields):
            yield
    finally:
        _CONTEXT.reset(token)
