from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from typing import Iterator

from .config import HarnessConfig, STANDARD_CONFIG, get_preset_config

_HARNESS_CONFIG: ContextVar[HarnessConfig] = ContextVar(
    "longcat_deepresearch_harness_config", default=STANDARD_CONFIG
)


def current_harness_config() -> HarnessConfig:
    return _HARNESS_CONFIG.get()


def current_harness_mode() -> str:
    """Compatibility view for callers that only need the preset name."""
    return current_harness_config().name


@contextmanager
def harness_config(config: HarnessConfig) -> Iterator[None]:
    token = _HARNESS_CONFIG.set(config)
    try:
        yield
    finally:
        _HARNESS_CONFIG.reset(token)


@contextmanager
def harness_mode(mode: str) -> Iterator[None]:
    """Backward-compatible named-preset context."""
    with harness_config(get_preset_config(mode)):
        yield
