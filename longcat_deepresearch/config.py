from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal

from .models import HarnessOptions


PlanningAcceptance = Literal["standard", "lossless", "structural"]
IntegrityPolicy = Literal["standard", "lossless"]
TerminalToolSchemaPolicy = Literal["drop", "preserve"]
RevisionProtocol = Literal["v1", "v2"]


@dataclass(frozen=True)
class HarnessConfig:
    """Declarative selection of harness behavior."""

    name: str
    planning: HarnessOptions
    candidate_name: str = "candidate_a"
    planning_max_tool_rounds: int = 10
    research_max_tool_rounds: int = 20
    full_run_research_max_tool_rounds: int = 20
    concurrency: int = 3
    planning_acceptance: PlanningAcceptance = "structural"
    integrity_policy: IntegrityPolicy = "standard"
    terminal_tool_schema: TerminalToolSchemaPolicy = "preserve"
    revision_protocol: RevisionProtocol = "v2"
    planning_writer_max_tokens: int = 12000
    planning_writer_terminal_max_tokens: int = 32000
    planning_judge_max_tokens: int = 12000
    planning_judge_terminal_max_tokens: int = 32000
    planning_judge_reject_truncation: bool = False
    subsection_researcher_max_tokens: int = 5000
    subsection_researcher_terminal_max_tokens: int = 32000
    immutable: bool = True

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("HarnessConfig.name must not be empty")
        for field_name in (
            "planning_max_tool_rounds",
            "research_max_tool_rounds",
            "full_run_research_max_tool_rounds",
            "concurrency",
            "planning_writer_max_tokens",
            "planning_writer_terminal_max_tokens",
            "planning_judge_max_tokens",
            "planning_judge_terminal_max_tokens",
            "subsection_researcher_max_tokens",
            "subsection_researcher_terminal_max_tokens",
        ):
            if getattr(self, field_name) < 1:
                raise ValueError(f"{field_name} must be positive")
        allowed = {
            "planning_acceptance": {"standard", "lossless", "structural"},
            "integrity_policy": {"standard", "lossless"},
            "terminal_tool_schema": {"drop", "preserve"},
            "revision_protocol": {"v1", "v2"},
        }
        for field_name, choices in allowed.items():
            if getattr(self, field_name) not in choices:
                raise ValueError(f"invalid {field_name}: {getattr(self, field_name)}")

    def variant(self, *, name: str, **changes: object) -> "HarnessConfig":
        if not name:
            raise ValueError("variant name must not be empty")
        return replace(self, name=name, immutable=False, **changes)


STANDARD_CONFIG = HarnessConfig(
    name="standard",
    planning=HarnessOptions(
        planning_writers=3,
        writer_search=True,
        judge_enabled=True,
        judge_search=True,
        critic_enabled=True,
        critic_search=True,
        reviser_enabled=True,
        plan_compact="off",
        planning_prompt="canonical",
        research_context="full",
        editor="standard",
        reviser_max_tokens=32000,
        planning_repair="full",
        planning_revision_rounds=2,
        editor_revision_rounds=2,
    ),
)

PRESET_CONFIGS: dict[str, HarnessConfig] = {
    STANDARD_CONFIG.name: STANDARD_CONFIG,
}


def get_preset_config(name: str) -> HarnessConfig:
    try:
        return PRESET_CONFIGS[name]
    except KeyError as exc:
        raise ValueError(f"unknown harness preset: {name}") from exc
