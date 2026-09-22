from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal


PlanningCompactMode = Literal[
    "off", "critic_reviser", "critic_reviser_guarded", "downstream", "all"
]
PlanningPromptMode = Literal["canonical", "deconflict"]
ResearchContextMode = Literal["full", "projected"]
EditorMode = Literal["standard", "lossless_patch", "off"]
PlanningRepairPolicy = Literal["full", "off"]


@dataclass(frozen=True)
class HarnessOptions:
    """Small, explicit switches for the research harness stages."""

    planning_writers: int = 3
    writer_search: bool = True
    judge_enabled: bool | None = None
    judge_search: bool = True
    critic_enabled: bool = True
    critic_search: bool | None = None
    reviser_enabled: bool = True
    plan_compact: PlanningCompactMode = "off"
    planning_prompt: PlanningPromptMode = "canonical"
    research_context: ResearchContextMode = "full"
    editor: EditorMode = "lossless_patch"
    reviser_max_tokens: int = 16000
    planning_repair: PlanningRepairPolicy = "full"
    planning_revision_rounds: int = 1
    editor_revision_rounds: int = 2

    def __post_init__(self) -> None:
        if not 1 <= self.planning_writers <= 8:
            raise ValueError("planning_writers must be between 1 and 8")
        if self.judge_enabled is None:
            object.__setattr__(self, "judge_enabled", self.planning_writers > 1)
        if self.critic_search is None:
            object.__setattr__(self, "critic_search", bool(self.judge_enabled))
        if not self.judge_enabled and self.planning_writers != 1:
            raise ValueError("judge can be disabled only with one planning writer")
        if self.reviser_enabled and not self.critic_enabled:
            raise ValueError("reviser requires critic; disable both for judge-only planning")
        if self.plan_compact not in {
            "off", "critic_reviser", "critic_reviser_guarded", "downstream", "all"
        }:
            raise ValueError(f"invalid plan_compact mode: {self.plan_compact}")
        if self.planning_prompt not in {"canonical", "deconflict"}:
            raise ValueError(f"invalid planning_prompt mode: {self.planning_prompt}")
        if self.research_context not in {"full", "projected"}:
            raise ValueError(f"invalid research_context mode: {self.research_context}")
        if self.editor not in {"standard", "lossless_patch", "off"}:
            raise ValueError(f"invalid editor mode: {self.editor}")
        if self.reviser_max_tokens < 1:
            raise ValueError("reviser_max_tokens must be positive")
        if self.planning_repair not in {"full", "off"}:
            raise ValueError(f"invalid planning_repair policy: {self.planning_repair}")
        if self.planning_revision_rounds not in {1, 2}:
            raise ValueError("planning_revision_rounds must be 1 or 2")
        if self.editor_revision_rounds not in {1, 2}:
            raise ValueError("editor_revision_rounds must be 1 or 2")

@dataclass(frozen=True)
class PlanningResult:
    run_dir: Path
    candidate_dir: Path
    research_spec_path: Path


@dataclass(frozen=True)
class SectionSpec:
    section_id: str
    title: str
    markdown: str
    parent_id: str = ""
    parent_title: str = ""
    heading_level: int = 2


@dataclass(frozen=True)
class PipelineResult:
    run_dir: Path
    draft_path: Path
    final_path: Path
