from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from contextvars import copy_context
import hashlib
import os
from pathlib import Path
import sys

from .backend import HarnessBackend, backend_scope
from . import execution_context as run_context
from . import telemetry
from .editorial import apply_editorial_strategy, _write_editorial_summary
from .config import (
    HarnessConfig,
    get_preset_config,
)
from .models import HarnessOptions, PipelineResult, PlanningResult, SectionSpec
from .planning import generate_candidate
from .policy import current_harness_config, harness_config
from .runtime import _activate_web_fetch_cache, _atomic_write
from .research import (
    _assemble,
    _h2_blocks,
    _parse_sections,
    _relevant_rowbook,
    _research_one_section,
    _subsection_blocks,
)


def _bodies_from_assembled_draft(
    draft: str, expected_sections: list[SectionSpec]
) -> dict[str, str]:
    parsed = _subsection_blocks(draft)
    if not parsed:
        parsed = [
            SectionSpec(section_id, title, block)
            for section_id, title, block in _h2_blocks(draft, "S")
        ]
    found = {section.section_id: section for section in parsed}
    bodies: dict[str, str] = {}
    for expected in expected_sections:
        actual = found.get(expected.section_id)
        if actual is None:
            raise ValueError(f"assembled draft is missing {expected.section_id}")
        if actual.title != expected.title:
            raise ValueError(f"assembled draft changed the title of {expected.section_id}")
        bodies[expected.section_id] = actual.markdown
    return bodies


def run_editorial_pipeline(
    question: str,
    candidate_dir: str | Path,
    *,
    draft_path: str | Path | None = None,
    concurrency: int = 3,
    editor_mode: str = "lossless_patch",
) -> PipelineResult:
    candidate_dir = Path(candidate_dir).expanduser().resolve()
    run_dir = candidate_dir.parent
    research_spec = (candidate_dir / "research_spec.md").read_text(encoding="utf-8")
    report_spec, sections, facts = _parse_sections(research_spec)
    if not sections:
        raise ValueError("ResearchSpec has no editable sections")
    source_path = (
        Path(draft_path).expanduser().resolve()
        if draft_path is not None
        else run_dir / "report_draft.md"
    )
    draft = source_path.read_text(encoding="utf-8")
    bodies = _bodies_from_assembled_draft(draft, sections)
    edited = apply_editorial_strategy(
        question,
        research_spec,
        report_spec,
        sections,
        facts,
        bodies,
        draft,
        run_dir,
        concurrency=concurrency,
        mode=editor_mode,
    )
    final = _assemble(question, sections, edited, strip_ids=True)
    final_path = run_dir / "report_edited.md"
    _atomic_write(final_path, final)
    if editor_mode != "off":
        _write_editorial_summary(run_dir, draft, final)
    return PipelineResult(run_dir, source_path, final_path)


def run_research_pipeline(
    question: str,
    candidate_dir: str | Path,
    *,
    section_ids: set[str] | None = None,
    concurrency: int = 3,
    max_tool_rounds: int = 8,
    refine: bool = True,
    research_context: str = "full",
    editor_mode: str = "lossless_patch",
) -> PipelineResult:
    candidate_dir = Path(candidate_dir).expanduser().resolve()
    run_dir = candidate_dir.parent
    _activate_web_fetch_cache(run_dir)
    research_spec = (candidate_dir / "research_spec.md").read_text(encoding="utf-8")
    report_spec, all_sections, facts = _parse_sections(research_spec)
    sections = [s for s in all_sections if section_ids is None or s.section_id in section_ids]
    if not sections:
        raise ValueError("no matching sections to research")
    if section_ids is not None and set(s.section_id for s in sections) != section_ids:
        missing = section_ids - set(s.section_id for s in sections)
        raise ValueError(f"unknown section IDs: {sorted(missing)}")

    section_dir = run_dir / "sections"
    trace_dir = run_dir / "traces"
    bodies: dict[str, str] = {}
    retry_failed_sections = (
        current_harness_config().planning_acceptance == "structural"
    )

    def execute_sections(
        pending: list[SectionSpec],
        *,
        attempt: int,
    ) -> dict[str, tuple[SectionSpec, Exception]]:
        failures: dict[str, tuple[SectionSpec, Exception]] = {}
        with ThreadPoolExecutor(
            max_workers=max(1, min(concurrency, len(pending)))
        ) as executor:
            futures = {}
            for section in pending:
                context = copy_context()
                future = executor.submit(
                    context.run,
                    _research_one_section,
                    question,
                    report_spec,
                    section,
                    _relevant_rowbook(section.section_id, facts),
                    max_tool_rounds=max_tool_rounds,
                    context_mode=research_context,
                    attempt=attempt,
                )
                futures[future] = section
            for future in as_completed(futures):
                section = futures[future]
                try:
                    body, trace = future.result()
                except Exception as exc:
                    if not retry_failed_sections:
                        raise
                    failures[section.section_id] = (section, exc)
                    _atomic_write(
                        section_dir
                        / f"{section.section_id}.attempt-{attempt}.error.txt",
                        f"{type(exc).__name__}: {exc}\n",
                    )
                    continue
                bodies[section.section_id] = body
                _atomic_write(section_dir / f"{section.section_id}.md", body)
                trace_name = (
                    f"research_{section.section_id}.md"
                    if attempt == 1
                    else f"research_{section.section_id}.retry-{attempt - 1}.md"
                )
                _atomic_write(
                    trace_dir / trace_name,
                    "\n".join(trace).rstrip() + "\n",
                )
        return failures

    failures = execute_sections(sections, attempt=1)
    if failures and retry_failed_sections:
        retry_sections = [item[0] for item in failures.values()]
        _atomic_write(
            run_dir / "research_retry_summary.md",
            "# Research Section Retry\n\n"
            + "\n".join(
                f"- {section.section_id}: retrying after "
                f"{type(error).__name__}: {error}"
                for section, error in failures.values()
            )
            + "\n",
        )
        failures = execute_sections(retry_sections, attempt=2)
    if failures:
        details = "; ".join(
            f"{section_id}: {type(error).__name__}: {error}"
            for section_id, (_, error) in sorted(failures.items())
        )
        raise RuntimeError(
            "research sections remain invalid after bounded retries: " + details
        )

    draft = _assemble(question, sections, bodies)
    draft_path = run_dir / "report_draft.md"
    _atomic_write(draft_path, draft)

    editing_enabled = refine and editor_mode != "off"
    if editing_enabled:
        bodies = apply_editorial_strategy(
            question,
            research_spec,
            report_spec,
            sections,
            facts,
            bodies,
            draft,
            run_dir,
            concurrency=concurrency,
            mode=editor_mode,
        )
    final = _assemble(question, sections, bodies, strip_ids=True)
    final_path = run_dir / "report_final.md"
    _atomic_write(final_path, final)
    if editing_enabled:
        _write_editorial_summary(run_dir, draft, final)
    return PipelineResult(run_dir, draft_path, final_path)


class LongCatDeepResearch:
    """Modular harness with a user-provided model and web backend."""

    def __init__(
        self,
        options: HarnessOptions | None = None,
        *,
        preset: str | None = None,
        config: HarnessConfig | None = None,
        backend: HarnessBackend | None = None,
        **overrides: object,
    ) -> None:
        if config is not None:
            if preset is not None or options is not None or overrides:
                raise ValueError(
                    "pass config alone, or select a preset/HarnessOptions, not both"
                )
            selected_config = config
        else:
            selected_config = get_preset_config(preset or "standard")
            if options is not None and overrides:
                raise ValueError("pass HarnessOptions or keyword overrides, not both")
            if options is not None or overrides:
                planning = options or HarnessOptions(**overrides)
                selected_config = selected_config.variant(
                    name="custom",
                    planning=planning,
                )
        self.preset = selected_config.name
        self.config = selected_config
        self.options = selected_config.planning
        self.backend = backend

    def _fixed_value(self, requested: int | None, configured: int, label: str) -> int:
        if self.config.immutable and requested not in {None, configured}:
            raise ValueError(f"{self.preset} fixes {label}={configured}")
        return configured if requested is None else requested

    def _candidate_name(self, requested: str) -> str:
        if self.config.immutable and requested != self.config.candidate_name:
            raise ValueError(
                f"{self.preset} fixes candidate_name='{self.config.candidate_name}'"
            )
        return requested

    def plan(
        self,
        question: str,
        *,
        run_root: str | Path,
        candidate_name: str = "candidate_a",
        max_tool_rounds: int | None = None,
    ) -> PlanningResult:
        candidate_name = self._candidate_name(candidate_name)
        rounds = self._fixed_value(
            max_tool_rounds,
            self.config.planning_max_tool_rounds,
            "max_tool_rounds",
        )
        with backend_scope(self.backend), harness_config(self.config):
            result = generate_candidate(
                question,
                run_root=run_root,
                candidate_name=candidate_name,
                max_tool_rounds=rounds,
                options=self.config.planning,
            )
            return result

    def research(
        self,
        question: str,
        candidate_dir: str | Path,
        *,
        section_ids: set[str] | None = None,
        concurrency: int | None = None,
        max_tool_rounds: int | None = None,
    ) -> PipelineResult:
        rounds = self._fixed_value(
            max_tool_rounds,
            self.config.research_max_tool_rounds,
            "max_tool_rounds",
        )
        workers = self._fixed_value(concurrency, self.config.concurrency, "concurrency")
        if self.config.immutable:
            if Path(candidate_dir).expanduser().resolve().name != self.config.candidate_name:
                raise ValueError(
                    f"{self.preset} fixes candidate directory name to "
                    f"'{self.config.candidate_name}'"
                )
            if section_ids is not None:
                raise ValueError(
                    f"{self.preset} full-run preset does not allow section filtering"
                )
        with backend_scope(self.backend), harness_config(self.config):
            return run_research_pipeline(
                question,
                candidate_dir,
                section_ids=section_ids,
                concurrency=workers,
                max_tool_rounds=rounds,
                refine=self.config.planning.editor != "off",
                research_context=self.config.planning.research_context,
                editor_mode=self.config.planning.editor,
            )

    def edit(
        self,
        question: str,
        candidate_dir: str | Path,
        *,
        draft_path: str | Path | None = None,
        concurrency: int | None = None,
    ) -> PipelineResult:
        workers = self._fixed_value(concurrency, self.config.concurrency, "concurrency")
        if self.config.immutable:
            if Path(candidate_dir).expanduser().resolve().name != self.config.candidate_name:
                raise ValueError(
                    f"{self.preset} fixes candidate directory name to "
                    f"'{self.config.candidate_name}'"
                )
            if draft_path is not None:
                raise ValueError(
                    f"{self.preset} uses the canonical report_draft.md"
                )
        with backend_scope(self.backend), harness_config(self.config):
            return run_editorial_pipeline(
                question,
                candidate_dir,
                draft_path=draft_path,
                concurrency=workers,
                editor_mode=self.config.planning.editor,
            )

    def run(
        self,
        question: str,
        *,
        run_root: str | Path,
        candidate_name: str = "candidate_a",
        concurrency: int | None = None,
        max_tool_rounds: int | None = None,
    ) -> PipelineResult:
        candidate_name = self._candidate_name(candidate_name)
        plan_rounds = self._fixed_value(
            max_tool_rounds,
            self.config.planning_max_tool_rounds,
            "max_tool_rounds",
        )
        research_rounds = self._fixed_value(
            max_tool_rounds,
            self.config.full_run_research_max_tool_rounds,
            "max_tool_rounds",
        )
        workers = self._fixed_value(concurrency, self.config.concurrency, "concurrency")
        question = str(question or "")
        run_id = hashlib.sha256(question.strip().encode("utf-8")).hexdigest()[:16]
        run_dir = Path(run_root).expanduser().resolve() / run_id
        case_id = os.getenv("DR_CASE_ID") or run_id
        with backend_scope(self.backend), run_context.scope(
            idx=case_id,
            run_id=run_id,
            run_dir=str(run_dir),
        ):
            try:
                planning = self.plan(
                    question,
                    run_root=run_root,
                    candidate_name=candidate_name,
                    max_tool_rounds=plan_rounds,
                )
                return self.research(
                    question,
                    planning.candidate_dir,
                    concurrency=workers,
                    max_tool_rounds=research_rounds,
                )
            finally:
                try:
                    telemetry.write_summary(run_dir)
                except Exception as exc:
                    print(
                        "warning: usage telemetry summary failed: "
                        f"{type(exc).__name__}: {exc}",
                        file=sys.stderr,
                    )
