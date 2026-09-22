from __future__ import annotations

import hashlib
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextvars import copy_context
from pathlib import Path

from . import execution_context as run_context
from .integrity import (
    _high_priority_lead_errors,
    _planning_semantic_regression_errors,
)
from .models import HarnessOptions, PlanningResult
from .policy import current_harness_config
from .repair import apply_exact_patches
from .prompts import (
    LEGACY_SINGLE_CRITIC_PROMPT,
    PLANNING_CRITIC_PROMPT,
    PLANNING_JUDGE_PROMPT,
    PLANNING_REVISER_PROMPT,
    PLANNING_REVISER_V2_PROMPT,
    PLANNING_SYSTEM_PROMPT,
    PLANNING_TOOL_FREE_CRITIC_PROMPT,
    PLANNING_TOOL_FREE_JUDGE_PROMPT,
    planning_option_prompt,
)
from .runtime import (
    DEFAULT_RUN_ROOT,
    MODEL,
    PLANNING_FILE_MARKER_RE,
    SECTION_RE,
    SUBSECTION_RE,
    ResearchAgent,
    _activate_web_fetch_cache,
    _atomic_write,
    _chat,
    _extract_terminal_text,
    _uses_auto_markdown_recovery,
)

def _run_planner(
    question: str,
    max_tool_rounds: int,
    writer_name: str = "agent_a",
    *,
    search_enabled: bool = True,
    system_prompt: str = PLANNING_SYSTEM_PROMPT,
) -> tuple[str, list[str]]:
    agent = ResearchAgent(
        system_prompt=system_prompt,
        label=f"Planning Writer {writer_name}",
        call_type="planning_writer_turn",
        actor=writer_name,
        trajectory_id=f"planning-writer-{writer_name}",
    )
    action = "Search and develop" if search_enabled else "Develop"
    user_prompt = (
        f"Research question:\n\n{question}\n\n"
        f"You are independent Planning Writer `{writer_name}`. {action} your own complete "
        "blueprint without assuming another writer will fill its gaps."
    )
    if search_enabled:
        config = current_harness_config()
        body, trace = agent.research(
            user_prompt=user_prompt,
            max_tool_rounds=max_tool_rounds,
            min_search_calls=0,
            min_fetch_calls=0,
            final_instruction="Produce the marked research_spec.write.md now.",
            max_tokens=config.planning_writer_max_tokens,
            terminal_max_tokens=config.planning_writer_terminal_max_tokens,
        )
    else:
        body = agent.complete(
            user_prompt,
            max_tokens=32000,
            empty_error=f"planning writer {writer_name} returned empty content",
        )
        trace = [
            f"# Planning Writer {writer_name} Trace",
            "",
            f"Model: `{MODEL}`",
            "",
            "Tool-free Planning Writer.",
        ]
    return body, trace


def _compile_planning_submission(body: str, spec_name: str) -> str:
    """Compile provider output into the original FILE transport contract."""
    text = str(body or "").strip()
    marker = PLANNING_FILE_MARKER_RE.search(text)
    if marker:
        text = marker.group(2).strip()
    if text.startswith("```") and text.endswith("```"):
        text = re.sub(r"^```(?:markdown|md)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text).strip()
    report_start = re.search(r"^#\s+ReportSpec\s*$", text, re.IGNORECASE | re.MULTILINE)
    if report_start:
        # Text before the root heading is transport commentary, not part of the
        # executable spec. This path is used only after strict parsing failed.
        text = text[report_start.start() :].strip()
    # Delimiter spelling is transport syntax, not research semantics.
    text = re.sub(
        r"^(#{2,3}\s+S\d+(?:\.\d+)?)\s*[|:]\s*",
        r"\1｜",
        text,
        flags=re.MULTILINE,
    )
    return f"<!-- FILE: {spec_name} -->\n{text}\n<!-- END FILE -->\n"


def _parse_bundle(text: str, required_name: str = "research_spec.md") -> dict[str, str]:
    documents: dict[str, str] = {}
    for name, content in PLANNING_FILE_MARKER_RE.findall(text):
        documents[name.lower()] = content.strip() + "\n"
    required = {required_name}
    missing = required - set(documents)
    if missing:
        raise ValueError(f"terminal answer is missing file markers for: {sorted(missing)}")
    return documents


def _split_research_spec(spec: str) -> tuple[str, str]:
    marker = re.search(r"^#\s+Rowbook\s*$", spec, re.IGNORECASE | re.MULTILINE)
    if not marker:
        # New Markdown-first specs embed each Rowbook contract directly under
        # its subsection. Keep this two-part return for legacy callers.
        return spec.strip() + "\n", ""
    return spec[: marker.start()].strip() + "\n", spec[marker.start() :].strip() + "\n"



def _validate_bundle(documents: dict[str, str], spec_name: str = "research_spec.md") -> list[str]:
    errors: list[str] = []
    spec = documents[spec_name]
    report_spec, rowbook = _split_research_spec(spec)
    if not re.search(r"^#\s+ReportSpec\s*$", report_spec, re.IGNORECASE | re.MULTILINE):
        errors.append("research_spec.md must begin its outline with `# ReportSpec`")

    sections = SECTION_RE.findall(spec)
    subsections = SUBSECTION_RE.findall(report_spec)
    if not sections:
        errors.append("ReportSpec has no `## S1｜...` main sections")
    if not subsections:
        errors.append("ReportSpec has no `### S1.1｜...` subsections")
    if len(sections) != len(set(sections)):
        errors.append("ReportSpec contains duplicate section IDs")
    if len(subsections) != len(set(subsections)):
        errors.append("ReportSpec contains duplicate subsection IDs")
    if sections and sections != [f"S{i}" for i in range(1, len(sections) + 1)]:
        errors.append("section IDs must be consecutive S1, S2, ...")
    if subsections and len(subsections) < 6:
        errors.append(f"ReportSpec must contain at least 6 substantive subsections, found {len(subsections)}")

    per_parent: dict[str, list[int]] = {}
    for subsection_id in subsections:
        parent, ordinal = subsection_id.split(".")
        if parent not in sections:
            errors.append(f"{subsection_id} has no matching parent main section")
        per_parent.setdefault(parent, []).append(int(ordinal))
    for parent in sections:
        ordinals = per_parent.get(parent, [])
        if not ordinals:
            errors.append(f"{parent} has no substantive subsections")
        elif ordinals != list(range(1, len(ordinals) + 1)):
            errors.append(f"{parent} subsection IDs must be consecutive from {parent}.1")

    subsection_blocks = re.split(
        r"(?=^###\s+S\d+\.\d+\s*[｜|:])", report_spec, flags=re.MULTILINE
    )[1:]
    for block in subsection_blocks:
        match = SUBSECTION_RE.search(block)
        label = match.group(1) if match else "unknown"
        if not ("#### 内容范围" in block or "#### What to Cover" in block):
            errors.append(f"{label} is missing What to Cover")
        if not ("#### 研究问题" in block or "#### Research Questions" in block):
            errors.append(f"{label} is missing Research Questions")
        if not (
            "#### 必需实体 / 案例" in block
            or "#### Required Entities / Cases" in block
        ):
            errors.append(f"{label} is missing Required Entities / Cases")
        if not ("#### 来源线索" in block or "#### Source Leads" in block):
            errors.append(f"{label} is missing Source Leads")

    if rowbook:
        errors.append("do not create a separate Rowbook; embed research questions under each subsection")

    return errors


def _repair_bundle(
    question: str,
    raw: str,
    errors: list[str],
    *,
    spec_name: str = "research_spec.md",
    system_prompt: str = PLANNING_REVISER_PROMPT,
    required_context: str = "",
    artifact_prefix: str = "repair",
    repair_index: int = 1,
) -> str:
    structural_recovery = (
        current_harness_config().planning_acceptance == "structural"
    )
    semantic_errors = []
    label_aliases = {
        "missing What to Cover": re.compile(
            r"^\*\*(?:What to Cover(?:（内容范围）|\s*\(内容范围\))?|内容范围)\*\*$",
            re.MULTILINE,
        ),
        "missing Research Questions": re.compile(
            r"^\*\*(?:Research Questions(?:（研究问题）|\s*\(研究问题\))?|研究问题)\*\*$",
            re.MULTILINE,
        ),
        "missing Required Entities / Cases": re.compile(
            r"^\*\*(?:Required Entities / Cases(?:（必需实体[／/]案例）|\s*\(必需实体\s*/\s*案例\))?|必需实体\s*[／/]\s*案例)\*\*$",
            re.MULTILINE,
        ),
        "missing Source Leads": re.compile(
            r"^\*\*(?:Source Leads(?:（来源线索）|\s*\(来源线索\))?|来源线索)\*\*$",
            re.MULTILINE,
        ),
    }
    for error in errors:
        lowered = error.lower()
        if any(
            marker in lowered
            for marker in (
                "lossless planning guard",
                "high-priority",
                "required hp",
                "lost url",
                "lost numeric",
                "lost required entity",
            )
        ):
            semantic_errors.append(error)
            continue
        for marker, alias_pattern in label_aliases.items():
            if marker.lower() in lowered and not alias_pattern.search(raw):
                semantic_errors.append(error)
                break
    if semantic_errors and not structural_recovery:
        raise ValueError(
            "planning repair is format-only; semantic errors are not repairable: "
            + "; ".join(semantic_errors)
        )
    prompt = f"""The planning bundle below failed deterministic validation.

Research question:
{question}

Validation errors:
{chr(10).join(f'- {error}' for error in errors)}

Return only the smallest exact-match patch transaction needed to repair it.
Do not rewrite or repeat the complete document. Every `old` anchor must be copied
verbatim from the bundle and occur exactly once. Preserve all content outside
the patches.
Every H3 subsection MUST contain four separate required H4 heading lines. Each line must
be exactly one of the accepted literals below -- do not rename it, add words to
it, combine the two headings, or replace Markdown headings with bold text:

- `#### What to Cover` or `#### 内容范围`
- `#### Research Questions` or `#### 研究问题`
- `#### Required Entities / Cases` or `#### 必需实体 / 案例`
- `#### Source Leads` or `#### 来源线索`

Create at least 6 substantive H3 subsections, with no arbitrary maximum.
Preserve all explicit user constraints and keep subsection IDs consecutive.
Source Leads must be concise authoritative URLs paired with what each source
may establish; they are leads for later fetch verification, not final evidence.
{required_context}
Return exactly:
<!-- PATCHES -->
{{"patches":[{{"id":"P1","old":"exact original text","new":"replacement text","expected_count":1}}]}}
<!-- END PATCHES -->

When the same malformed literal must be replaced globally, use one patch with
its exact `expected_count`; do not emit one copy per occurrence.

Bundle to repair:
{raw}
"""
    actor = f"{artifact_prefix}.repair-{repair_index}"
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt},
    ]
    with run_context.scope(
        call_type="planning_repair_patch",
        actor=actor,
        trajectory_id=f"planning-repair-{artifact_prefix}",
        turn_index=repair_index,
        degraded=True,
    ):
        message = _chat(messages, tools=False, max_tokens=8000)
    patch = _extract_terminal_text(message)
    if not patch or message.get("tool_calls"):
        raise RuntimeError("planning repair patch returned invalid content")
    return apply_exact_patches(raw, patch)


def _parse_and_validate_bundle(raw: str, spec_name: str) -> tuple[dict[str, str], list[str]]:
    """Parse a planning bundle and express marker failures as validation errors."""
    try:
        documents = _parse_bundle(raw, spec_name)
        return documents, _validate_bundle(documents, spec_name)
    except ValueError as exc:
        return {}, [str(exc)]


def _candidate_for_next_planning_stage(
    raw: str,
    spec_name: str,
) -> tuple[str, list[str]]:
    """Return usable candidate text while keeping structural errors diagnostic."""
    documents, errors = _parse_and_validate_bundle(raw, spec_name)
    if documents:
        return documents[spec_name], errors
    recovered = _compile_planning_submission(raw, spec_name)
    documents, recovered_errors = _parse_and_validate_bundle(recovered, spec_name)
    if documents:
        return documents[spec_name], recovered_errors
    return str(raw or "").strip(), errors


def _record_planning_diagnostics(
    raw_dir: Path,
    artifact_prefix: str,
    errors: list[str],
) -> None:
    if not errors:
        return
    _atomic_write(
        raw_dir / f"{artifact_prefix}.diagnostics.txt",
        "Intermediate planning diagnostics; downstream stages may repair these issues:\n"
        + "\n".join(f"- {error}" for error in errors)
        + "\n",
    )


def _repair_until_valid_standard(
    question: str,
    raw: str,
    errors: list[str],
    *,
    spec_name: str,
    system_prompt: str,
    raw_dir: Path,
    artifact_prefix: str,
    max_repairs: int = 3,
    required_context: str = "",
    required_lead_critique: str = "",
) -> tuple[str, dict[str, str], list[str]]:
    """Retry only invalid planning output and retain every repair response."""
    def validate(candidate: str) -> tuple[dict[str, str], list[str]]:
        candidate_documents, candidate_errors = _parse_and_validate_bundle(
            candidate, spec_name
        )
        if not candidate_errors and required_lead_critique:
            candidate_errors.extend(
                _high_priority_lead_errors(
                    required_lead_critique, candidate_documents[spec_name]
                )
            )
        return candidate_documents, candidate_errors

    # Recompute the baseline so every repair is compared against the exact same
    # validator. A repair may replace the current candidate only when it
    # strictly reduces the remaining contract violations.
    documents, validated_errors = validate(raw)
    if validated_errors:
        errors = validated_errors

    if errors and _uses_auto_markdown_recovery():
        recovered = _compile_planning_submission(raw, spec_name)
        recovered_documents, recovered_errors = validate(recovered)
        if not recovered_errors:
            _atomic_write(raw_dir / f"{artifact_prefix}.recovered.md", recovered)
            return recovered, recovered_documents, []
        if recovered_documents:
            # Transport recovery succeeded. Keep its precise semantic errors
            # instead of misleading the next repair with "missing markers".
            raw, documents, errors = recovered, recovered_documents, recovered_errors
            _atomic_write(
                raw_dir / f"{artifact_prefix}.recovered.invalid.md", recovered
            )

    for attempt in range(1, max_repairs + 1):
        if not errors:
            break
        candidate_raw = _repair_bundle(
            question,
            raw,
            errors,
            spec_name=spec_name,
            system_prompt=system_prompt,
            required_context=required_context,
            artifact_prefix=artifact_prefix,
            repair_index=attempt,
        )
        _atomic_write(
            raw_dir / f"{artifact_prefix}.repair-{attempt}.raw.md", candidate_raw
        )
        candidate_documents, candidate_errors = validate(candidate_raw)
        if candidate_errors and _uses_auto_markdown_recovery():
            recovered = _compile_planning_submission(candidate_raw, spec_name)
            recovered_documents, recovered_errors = validate(recovered)
            if not recovered_errors:
                candidate_raw = recovered
                candidate_documents = recovered_documents
                candidate_errors = []
                _atomic_write(
                    raw_dir / f"{artifact_prefix}.repair-{attempt}.recovered.md",
                    recovered,
                )
            elif recovered_documents:
                candidate_raw = recovered
                candidate_documents = recovered_documents
                candidate_errors = recovered_errors
                _atomic_write(
                    raw_dir
                    / f"{artifact_prefix}.repair-{attempt}.recovered.invalid.md",
                    recovered,
                )
        if len(candidate_errors) < len(errors):
            raw, documents, errors = (
                candidate_raw,
                candidate_documents,
                candidate_errors,
            )
        else:
            _atomic_write(
                raw_dir / f"{artifact_prefix}.repair-{attempt}.rejected.md",
                candidate_raw,
            )
    return raw, documents, errors


def _repair_until_valid(
    question: str,
    raw: str,
    errors: list[str],
    *,
    spec_name: str,
    system_prompt: str,
    raw_dir: Path,
    artifact_prefix: str,
    max_repairs: int = 1,
    required_context: str = "",
    required_lead_critique: str = "",
    lossless_source: str = "",
) -> tuple[str, dict[str, str], list[str]]:
    """Retry only invalid planning output and retain every repair response."""
    def validate(candidate: str) -> tuple[dict[str, str], list[str]]:
        candidate_documents, candidate_errors = _parse_and_validate_bundle(
            candidate, spec_name
        )
        if not candidate_errors and required_lead_critique:
            candidate_errors.extend(
                _high_priority_lead_errors(
                    required_lead_critique, candidate_documents[spec_name]
                )
            )
        if candidate_documents and lossless_source:
            candidate_errors.extend(
                _planning_semantic_regression_errors(
                    lossless_source, candidate_documents[spec_name]
                )
            )
        return candidate_documents, candidate_errors

    # Recompute the baseline so every repair is compared against the exact same
    # validator. A repair may replace the current candidate only when it
    # strictly reduces the remaining contract violations.
    documents, validated_errors = validate(raw)
    if validated_errors:
        errors = validated_errors

    if errors and _uses_auto_markdown_recovery():
        recovered = _compile_planning_submission(raw, spec_name)
        recovered_documents, recovered_errors = validate(recovered)
        if not recovered_errors:
            _atomic_write(raw_dir / f"{artifact_prefix}.recovered.md", recovered)
            return recovered, recovered_documents, []
        if recovered_documents:
            # Transport recovery succeeded. Keep its precise semantic errors
            # instead of misleading the next repair with "missing markers".
            raw, documents, errors = recovered, recovered_documents, recovered_errors
            _atomic_write(
                raw_dir / f"{artifact_prefix}.recovered.invalid.md", recovered
            )

    for attempt in range(1, max_repairs + 1):
        if not errors:
            break
        candidate_raw = _repair_bundle(
            question,
            raw,
            errors,
            spec_name=spec_name,
            system_prompt=system_prompt,
            required_context=(
                required_context
                + (
                    "\n\n# Lossless Baseline\n\n"
                    "The repaired ResearchSpec must preserve every URL, exact numeric fact, "
                    "year/date, and Required Entity / Case from this validated baseline. "
                    "Move or deduplicate them if needed, but do not omit them.\n\n"
                    + lossless_source
                    if lossless_source
                    else ""
                )
            ),
            artifact_prefix=artifact_prefix,
            repair_index=attempt,
        )
        _atomic_write(
            raw_dir / f"{artifact_prefix}.repair-{attempt}.raw.md", candidate_raw
        )
        candidate_documents, candidate_errors = validate(candidate_raw)
        if candidate_errors and _uses_auto_markdown_recovery():
            recovered = _compile_planning_submission(candidate_raw, spec_name)
            recovered_documents, recovered_errors = validate(recovered)
            if not recovered_errors:
                candidate_raw = recovered
                candidate_documents = recovered_documents
                candidate_errors = []
                _atomic_write(
                    raw_dir / f"{artifact_prefix}.repair-{attempt}.recovered.md",
                    recovered,
                )
            elif recovered_documents:
                candidate_raw = recovered
                candidate_documents = recovered_documents
                candidate_errors = recovered_errors
                _atomic_write(
                    raw_dir
                    / f"{artifact_prefix}.repair-{attempt}.recovered.invalid.md",
                    recovered,
                )
        if len(candidate_errors) < len(errors):
            raw, documents, errors = (
                candidate_raw,
                candidate_documents,
                candidate_errors,
            )
        else:
            _atomic_write(
                raw_dir / f"{artifact_prefix}.repair-{attempt}.rejected.md",
                candidate_raw,
            )
    return raw, documents, errors



def _judge_research_specs(
    question: str,
    writer_specs: dict[str, str],
    max_tool_rounds: int,
    *,
    search_enabled: bool = True,
    system_prompt: str | None = None,
) -> tuple[str, list[str]]:
    candidates = "\n\n".join(
        f"# Independent Candidate: {name}\n\n{spec}" for name, spec in writer_specs.items()
    )
    agent = ResearchAgent(
        system_prompt=system_prompt or (
            PLANNING_JUDGE_PROMPT if search_enabled else PLANNING_TOOL_FREE_JUDGE_PROMPT
        ),
        label="Planning Judge",
        call_type="planning_judge_turn",
        actor="planning_judge",
        trajectory_id="planning-judge",
    )
    user_prompt = f"# Original Question\n\n{question}\n\n{candidates}"
    if search_enabled:
        config = current_harness_config()
        body, trace = agent.research(
            user_prompt=user_prompt,
            max_tool_rounds=max_tool_rounds,
            min_search_calls=0,
            min_fetch_calls=0,
            final_instruction="Produce the merged marked research_spec.write.md now.",
            max_tokens=config.planning_judge_max_tokens,
            terminal_max_tokens=config.planning_judge_terminal_max_tokens,
            reject_truncation=config.planning_judge_reject_truncation,
        )
    else:
        body = agent.complete(
            user_prompt,
            max_tokens=32000,
            empty_error="planning judge returned empty content",
        )
        trace = [
            "# Planning Judge Trace",
            "",
            f"Model: `{MODEL}`",
            "",
            "Tool-free candidate merge.",
        ]
    return body, trace


def _criticize_research_spec(
    question: str,
    writer_spec: str,
    max_tool_rounds: int,
    *,
    search_enabled: bool,
    system_prompt: str | None = None,
    legacy_single_writer: bool = False,
) -> tuple[str, list[str]]:
    draft_label = "Writer Draft" if legacy_single_writer else "Judge-Merged Draft"
    user_prompt = f"# Original Question\n\n{question}\n\n# {draft_label}\n\n{writer_spec}"
    if not search_enabled:
        agent = ResearchAgent(
            system_prompt=system_prompt or LEGACY_SINGLE_CRITIC_PROMPT,
            label="Planning Critic",
            call_type=(
                "planning_legacy_critic" if legacy_single_writer else "planning_critic_turn"
            ),
            actor="planning_critic",
            trajectory_id=(
                "planning-legacy-critic" if legacy_single_writer else "planning-critic"
            ),
        )
        critique = agent.complete(
            user_prompt,
            max_tokens=8000,
            empty_error="planning critic returned empty content",
        )
        trace = [
            "# Planning Critic Trace",
            "",
            f"Model: `{MODEL}`",
            "",
            (
                "Tool-free legacy single-writer critique."
                if legacy_single_writer
                else "Tool-free critique of the merged Planning draft."
            ),
        ]
        return critique.strip() + "\n", trace

    agent = ResearchAgent(
        system_prompt=system_prompt or PLANNING_CRITIC_PROMPT,
        label="Planning Critic",
        call_type="planning_critic_turn",
        actor="planning_critic",
        trajectory_id="planning-critic",
    )
    critique, trace = agent.research(
        user_prompt=user_prompt,
        max_tool_rounds=max_tool_rounds,
        min_search_calls=0,
        min_fetch_calls=0,
        final_instruction=(
            "Finish the Markdown critique now. Begin with `# ResearchSpec Critique` and "
            "include concrete missing named methods and source leads."
        ),
    )
    if not critique:
        raise RuntimeError("planning critic returned empty content")
    return critique.strip() + "\n", trace


def _revise_research_spec(
    question: str,
    writer_spec: str,
    critique: str,
    *,
    system_prompt: str = PLANNING_REVISER_PROMPT,
    max_tokens: int = 16000,
) -> str:
    agent = ResearchAgent(
        system_prompt=system_prompt,
        label="Planning Reviser",
        call_type="planning_reviser",
        actor="planning_reviser",
        trajectory_id="planning-reviser",
    )
    body = agent.complete(
        (
            f"# Original Question\n\n{question}\n\n"
            f"# Writer Draft\n\n{writer_spec}\n\n"
            f"# Critic Findings\n\n{critique}"
        ),
        max_tokens=max_tokens,
        empty_error="planning reviser returned empty content",
    )
    return body


def _reviser_system_prompt() -> str:
    return (
        PLANNING_REVISER_V2_PROMPT
        if current_harness_config().revision_protocol == "v2"
        else PLANNING_REVISER_PROMPT
    )



def generate_standard_candidate(
    question: str,
    *,
    run_root: str | Path = DEFAULT_RUN_ROOT,
    candidate_name: str = "candidate_a",
    max_tool_rounds: int = 10,
    planning_writers: int = 3,
) -> PlanningResult:
    question = str(question or "").strip()
    if not question:
        raise ValueError("question must not be empty")
    if not re.fullmatch(r"candidate_[a-z0-9_-]+", candidate_name):
        raise ValueError("candidate_name must look like candidate_a")
    if not 1 <= planning_writers <= 8:
        raise ValueError("planning_writers must be between 1 and 8")

    run_id = hashlib.sha256(question.encode("utf-8")).hexdigest()[:16]
    run_dir = Path(run_root).expanduser().resolve() / run_id
    candidate_dir = run_dir / candidate_name
    run_dir.mkdir(parents=True, exist_ok=True)
    _activate_web_fetch_cache(run_dir)
    _atomic_write(run_dir / "question.md", f"# Research Question\n\n{question}\n")
    raw_dir = run_dir / "planning_raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    writer_names = tuple(f"agent_{chr(ord('a') + index)}" for index in range(planning_writers))
    writer_runs: dict[str, tuple[str, list[str]]] = {}
    with ThreadPoolExecutor(max_workers=len(writer_names)) as executor:
        futures = {}
        for name in writer_names:
            context = copy_context()
            future = executor.submit(
                context.run, _run_planner, question, max_tool_rounds, name
            )
            futures[future] = name
        for future in as_completed(futures):
            name = futures[future]
            writer_runs[name] = future.result()
            # Persist terminal output before parsing so a later validation
            # exception cannot erase the evidence needed to diagnose it.
            _atomic_write(raw_dir / "writers" / f"{name}.raw.md", writer_runs[name][0])

    writer_specs: dict[str, str] = {}
    writer_traces: dict[str, list[str]] = {}
    for name in writer_names:
        writer_raw, writer_trace = writer_runs[name]
        writer_traces[name] = writer_trace
        writer_documents, writer_errors = _parse_and_validate_bundle(
            writer_raw, "research_spec.write.md"
        )
        if writer_errors:
            writer_raw, writer_documents, writer_errors = _repair_until_valid_standard(
                question,
                writer_raw,
                writer_errors,
                spec_name="research_spec.write.md",
                system_prompt=PLANNING_SYSTEM_PROMPT,
                raw_dir=raw_dir,
                artifact_prefix=f"writer-{name}",
            )
        if writer_errors:
            raise RuntimeError(
                f"{name} writer draft remains invalid after repair: " + "; ".join(writer_errors)
            )
        writer_specs[name] = writer_documents["research_spec.write.md"]

    judge_trace: list[str] = []
    if planning_writers == 1:
        # This is the pre-multi-writer topology: one search-enabled Writer,
        # followed directly by a tool-free Critic and the Reviser.
        writer_spec = writer_specs[writer_names[0]]
    else:
        judge_raw, judge_trace = _judge_research_specs(
            question, writer_specs, max_tool_rounds=max_tool_rounds
        )
        _atomic_write(raw_dir / "judge.raw.md", judge_raw)
        judge_documents, judge_errors = _parse_and_validate_bundle(
            judge_raw, "research_spec.write.md"
        )
        if judge_errors:
            judge_raw, judge_documents, judge_errors = _repair_until_valid_standard(
                question,
                judge_raw,
                judge_errors,
                spec_name="research_spec.write.md",
                system_prompt=PLANNING_JUDGE_PROMPT,
                raw_dir=raw_dir,
                artifact_prefix="judge",
            )
        if judge_errors:
            raise RuntimeError(
                "judge draft remains invalid after repair: " + "; ".join(judge_errors)
            )
        writer_spec = judge_documents["research_spec.write.md"]

    critique, critic_trace = _criticize_research_spec(
        question,
        writer_spec,
        max_tool_rounds=max_tool_rounds,
        search_enabled=planning_writers > 1,
    )
    _atomic_write(raw_dir / "critic.raw.md", critique)
    final_raw = _revise_research_spec(question, writer_spec, critique)
    _atomic_write(raw_dir / "reviser.raw.md", final_raw)
    documents, errors = _parse_and_validate_bundle(final_raw, "research_spec.md")
    if not errors:
        errors.extend(_high_priority_lead_errors(critique, documents["research_spec.md"]))
    if errors:
        final_raw, documents, errors = _repair_until_valid_standard(
            question,
            final_raw,
            errors,
            spec_name="research_spec.md",
            system_prompt=PLANNING_REVISER_PROMPT,
            raw_dir=raw_dir,
            artifact_prefix="reviser",
            required_context=(
                "The Critic findings below are binding. Every HP ID must appear either "
                "where absorbed in an H3 or under `### Rejected High-Priority Leads` "
                "with a concrete reason.\n\n# Critic Findings\n\n" + critique
            ),
            required_lead_critique=critique,
        )
    if errors:
        raise RuntimeError("revised planning bundle remains invalid after repair: " + "; ".join(errors))

    candidate_dir.mkdir(parents=True, exist_ok=True)
    legacy_memory = candidate_dir / "research_memory.md"
    if legacy_memory.exists():
        legacy_memory.unlink()
    writers_dir = candidate_dir / "candidate_writers"
    for name in writer_names:
        _atomic_write(writers_dir / f"{name}.md", writer_specs[name])
        _atomic_write(writers_dir / f"{name}.trace.md", "\n".join(writer_traces[name]).rstrip() + "\n")
    judge_path = candidate_dir / "research_spec.judge.md"
    if planning_writers > 1:
        _atomic_write(judge_path, writer_spec)
    elif judge_path.exists():
        judge_path.unlink()
    _atomic_write(candidate_dir / "research_spec.write.md", writer_spec)
    _atomic_write(candidate_dir / "research_spec.critique.md", critique)
    for name, content in documents.items():
        _atomic_write(candidate_dir / name, content)
    trace_title = "Multi-Agent Planning Trace" if planning_writers > 1 else "Single-Writer Planning Trace"
    combined_trace = [f"# {trace_title}", "", f"Model: `{MODEL}`", ""]
    for name in writer_names:
        combined_trace.extend([f"## {name}", "", *writer_traces[name][4:], ""])
    if planning_writers > 1:
        combined_trace.extend(["## Judge", "", *judge_trace[4:], ""])
    combined_trace.extend(["## Critic", "", *critic_trace[4:], ""])
    artifact_lines = [
        "## Artifacts",
        "",
        f"- {planning_writers} independent writer draft(s) saved.",
    ]
    if planning_writers > 1:
        artifact_lines.extend(
            ["- Search-enabled Judge merge saved.", "- Search-enabled Critique saved."]
        )
    else:
        artifact_lines.append("- Tool-free legacy Critique saved; Judge skipped.")
    artifact_lines.extend(["- Revised ResearchSpec saved.", ""])
    combined_trace.extend(artifact_lines)
    _atomic_write(
        candidate_dir / "planning_config.md",
        (
            "# Planning Configuration\n\n"
            f"- planning_writers: `{planning_writers}`\n"
            f"- judge: `{'search-enabled' if planning_writers > 1 else 'skipped'}`\n"
            f"- critic: `{'search-enabled' if planning_writers > 1 else 'tool-free legacy'}`\n"
        ),
    )
    _atomic_write(run_dir / "planning_trace.md", "\n".join(combined_trace).rstrip() + "\n")

    return PlanningResult(
        run_dir=run_dir,
        candidate_dir=candidate_dir,
        research_spec_path=candidate_dir / "research_spec.md",
    )

def generate_candidate(
    question: str,
    *,
    run_root: str | Path = DEFAULT_RUN_ROOT,
    candidate_name: str = "candidate_a",
    max_tool_rounds: int = 10,
    planning_writers: int = 3,
    options: HarnessOptions | None = None,
) -> PlanningResult:
    question = str(question or "").strip()
    if not question:
        raise ValueError("question must not be empty")
    if not re.fullmatch(r"candidate_[a-z0-9_-]+", candidate_name):
        raise ValueError("candidate_name must look like candidate_a")
    if options is None:
        options = HarnessOptions(
            planning_writers=planning_writers,
            judge_enabled=planning_writers > 1,
            critic_search=planning_writers > 1,
        )
    planning_writers = options.planning_writers
    structural_recovery = (
        current_harness_config().planning_acceptance == "structural"
    )
    lossless_acceptance = (
        current_harness_config().planning_acceptance == "lossless"
    )
    guarded_compaction = options.plan_compact == "critic_reviser_guarded"

    writer_prompt = planning_option_prompt(
        PLANNING_SYSTEM_PROMPT,
        "writer",
        compact=options.plan_compact,
        prompt_mode=options.planning_prompt,
    )
    judge_base_prompt = (
        PLANNING_JUDGE_PROMPT
        if options.judge_search
        else PLANNING_TOOL_FREE_JUDGE_PROMPT
    )
    judge_prompt = planning_option_prompt(
        judge_base_prompt,
        "judge",
        compact=options.plan_compact,
        prompt_mode=options.planning_prompt,
    )
    critic_base_prompt = PLANNING_CRITIC_PROMPT
    if not options.critic_search:
        critic_base_prompt = (
            LEGACY_SINGLE_CRITIC_PROMPT
            if planning_writers == 1 and not options.judge_enabled
            else PLANNING_TOOL_FREE_CRITIC_PROMPT
        )
    critic_prompt = planning_option_prompt(
        critic_base_prompt,
        "critic",
        compact=options.plan_compact,
        prompt_mode=options.planning_prompt,
    )
    reviser_base_prompt = _reviser_system_prompt()
    reviser_prompt = planning_option_prompt(
        reviser_base_prompt,
        "reviser",
        compact=options.plan_compact,
        prompt_mode=options.planning_prompt,
    )

    run_id = hashlib.sha256(question.encode("utf-8")).hexdigest()[:16]
    run_dir = Path(run_root).expanduser().resolve() / run_id
    candidate_dir = run_dir / candidate_name
    run_dir.mkdir(parents=True, exist_ok=True)
    _activate_web_fetch_cache(run_dir)
    _atomic_write(run_dir / "question.md", f"# Research Question\n\n{question}\n")
    raw_dir = run_dir / "planning_raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    writer_names = tuple(f"agent_{chr(ord('a') + index)}" for index in range(planning_writers))
    writer_runs: dict[str, tuple[str, list[str]]] = {}
    planning_diagnostics: list[str] = []
    with ThreadPoolExecutor(max_workers=len(writer_names)) as executor:
        futures = {}
        for name in writer_names:
            context = copy_context()
            future = executor.submit(
                context.run,
                _run_planner,
                question,
                max_tool_rounds,
                name,
                search_enabled=options.writer_search,
                system_prompt=writer_prompt,
            )
            futures[future] = name
        for future in as_completed(futures):
            name = futures[future]
            try:
                writer_runs[name] = future.result()
            except Exception as exc:
                if not structural_recovery:
                    raise
                error = f"{type(exc).__name__}: {exc}"
                _record_planning_diagnostics(raw_dir, f"writer-{name}", [error])
                planning_diagnostics.append(
                    f"Writer {name} produced no usable output: {error}"
                )
                continue
            # Persist terminal output before parsing so a later validation
            # exception cannot erase the evidence needed to diagnose it.
            _atomic_write(raw_dir / "writers" / f"{name}.raw.md", writer_runs[name][0])

    writer_specs: dict[str, str] = {}
    writer_traces: dict[str, list[str]] = {}
    for name in writer_names:
        if name not in writer_runs:
            writer_traces[name] = []
            continue
        writer_raw, writer_trace = writer_runs[name]
        writer_traces[name] = writer_trace
        if structural_recovery:
            writer_spec, writer_errors = _candidate_for_next_planning_stage(
                writer_raw, "research_spec.write.md"
            )
            if not writer_spec:
                planning_diagnostics.append(
                    f"Writer {name} returned empty candidate text and was skipped."
                )
                continue
            _record_planning_diagnostics(
                raw_dir, f"writer-{name}", writer_errors
            )
            if writer_errors:
                planning_diagnostics.append(
                    f"Writer {name} continued to Judge with diagnostics: "
                    + "; ".join(writer_errors)
                )
            writer_specs[name] = writer_spec
            continue
        writer_documents, writer_errors = _parse_and_validate_bundle(
            writer_raw, "research_spec.write.md"
        )
        if writer_errors:
            writer_raw, writer_documents, writer_errors = _repair_until_valid(
                question,
                writer_raw,
                writer_errors,
                spec_name="research_spec.write.md",
                system_prompt=writer_prompt,
                raw_dir=raw_dir,
                artifact_prefix=f"writer-{name}",
            )
        if writer_errors:
            raise RuntimeError(
                f"{name} writer draft remains invalid after repair: " + "; ".join(writer_errors)
            )
        writer_specs[name] = writer_documents["research_spec.write.md"]

    available_writer_names = tuple(name for name in writer_names if name in writer_specs)
    if not available_writer_names:
        raise RuntimeError("all planning writers failed to return usable text")
    validated_baseline = next(
        (
            writer_specs[name]
            for name in available_writer_names
            if not _validate_bundle(
                {"research_spec.write.md": writer_specs[name]},
                "research_spec.write.md",
            )
        ),
        "",
    )

    judge_trace: list[str] = []
    judge_succeeded = not options.judge_enabled
    writer_spec = writer_specs[available_writer_names[0]]
    if not options.judge_enabled:
        # This is the pre-multi-writer topology: one search-enabled Writer,
        # followed directly by a tool-free Critic and the Reviser.
        writer_spec = writer_specs[available_writer_names[0]]
    else:
        try:
            judge_raw, judge_trace = _judge_research_specs(
                question,
                writer_specs,
                max_tool_rounds=max_tool_rounds,
                search_enabled=options.judge_search,
                system_prompt=judge_prompt,
            )
            _atomic_write(raw_dir / "judge.raw.md", judge_raw)
            judge_succeeded = True
            if structural_recovery:
                writer_spec, judge_errors = _candidate_for_next_planning_stage(
                    judge_raw, "research_spec.write.md"
                )
                _record_planning_diagnostics(raw_dir, "judge", judge_errors)
                if judge_errors:
                    planning_diagnostics.append(
                        "Judge continued to Critic/Reviser with diagnostics: "
                        + "; ".join(judge_errors)
                    )
                else:
                    validated_baseline = writer_spec
            else:
                judge_documents, judge_errors = _parse_and_validate_bundle(
                    judge_raw, "research_spec.write.md"
                )
                if judge_errors:
                    judge_raw, judge_documents, judge_errors = _repair_until_valid(
                        question,
                        judge_raw,
                        judge_errors,
                        spec_name="research_spec.write.md",
                        system_prompt=judge_prompt,
                        raw_dir=raw_dir,
                        artifact_prefix="judge",
                    )
                if judge_errors:
                    raise RuntimeError(
                        "judge draft remains invalid after repair: "
                        + "; ".join(judge_errors)
                    )
                writer_spec = judge_documents["research_spec.write.md"]
        except Exception as exc:
            if not structural_recovery:
                raise
            error = f"{type(exc).__name__}: {exc}"
            _record_planning_diagnostics(raw_dir, "judge", [error])
            planning_diagnostics.append(
                "Judge produced no usable output; Critic/Reviser received the first "
                f"available Writer candidate: {error}"
            )

    critique = ""
    critic_trace: list[str] = []
    critic_succeeded = not options.critic_enabled
    if options.critic_enabled:
        try:
            critique, critic_trace = _criticize_research_spec(
                question,
                writer_spec,
                max_tool_rounds=max_tool_rounds,
                search_enabled=options.critic_search,
                system_prompt=critic_prompt,
                legacy_single_writer=(planning_writers == 1 and not options.judge_enabled),
            )
            _atomic_write(raw_dir / "critic.raw.md", critique)
            critic_succeeded = True
        except Exception as exc:
            if not structural_recovery:
                raise
            error = f"{type(exc).__name__}: {exc}"
            _record_planning_diagnostics(raw_dir, "critic", [error])
            planning_diagnostics.append(
                "Critic produced no usable output; Reviser continued with the "
                f"upstream candidate: {error}"
            )
    if options.reviser_enabled:
        try:
            final_raw = _revise_research_spec(
                question,
                writer_spec,
                critique,
                system_prompt=reviser_prompt,
                max_tokens=options.reviser_max_tokens,
            )
            _atomic_write(raw_dir / "reviser.raw.md", final_raw)
        except Exception as exc:
            if not structural_recovery:
                raise
            error = f"{type(exc).__name__}: {exc}"
            _record_planning_diagnostics(raw_dir, "reviser", [error])
            planning_diagnostics.append(
                "Reviser produced no usable output; final Repair received the "
                f"upstream candidate: {error}"
            )
            final_raw = _compile_planning_submission(
                writer_spec, "research_spec.md"
            )
    else:
        final_raw = _compile_planning_submission(writer_spec, "research_spec.md")
        _atomic_write(raw_dir / "review_passthrough.raw.md", final_raw)
    documents, errors = _parse_and_validate_bundle(final_raw, "research_spec.md")
    if not errors and options.reviser_enabled:
        errors.extend(_high_priority_lead_errors(critique, documents["research_spec.md"]))
        if lossless_acceptance or guarded_compaction:
            errors.extend(
                _planning_semantic_regression_errors(
                    writer_spec, documents["research_spec.md"]
                )
            )
    if (
        not errors
        and options.reviser_enabled
        and options.critic_enabled
        and options.planning_revision_rounds == 2
    ):
        round_one_spec = documents["research_spec.md"]
        try:
            critique_two, critic_trace_two = _criticize_research_spec(
                question,
                round_one_spec,
                max_tool_rounds=max_tool_rounds,
                search_enabled=options.critic_search,
                system_prompt=critic_prompt,
            )
            _atomic_write(raw_dir / "critic.round2.raw.md", critique_two)
            critic_trace.extend(["", "## Round 2", "", *critic_trace_two[4:]])
            unresolved = _high_priority_lead_errors(critique_two, round_one_spec)
            if unresolved:
                final_raw = _revise_research_spec(
                    question,
                    round_one_spec,
                    critique_two,
                    system_prompt=reviser_prompt,
                    max_tokens=options.reviser_max_tokens,
                )
                critique = critique_two
                _atomic_write(raw_dir / "reviser.round2.raw.md", final_raw)
                documents, errors = _parse_and_validate_bundle(
                    final_raw, "research_spec.md"
                )
                if not errors:
                    errors.extend(
                        _high_priority_lead_errors(
                            critique_two, documents["research_spec.md"]
                        )
                    )
                    if lossless_acceptance or guarded_compaction:
                        errors.extend(
                            _planning_semantic_regression_errors(
                                writer_spec, documents["research_spec.md"]
                            )
                        )
        except Exception as exc:
            if not structural_recovery:
                raise
            error = f"{type(exc).__name__}: {exc}"
            _record_planning_diagnostics(raw_dir, "revision-round2", [error])
            planning_diagnostics.append(
                "Second Critic/Reviser round failed; final Repair retained the "
                f"validated round-one output: {error}"
            )
            final_raw = _compile_planning_submission(
                round_one_spec, "research_spec.md"
            )
            documents = {"research_spec.md": round_one_spec}
            errors = []
    if errors and options.reviser_enabled and options.planning_repair == "full":
        try:
            final_raw, documents, errors = _repair_until_valid(
                question,
                final_raw,
                errors,
                spec_name="research_spec.md",
                system_prompt=reviser_prompt,
                raw_dir=raw_dir,
                artifact_prefix="reviser",
                required_context=(
                    "The Critic findings below are binding. Every HP ID must appear either "
                    "where absorbed in an H3 or under `### Rejected High-Priority Leads` "
                    "with a concrete reason.\n\n# Critic Findings\n\n" + critique
                ),
                required_lead_critique=critique,
                lossless_source=(
                    writer_spec if lossless_acceptance or guarded_compaction else ""
                ),
            )
        except Exception as exc:
            if not structural_recovery:
                raise
            errors = [f"{type(exc).__name__}: {exc}"]
            _record_planning_diagnostics(raw_dir, "reviser-repair", errors)
            planning_diagnostics.append(
                "Final Repair failed: " + errors[0]
            )
    if errors and options.reviser_enabled and guarded_compaction:
        # Guarded compaction is fail-closed: an invalid or lossy merge cannot
        # replace the already validated Judge/Writer baseline. This preserves
        # coverage without adding another model call or changing other arms.
        fallback_raw = _compile_planning_submission(writer_spec, "research_spec.md")
        fallback_documents, fallback_errors = _parse_and_validate_bundle(
            fallback_raw, "research_spec.md"
        )
        if not fallback_errors:
            _atomic_write(raw_dir / "reviser.coverage-fallback.md", fallback_raw)
            _atomic_write(
                raw_dir / "reviser.coverage-fallback.txt",
                "Rejected guarded compaction and preserved the validated Judge/Writer baseline:\n"
                + "\n".join(f"- {error}" for error in errors)
                + "\n",
            )
            final_raw, documents, errors = fallback_raw, fallback_documents, []
    if errors and options.reviser_enabled and lossless_acceptance and options.planning_repair == "full":
        # The Judge/Writer baseline was already validated. Prefer it over a
        # structurally valid but semantically lossy Reviser result.
        fallback_raw = _compile_planning_submission(writer_spec, "research_spec.md")
        fallback_documents, fallback_errors = _parse_and_validate_bundle(
            fallback_raw, "research_spec.md"
        )
        if fallback_errors:
            raise RuntimeError(
                "revised planning bundle remains invalid after repair: "
                + "; ".join(errors)
            )
        _atomic_write(raw_dir / "reviser.lossless-fallback.md", fallback_raw)
        _atomic_write(
            raw_dir / "reviser.lossless-fallback.txt",
            "Rejected the Reviser output and preserved the validated Judge/Writer baseline:\n"
            + "\n".join(f"- {error}" for error in errors)
            + "\n",
        )
        final_raw, documents, errors = fallback_raw, fallback_documents, []
    if errors and structural_recovery and validated_baseline:
        fallback_raw = _compile_planning_submission(
            validated_baseline, "research_spec.md"
        )
        fallback_documents, fallback_errors = _parse_and_validate_bundle(
            fallback_raw, "research_spec.md"
        )
        if not fallback_errors:
            _atomic_write(raw_dir / "planning.validated-fallback.md", fallback_raw)
            _atomic_write(
                raw_dir / "planning.validated-fallback.txt",
                "All downstream repair opportunities were exhausted; preserved the "
                "latest validated upstream ResearchSpec:\n"
                + "\n".join(f"- {error}" for error in errors)
                + "\n",
            )
            planning_diagnostics.append(
                "Final candidate remained invalid after Repair; used the latest "
                "validated upstream ResearchSpec."
            )
            final_raw, documents, errors = fallback_raw, fallback_documents, []
    if errors:
        raise RuntimeError(
            "planning passthrough remains invalid: " + "; ".join(errors)
        )

    candidate_dir.mkdir(parents=True, exist_ok=True)
    legacy_memory = candidate_dir / "research_memory.md"
    if legacy_memory.exists():
        legacy_memory.unlink()
    writers_dir = candidate_dir / "candidate_writers"
    for name in writer_names:
        if name not in writer_specs:
            continue
        _atomic_write(writers_dir / f"{name}.md", writer_specs[name])
        _atomic_write(writers_dir / f"{name}.trace.md", "\n".join(writer_traces[name]).rstrip() + "\n")
    judge_path = candidate_dir / "research_spec.judge.md"
    if options.judge_enabled and judge_succeeded:
        _atomic_write(judge_path, writer_spec)
    elif judge_path.exists():
        judge_path.unlink()
    _atomic_write(candidate_dir / "research_spec.write.md", writer_spec)
    critique_path = candidate_dir / "research_spec.critique.md"
    if options.critic_enabled and critic_succeeded:
        _atomic_write(critique_path, critique)
    elif critique_path.exists():
        critique_path.unlink()
    for name, content in documents.items():
        _atomic_write(candidate_dir / name, content)
    trace_title = "Multi-Agent Planning Trace" if planning_writers > 1 else "Single-Writer Planning Trace"
    combined_trace = [f"# {trace_title}", "", f"Model: `{MODEL}`", ""]
    for name in writer_names:
        trace = writer_traces.get(name) or []
        combined_trace.extend([f"## {name}", "", *trace[4:], ""])
    if options.judge_enabled and judge_succeeded:
        combined_trace.extend(["## Judge", "", *judge_trace[4:], ""])
    if options.critic_enabled and critic_succeeded:
        combined_trace.extend(["## Critic", "", *critic_trace[4:], ""])
    if planning_diagnostics:
        combined_trace.extend(
            [
                "## Planning Diagnostics",
                "",
                *[f"- {item}" for item in planning_diagnostics],
                "",
            ]
        )
    artifact_lines = [
        "## Artifacts",
        "",
        f"- {len(writer_specs)}/{planning_writers} writer draft(s) remained usable.",
    ]
    artifact_lines.append(
        f"- Judge: {'search-enabled' if options.judge_search else 'tool-free'} merge saved."
        if options.judge_enabled and judge_succeeded
        else "- Judge failed; downstream stages received a Writer candidate."
        if options.judge_enabled
        else "- Judge skipped."
    )
    artifact_lines.append(
        f"- Critic: {'search-enabled' if options.critic_search else 'tool-free'} critique saved."
        if options.critic_enabled and critic_succeeded
        else "- Critic failed; Reviser continued without a Critic response."
        if options.critic_enabled
        else "- Critic skipped."
    )
    artifact_lines.extend(
        [
            "- Revised ResearchSpec saved."
            if options.reviser_enabled
            else "- Validated upstream ResearchSpec passed through without Reviser.",
            "",
        ]
    )
    combined_trace.extend(artifact_lines)
    _atomic_write(
        candidate_dir / "planning_config.md",
        (
            "# Planning Configuration\n\n"
            f"- planning_writers: `{planning_writers}`\n"
            f"- writer_search: `{options.writer_search}`\n"
            f"- judge: `{'skipped' if not options.judge_enabled else 'search-enabled' if options.judge_search else 'tool-free'}`\n"
            f"- critic: `{'skipped' if not options.critic_enabled else 'search-enabled' if options.critic_search else 'tool-free'}`\n"
            f"- reviser: `{'enabled' if options.reviser_enabled else 'skipped'}`\n"
            f"- plan_compact: `{options.plan_compact}`\n"
            f"- planning_prompt: `{options.planning_prompt}`\n"
        ),
    )
    _atomic_write(run_dir / "planning_trace.md", "\n".join(combined_trace).rstrip() + "\n")

    return PlanningResult(
        run_dir=run_dir,
        candidate_dir=candidate_dir,
        research_spec_path=candidate_dir / "research_spec.md",
    )
