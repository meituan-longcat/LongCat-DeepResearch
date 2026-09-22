from __future__ import annotations

import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextvars import copy_context
from typing import Any

from . import execution_context as run_context
from . import telemetry

from .models import SectionSpec
from .planning import _split_research_spec
from .policy import current_harness_config
from .prompts import RESEARCHER_SYSTEM_PROMPT, SECTION_OUTPUT_RE
from .repair import apply_exact_patches
from .runtime import (
    MODEL,
    _controller_terminal,
    _assistant_message,
    _chat,
    _extract_terminal_text,
    _run_tool_call,  # noqa: F401 - retained as a documented monkeypatch seam
    _run_scoped_tool_call,
    _uses_auto_markdown_recovery,
)

def _compile_section_submission(body: str, expected: SectionSpec) -> str:
    """Compile body-only model output into one exact assigned section."""
    text = str(body or "").strip()
    wrapped = SECTION_OUTPUT_RE.search(text)
    if wrapped:
        text = wrapped.group(1).strip()
    if text.startswith("```") and text.endswith("```"):
        text = re.sub(r"^```(?:markdown|md)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text).strip()
    text = re.sub(
        r"(?mi)^\s*<!--\s*(?:SECTION|END\s+SECTION|章节|小节|结束章节)\s*-->\s*$",
        "",
        text,
    ).strip()

    assigned = re.compile(
        rf"^#{{1,6}}\s+{re.escape(expected.section_id)}\s*[｜|:]\s*"
        rf"{re.escape(expected.title)}\s*$",
        re.MULTILINE,
    )
    text = assigned.sub("", text, count=1).strip()
    # A model may organize the body with extra H1-H3 headings. They are useful
    # prose structure but cannot become sibling report sections, so demote them
    # deterministically below the assigned H3 boundary.
    text = re.sub(r"^#{1,3}\s+", "#### ", text, flags=re.MULTILINE)
    hashes = "#" * expected.heading_level
    heading = f"{hashes} {expected.section_id}｜{expected.title}"
    return f"<!-- SECTION -->\n{heading}\n{text}\n<!-- END SECTION -->\n"


def _recover_section_submission(body: str, expected: SectionSpec) -> str | None:
    """Recover transport syntax without asking the model to resample content."""
    recovered = _compile_section_submission(body, expected)
    wrapped = SECTION_OUTPUT_RE.search(recovered)
    if not wrapped:
        return None
    section = wrapped.group(1).strip()
    hashes = "#" * expected.heading_level
    prose = re.sub(
        rf"^{re.escape(hashes)}\s+{re.escape(expected.section_id)}\s*[｜|:]\s*"
        rf"{re.escape(expected.title)}\s*$",
        "",
        section,
        count=1,
        flags=re.MULTILINE,
    ).strip()
    if len(prose) < 200:
        return None
    if re.fullmatch(
        r"(?is)(?:let me|i will|below i(?:'ll| will)|下面(?:我|将)|现在(?:我|将)).{0,240}",
        prose,
    ):
        return None
    return recovered


def _h2_blocks(markdown: str, prefix: str) -> list[tuple[str, str, str]]:
    pattern = re.compile(rf"^##\s+({re.escape(prefix)}\d+)\s*[｜|:]\s*(.+?)\s*$", re.MULTILINE)
    matches = list(pattern.finditer(markdown))
    blocks: list[tuple[str, str, str]] = []
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(markdown)
        blocks.append((match.group(1), match.group(2).strip(), markdown[match.start() : end].strip() + "\n"))
    return blocks


def _subsection_blocks(markdown: str) -> list[SectionSpec]:
    main_pattern = re.compile(r"^##\s+(S\d+)\s*[｜|:]\s*(.+?)\s*$", re.MULTILINE)
    sub_pattern = re.compile(r"^###\s+(S\d+\.\d+)\s*[｜|:]\s*(.+?)\s*$", re.MULTILINE)
    mains = list(main_pattern.finditer(markdown))
    subs = list(sub_pattern.finditer(markdown))
    results: list[SectionSpec] = []
    for index, match in enumerate(subs):
        parent_id = match.group(1).split(".", 1)[0]
        parent = next((item for item in reversed(mains) if item.start() < match.start()), None)
        if parent is None or parent.group(1) != parent_id:
            continue
        end = subs[index + 1].start() if index + 1 < len(subs) else len(markdown)
        # Do not absorb the next H2 main heading into the previous subsection.
        next_main = next((item for item in mains if match.start() < item.start() < end), None)
        if next_main is not None:
            end = next_main.start()
        results.append(
            SectionSpec(
                match.group(1),
                match.group(2).strip(),
                markdown[match.start() : end].strip() + "\n",
                parent_id=parent_id,
                parent_title=parent.group(2).strip(),
                heading_level=3,
            )
        )
    return results


def _parse_sections(research_spec: str) -> tuple[str, list[SectionSpec], dict[str, str]]:
    report_spec, rowbook = _split_research_spec(research_spec)
    sections = _subsection_blocks(report_spec)
    if not sections:  # backward compatibility with the first Markdown prototype
        sections = [
            SectionSpec(section_id, title, block)
            for section_id, title, block in _h2_blocks(report_spec, "S")
        ]
    facts = {fact_id: block for fact_id, _title, block in _h2_blocks(rowbook, "F")}
    return report_spec, sections, facts


def _relevant_rowbook(section_id: str, facts: dict[str, str]) -> str:
    if not facts:
        return ""
    selected = [block for block in facts.values() if re.search(rf"\b{re.escape(section_id)}\b", block)]
    return "# Relevant Rowbook\n\n" + "\n".join(selected).strip() + "\n"


def project_research_context(report_spec: str, mode: str = "full") -> str:
    """Keep global constraints and the heading map while hiding sibling details."""
    if mode == "full":
        return report_spec
    if mode != "projected":
        raise ValueError(f"invalid research context mode: {mode}")
    first_numbered_h2 = re.search(r"^##\s+S\d+\b.*$", report_spec, re.MULTILINE)
    if first_numbered_h2 is None:
        return report_spec
    global_contract = report_spec[: first_numbered_h2.start()].strip()
    heading_matches = list(
        re.finditer(
            r"^(?P<level>##|###)\s+S\d+(?:\.\d+)?\s*[｜|:].*$",
            report_spec,
            re.MULTILINE,
        )
    )
    if not heading_matches:
        return report_spec
    outline_parts: list[str] = []
    for index, match in enumerate(heading_matches):
        outline_parts.append(match.group(0).rstrip())
        if match.group("level") == "##":
            end = (
                heading_matches[index + 1].start()
                if index + 1 < len(heading_matches)
                else len(report_spec)
            )
            parent_contract = report_spec[match.end() : end].strip()
            if parent_contract:
                outline_parts.append(parent_contract)
    outline = "\n".join(outline_parts)
    return (
        f"{global_contract}\n\n# Complete Report Heading Outline\n\n{outline}\n\n"
        "The assigned subsection below contains its complete What to Cover, "
        "Research Questions, Required Entities / Cases, and Source Leads contract."
    )


def researcher_system_prompt(context_mode: str = "full") -> str:
    if context_mode == "full":
        return RESEARCHER_SYSTEM_PROMPT
    if context_mode != "projected":
        raise ValueError(f"invalid research context mode: {context_mode}")
    return RESEARCHER_SYSTEM_PROMPT.replace(
        "the complete ReportSpec, and your assigned subsection",
        "the complete global constraints and report heading outline, and your assigned subsection",
    ).replace(
        "Read the complete ReportSpec so your section fits the whole report.",
        "Read all global constraints and the complete heading outline so your section fits the whole report.",
    )


def _run_tool_agent(
    system_prompt: str,
    user_prompt: str,
    *,
    max_tool_rounds: int,
    label: str,
    max_tool_calls: int = 20,
    call_type: str,
    actor: str,
    trajectory_id: str,
) -> tuple[str, list[str], list[dict[str, Any]]]:
    config = current_harness_config()
    output_max_tokens = config.subsection_researcher_max_tokens
    terminal_max_tokens = config.subsection_researcher_terminal_max_tokens
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    trace = [f"# {label} Trace", "", f"Model: `{MODEL}`", ""]
    tool_call_count = 0
    auto = _uses_auto_markdown_recovery()
    def force_terminal(
        turn_index: int, reason: str
    ) -> tuple[str, list[str], list[dict[str, Any]]]:
        instruction = "Return the required terminal Markdown now."
        terminal, message, attempt = _controller_terminal(
                messages,
                final_instruction=instruction,
                call_type=call_type,
                actor=actor,
                trajectory_id=trajectory_id,
                turn_index=turn_index,
                reason=reason,
                chat_fn=_chat,
                max_tokens=terminal_max_tokens,
        )
        messages.append(_assistant_message(message))
        trace.append(
            f"Forced terminal answer after {tool_call_count} tool calls"
            f"{' with one identical protocol retry' if attempt else ''}."
        )
        return terminal, trace, messages

    for round_index in range(1, max_tool_rounds + 1):
        if tool_call_count >= max_tool_calls or (not auto and round_index == max_tool_rounds):
            return force_terminal(round_index, "tool_budget_exhausted")
        with run_context.scope(
            call_type=call_type,
            actor=actor,
            trajectory_id=trajectory_id,
            turn_index=round_index,
        ):
            message = _chat(messages, tools=True, max_tokens=output_max_tokens)
        calls = message.get("tool_calls") or []
        if calls:
            print(f"[{label}] tool round {round_index}: {len(calls)} call(s)", file=sys.stderr, flush=True)
            trace.append(f"## Tool round {round_index}")
            if len(calls) > max_tool_calls:
                with telemetry.scope(
                    call_type=call_type,
                    actor=actor,
                    trajectory_id=trajectory_id,
                    turn_index=round_index,
                ):
                    telemetry.record_tool_batch(calls, allowed_call_ids=set())
                trace.extend(
                    [
                        f"- Controller rejected an abnormal {len(calls)}-call action; "
                        "no tool was executed and no synthetic observation was added.",
                        "",
                    ]
                )
                return force_terminal(round_index + 1, "abnormal_tool_action")
            allowed_calls = list(calls)
            rejected_calls: list[dict[str, Any]] = []
            messages.append(_assistant_message(message))
            tool_call_count += len(allowed_calls)
            results: dict[str, str] = {
                str(call.get("id") or "missing-tool-id"): "Tool budget exhausted; finalize from collected evidence."
                for call in rejected_calls
            }
            with ThreadPoolExecutor(max_workers=max(1, min(4, len(allowed_calls)))) as executor:
                execution_fields = run_context.context_fields()
                futures = {}
                for call in allowed_calls:
                    context = copy_context()
                    future = executor.submit(
                        context.run,
                        _run_scoped_tool_call,
                        call,
                        call_type=call_type,
                        actor=actor,
                        trajectory_id=trajectory_id,
                        turn_index=round_index,
                        execution_fields=execution_fields,
                    )
                    futures[future] = call
                for future in as_completed(futures):
                    call_id, result = future.result()
                    results[call_id] = result
            with telemetry.scope(
                call_type=call_type,
                actor=actor,
                trajectory_id=trajectory_id,
                turn_index=round_index,
            ):
                telemetry.record_tool_batch(
                    calls,
                    allowed_call_ids={
                        str(call.get("id") or "missing-tool-id")
                        for call in allowed_calls
                    },
                )
            for call in calls:
                call_id = str(call.get("id") or "missing-tool-id")
                name = str((call.get("function") or {}).get("name") or "unknown")
                trace.append(f"- `{name}`: `{call_id}`")
                messages.append({"role": "tool", "tool_call_id": call_id, "content": results[call_id]})
            trace.append("")
            if tool_call_count >= max_tool_calls:
                return force_terminal(round_index + 1, "tool_budget_soft_threshold")
            continue
        terminal = _extract_terminal_text(message)
        if terminal:
            messages.append(_assistant_message(message))
            return terminal, trace, messages
        messages.append(_assistant_message(message))
        messages.append({"role": "user", "content": "Return the required terminal Markdown now."})
    if auto:
        return force_terminal(max_tool_rounds + 1, "round_limit")
    raise RuntimeError(f"{label} did not finish within {max_tool_rounds} tool rounds")


def _parse_section_output(raw: str, expected: SectionSpec, *, allow_unwrapped: bool = False) -> str:
    section_match = SECTION_OUTPUT_RE.search(raw)
    if section_match:
        section = section_match.group(1).strip() + "\n"
    elif allow_unwrapped:
        # A refiner occasionally returns the exact assigned heading without the
        # requested HTML comments. The ID/title checks below still make this a
        # safe, deterministic local replacement rather than a permissive parse.
        section = raw.strip() + "\n"
    else:
        raise ValueError(f"{expected.section_id} output is missing SECTION markers")
    hashes = "#" * expected.heading_level
    headings = re.findall(rf"^{hashes}\s+(.+?)\s*$", section, re.MULTILINE)
    if len(headings) != 1 or not re.match(rf"{re.escape(expected.section_id)}\s*[｜|:]", headings[0]):
        raise ValueError(f"{expected.section_id} output must contain exactly its assigned H{expected.heading_level}")
    returned_title = re.sub(rf"^{re.escape(expected.section_id)}\s*[｜|:]\s*", "", headings[0]).strip()
    if returned_title != expected.title:
        raise ValueError(f"{expected.section_id} changed its assigned title")
    return section


def _repair_researcher_output(
    messages: list[dict[str, Any]],
    raw: str,
    expected: SectionSpec,
    error: ValueError,
) -> str:
    repair_messages = list(messages)
    repair_messages.append(
        {
            "role": "user",
            "content": f"""Your terminal subsection failed deterministic validation:
- {error}

Return only one smallest exact-match patch transaction. Do not rewrite or repeat
the complete section. Every `old` anchor must be copied verbatim from the malformed
output and occur exactly once. Preserve all text outside the patch.

Return exactly:
<!-- PATCHES -->
{{"patches":[{{"id":"P1","old":"exact original text","new":"replacement text","expected_count":1}}]}}
<!-- END PATCHES -->

Malformed terminal output:
{raw}""",
        }
    )
    with run_context.scope(
        call_type="subsection_researcher_repair_patch",
        actor=f"research-{expected.section_id}.repair-1",
        trajectory_id=f"subsection-researcher-{expected.section_id}",
        turn_index=1,
        degraded=True,
    ):
        message = _chat(repair_messages, tools=False, max_tokens=8000)
    patch = _extract_terminal_text(message)
    if not patch or message.get("tool_calls"):
        raise RuntimeError(f"{expected.section_id} repair patch returned invalid content")
    return apply_exact_patches(raw, patch)


def build_researcher_prompt(
    question: str,
    report_spec: str,
    section: SectionSpec,
    relevant_rowbook: str,
) -> str:
    heading = "#" * section.heading_level + f" {section.section_id}｜{section.title}"
    return f"""# Original Question

{question}

# Complete ReportSpec

{report_spec}

# Assigned Section

{section.markdown}

{relevant_rowbook}

# Required Terminal Format

<!-- SECTION -->
{heading}
Complete assigned report unit Markdown with citations.
<!-- END SECTION -->
"""


def _research_one_section(
    question: str,
    report_spec: str,
    section: SectionSpec,
    relevant_rowbook: str,
    *,
    max_tool_rounds: int,
    context_mode: str = "full",
    attempt: int = 1,
) -> tuple[str, list[str]]:
    retry_suffix = "" if attempt == 1 else f".retry-{attempt - 1}"
    actor = f"research-{section.section_id}{retry_suffix}"
    trajectory_id = f"subsection-researcher-{section.section_id}{retry_suffix}"
    user = build_researcher_prompt(
        question,
        project_research_context(report_spec, context_mode),
        section,
        relevant_rowbook,
    )
    raw, trace, messages = _run_tool_agent(
        researcher_system_prompt(context_mode),
        user,
        max_tool_rounds=max_tool_rounds,
        label=actor,
        call_type="subsection_researcher_turn",
        actor=actor,
        trajectory_id=trajectory_id,
    )
    try:
        return _parse_section_output(raw, section), trace
    except ValueError as error:
        if _uses_auto_markdown_recovery():
            recovered = _recover_section_submission(raw, section)
            if recovered is not None:
                trace.append("Section transport recovered deterministically; body was not resampled.")
                return _parse_section_output(recovered, section), trace
        compact = str(raw or "").strip()
        local_patch_candidate = (
            len(compact) >= 500
            and (
                section.section_id in compact
                or section.title in compact
                or re.search(r"(?m)^#{2,4}\s+", compact) is not None
            )
        )
        if not local_patch_candidate:
            raise RuntimeError(
                f"{section.section_id} requires section regeneration after non-local "
                f"terminal failure: {error}"
            ) from error
        repaired = _repair_researcher_output(messages, raw, section, error)
        trace.append("Researcher terminal repair attempted once.")
        return _parse_section_output(repaired, section), trace


def _report_title(question: str) -> str:
    compact = re.sub(r"\s+", " ", question).strip()
    sentences = re.split(r"(?<=[.!?。！？])\s+", compact)
    title = ""
    request_prefix = re.compile(
        r"^(?:please\s+)?(?:help\s+me\s+)?(?:investigate|research|analy[sz]e)\s+",
        re.IGNORECASE,
    )
    for sentence in sentences:
        if request_prefix.match(sentence):
            title = request_prefix.sub("", sentence).strip(" .!?。！？")
            break
    if not title and sentences:
        title = sentences[0].strip(" .!?。！？")
    if title:
        title = title[:1].upper() + title[1:]
    if len(title) > 180:
        title = title[:177].rstrip() + "..."
    return title or "Deep Research Report"


def _assemble(question: str, sections: list[SectionSpec], bodies: dict[str, str], *, strip_ids: bool = False) -> str:
    # The original question is an execution contract, not report prose. Using
    # it verbatim leaked formatting instructions and forbidden URLs into the
    # report preamble. Keep only a compact topic sentence as the H1.
    parts = [f"# {_report_title(question)}", ""]
    current_parent = ""
    for section in sections:
        if section.heading_level == 3 and section.parent_id != current_parent:
            parent_heading = f"## {section.parent_id}｜{section.parent_title}"
            if strip_ids:
                parent_heading = f"## {section.parent_title}"
            parts.extend([parent_heading, ""])
            current_parent = section.parent_id
        body = bodies[section.section_id].strip()
        if strip_ids:
            hashes = "#" * section.heading_level
            body = re.sub(
                rf"^{hashes}\s+{re.escape(section.section_id)}\s*[｜|:]\s*",
                f"{hashes} ",
                body,
                count=1,
                flags=re.MULTILINE,
            )
        parts.extend([body, ""])
    return "\n".join(parts).rstrip() + "\n"
