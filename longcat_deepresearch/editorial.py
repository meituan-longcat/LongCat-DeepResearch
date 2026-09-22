from __future__ import annotations

import difflib
import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextvars import copy_context
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from . import execution_context as run_context
from .integrity import (
    _citation_attribution_key,
    _citation_attribution_keys,
    _citation_attribution_literals,
    _authorized_table_removal_text,
    _delete_lines,
    _deletion_authorizes,
    _entity_labels,
    _http_literals,
    _lossless_normalize,
    _normalized_text,
    _numeric_fact_literals,
    _patch_regression_errors,
    _standard_patch_regression_errors,
)
from .models import SectionSpec
from .policy import current_harness_config
from .prompts import (
    EDIT_PLAN_RE,
    EDITORIAL_PLANNER_SYSTEM_PROMPT,
    LOSSLESS_EDITORIAL_PLANNER_SYSTEM_PROMPT,
    LOCAL_EDITOR_SYSTEM_PROMPT,
    LOCAL_PATCH_EDITOR_SYSTEM_PROMPT,
    PATCH_OUTPUT_RE,
)
from .research import (
    _parse_section_output,
    _recover_section_submission,
    _relevant_rowbook,
    _subsection_blocks,
)
from .runtime import (
    ResearchAgent,
    _append_tool_free_retry,
    _atomic_write,
    _chat,
    _extract_terminal_text,
    _uses_auto_markdown_recovery,
)

@dataclass(frozen=True)
class MarkdownPatchBlock:
    block_id: str
    text: str
    start: int
    end: int
    separator: str


@dataclass(frozen=True)
class SectionPatchTransaction:
    transaction_id: str
    operation: str
    targets: tuple[str, ...]
    content: str = ""


def _parse_section_directives(markdown: str) -> dict[str, str]:
    section_heading = re.compile(
        r"^##\s+(S\d+(?:\.\d+)?)(?:\s*[｜|:].*)?\s*$", re.MULTILINE
    )
    matches = list(section_heading.finditer(markdown))
    notes: dict[str, str] = {}
    for index, item in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(markdown)
        notes[item.group(1)] = markdown[item.start() : end].strip() + "\n"
    return notes



def _plan_editorial_changes(
    question: str, research_spec: str, draft: str
) -> tuple[str, dict[str, str]]:
    planner_prompt = (
        LOSSLESS_EDITORIAL_PLANNER_SYSTEM_PROMPT
        if current_harness_config().integrity_policy == "lossless"
        else EDITORIAL_PLANNER_SYSTEM_PROMPT
    )
    messages = [
        {"role": "system", "content": planner_prompt},
        {
            "role": "user",
            "content": (
                f"# Original Question\n\n{question}\n\n"
                f"{research_spec}\n\n# Assembled Draft\n\n"
                f"Draft characters: {len(draft):,}\n\n{draft}"
            ),
        },
    ]
    raw = ""
    budgets = (
        (32000, 64000)
        if current_harness_config().revision_protocol == "v2"
        else (12000, 64000)
    )
    for attempt, budget in enumerate(budgets):
        with run_context.scope(
            call_type="global_editorial_plan",
            actor="global_editor",
            trajectory_id="global-editorial-plan",
            turn_index=1 + attempt,
            terminal_retry=bool(attempt),
        ):
            message = _chat(messages, tools=False, max_tokens=budget)
        raw = _extract_terminal_text(message)
        if raw and not message.get("tool_calls"):
            break
        if not attempt:
            _append_tool_free_retry(
                messages,
                message,
                "Tools are unavailable. Return only the complete Global Editorial Plan "
                "Markdown now; do not explain your reasoning.",
            )
    if not raw or message.get("tool_calls"):
        # Editing is optional and local editors already pass through rejected
        # patches. If the planner cannot produce a visible plan after one
        # bounded retry, preserve the assembled draft byte-for-byte.
        return (
            "# Global Editorial Plan\n\n"
            "No safe editorial directives were produced; preserve every section.\n",
            {},
        )
    match = EDIT_PLAN_RE.search(raw)
    plan = (match.group(1) if match else raw).strip() + "\n"
    notes = {
        section_id: note
        for section_id, note in _parse_section_directives(plan).items()
        if re.search(
            r"(?mi)^\s*[-*]\s*(?:DELETE|MERGE|MOVE|CROSS-REFERENCE)\s*:", note
        )
    }
    return plan, notes


def _edit_one_section(
    question: str,
    report_spec: str,
    full_draft: str,
    editorial_plan: str,
    section: SectionSpec,
    current_body: str,
    relevant_rowbook: str,
    local_directives: str,
    *,
    max_tokens: int = 32000,
) -> str:
    heading = "#" * section.heading_level + f" {section.section_id}｜{section.title}"
    user = f"""# Original Question

{question}

# Complete ReportSpec

{report_spec}

# Complete Assembled Draft

{full_draft}

# Global Editorial Plan

{editorial_plan}

# Assigned Section To Patch

{current_body}

{relevant_rowbook}

# Local Directives For This Section

{local_directives}

# Required Terminal Format

<!-- SECTION -->
{heading}
Complete revised assigned report unit Markdown with citations.
<!-- END SECTION -->
"""
    body = ResearchAgent(
        system_prompt=LOCAL_EDITOR_SYSTEM_PROMPT,
        label=f"editor-{section.section_id}",
        call_type="local_editor",
        actor=f"editor-{section.section_id}",
        trajectory_id=f"local-editor-{section.section_id}",
    ).complete(
        user,
        max_tokens=max_tokens,
        empty_error=f"editor-{section.section_id} returned empty content",
    )
    return body


def _section_patch_blocks(
    markdown: str, expected: SectionSpec
) -> list[MarkdownPatchBlock]:
    """Split an assigned section into addressable paragraph/table blocks."""
    hashes = "#" * expected.heading_level
    heading = re.search(
        rf"^{re.escape(hashes)}\s+{re.escape(expected.section_id)}\s*[｜|:]\s*"
        rf"{re.escape(expected.title)}\s*$",
        markdown,
        re.MULTILINE,
    )
    if not heading:
        raise ValueError(f"{expected.section_id} assigned heading was not found")
    body_start = heading.end()
    body = markdown[body_start:]
    blocks: list[MarkdownPatchBlock] = []
    pattern = re.compile(r"(?ms)(?P<text>\S(?:.*?\S)?)(?P<sep>\n{2,}|\n*\Z)")
    for index, match in enumerate(pattern.finditer(body), 1):
        text = match.group("text")
        if not text.strip():
            continue
        blocks.append(
            MarkdownPatchBlock(
                block_id=f"B{index:04d}",
                text=text,
                start=body_start + match.start("text"),
                end=body_start + match.end(),
                separator=match.group("sep"),
            )
        )
    return blocks


def _format_patch_blocks(blocks: list[MarkdownPatchBlock]) -> str:
    return "\n\n".join(
        f"[{block.block_id}]\n{block.text}\n[/{block.block_id}]" for block in blocks
    )


def _parse_patch_transactions(
    raw: str, blocks: list[MarkdownPatchBlock]
) -> list[SectionPatchTransaction]:
    match = PATCH_OUTPUT_RE.search(str(raw or ""))
    if not match:
        raise ValueError("patch output is missing PATCHES markers")
    try:
        payload = json.loads(match.group(1).strip())
    except json.JSONDecodeError as exc:
        raise ValueError(f"patch output is not valid JSON: {exc}") from exc
    rows = payload.get("transactions") if isinstance(payload, dict) else None
    if not isinstance(rows, list):
        raise ValueError("patch output must contain a transactions list")
    if len(rows) > 12:
        raise ValueError("patch output contains more than 12 transactions")
    known = {block.block_id for block in blocks}
    order = {block.block_id: index for index, block in enumerate(blocks)}
    seen_ids: set[str] = set()
    transactions: list[SectionPatchTransaction] = []
    for index, row in enumerate(rows, 1):
        if not isinstance(row, dict):
            raise ValueError(f"transaction {index} is not an object")
        transaction_id = str(row.get("id") or f"T{index}").strip()
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,32}", transaction_id):
            raise ValueError(f"transaction {index} has an invalid id")
        if transaction_id in seen_ids:
            raise ValueError(f"duplicate transaction id: {transaction_id}")
        seen_ids.add(transaction_id)
        operation = str(row.get("op") or "").strip().lower()
        if operation not in {"replace", "delete"}:
            raise ValueError(f"{transaction_id} uses unsupported operation: {operation}")
        raw_targets = row.get("targets")
        if not isinstance(raw_targets, list) or not raw_targets:
            raise ValueError(f"{transaction_id} must name at least one target block")
        targets = tuple(str(item).strip() for item in raw_targets)
        if len(set(targets)) != len(targets) or any(item not in known for item in targets):
            raise ValueError(f"{transaction_id} contains duplicate or unknown targets")
        positions = sorted(order[item] for item in targets)
        if positions != list(range(positions[0], positions[-1] + 1)):
            raise ValueError(f"{transaction_id} targets must be contiguous")
        targets = tuple(blocks[position].block_id for position in positions)
        content = str(row.get("content") or "").strip()
        if operation == "replace" and not content:
            raise ValueError(f"{transaction_id} replacement content is empty")
        if re.search(r"<!--\s*(?:SECTION|PATCHES|END)", content, re.IGNORECASE):
            raise ValueError(f"{transaction_id} content contains transport markers")
        transactions.append(
            SectionPatchTransaction(transaction_id, operation, targets, content)
        )
    return transactions


def _apply_patch_transactions(
    before: str,
    blocks: list[MarkdownPatchBlock],
    transactions: list[SectionPatchTransaction],
) -> str:
    by_id = {block.block_id: block for block in blocks}
    edits: list[tuple[int, int, str]] = []
    for transaction in transactions:
        selected = [by_id[item] for item in transaction.targets]
        start, end = selected[0].start, selected[-1].end
        replacement = ""
        if transaction.operation == "replace":
            replacement = transaction.content.rstrip() + selected[-1].separator
        edits.append((start, end, replacement))
    result = before
    for start, end, replacement in sorted(edits, reverse=True):
        result = result[:start] + replacement + result[end:]
    return result


def _validate_patched_section(
    before: str,
    candidate: str,
    expected: SectionSpec,
    directives: str,
    allowed_source: str,
) -> list[str]:
    errors: list[str] = []
    try:
        _parse_section_output(candidate, expected, allow_unwrapped=True)
    except ValueError as exc:
        errors.append(str(exc))
    if current_harness_config().integrity_policy == "standard":
        errors.extend(
            _standard_patch_regression_errors(
                before, candidate, expected.markdown, directives
            )
        )
    else:
        errors.extend(
            _patch_regression_errors(
                before,
                candidate,
                expected.markdown,
                directives,
                allowed_source=allowed_source,
            )
        )
    return errors


def _transaction_regression_errors(
    transaction: SectionPatchTransaction,
    blocks: list[MarkdownPatchBlock],
    expected: SectionSpec,
    directives: str,
    allowed_source: str,
) -> list[str]:
    """Protect facts inside the edited range, not only section-wide totals."""
    by_id = {block.block_id: block for block in blocks}
    source = "\n\n".join(by_id[item].text for item in transaction.targets)
    replacement = transaction.content if transaction.operation == "replace" else ""
    replacement_normalized = _lossless_normalize(replacement)
    authorized_table_removal = _authorized_table_removal_text(source, directives)
    authorized_table_urls = set(_http_literals(authorized_table_removal))
    errors: list[str] = []

    if transaction.operation == "delete" and not _delete_lines(directives).strip():
        errors.append("delete transaction has no local DELETE directive")

    # Preserve local claim-to-citation binding. A repeated URL may disappear
    # section-wide without triggering the set-based guard, so reject a local
    # URL removal when a high-information claim marker from the same edited
    # range (number, attribution, or required entity) remains in the rewrite.
    removed_local_urls = set(_http_literals(source)) - set(_http_literals(replacement))
    if (
        transaction.operation == "replace"
        and removed_local_urls
        and not _http_literals(replacement)
    ):
        source_markers = set(_numeric_fact_literals(source)) | _citation_attribution_keys(
            source
        )
        replacement_markers = set(
            _numeric_fact_literals(replacement)
        ) | _citation_attribution_keys(replacement)
        source_entities = {
            _lossless_normalize(entity)
            for entity in _entity_labels(expected.markdown)
            if _normalized_text(entity) in _normalized_text(source)
        }
        retained_entities = {
            entity for entity in source_entities if entity in replacement_normalized
        }
        if source_markers.intersection(replacement_markers) or retained_entities:
            for url in sorted(removed_local_urls):
                citation_delete = any(
                    _deletion_authorizes(attribution, directives)
                    for attribution in _citation_attribution_literals(source)
                    if attribution not in replacement_normalized
                )
                if not citation_delete and not _deletion_authorizes(
                    url, directives, url=True
                ) and url not in authorized_table_urls:
                    errors.append(
                        "transaction detached a retained local claim from URL: " + url
                    )

    # A transaction may reuse facts from its own source range or from evidence
    # explicitly supplied to this editor. It may not silently detach a fact or
    # citation from some unrelated block elsewhere in the section.
    local_allowed = source + "\n" + allowed_source
    allowed_urls = set(_http_literals(local_allowed))
    for url in _http_literals(replacement):
        if url not in allowed_urls:
            errors.append(f"transaction imported URL from outside its evidence: {url}")
    allowed_numeric = set(_numeric_fact_literals(local_allowed))
    for literal in _numeric_fact_literals(replacement):
        if literal not in allowed_numeric:
            errors.append(
                f"transaction imported numeric fact from outside its evidence: {literal}"
            )
    allowed_attribution_keys = _citation_attribution_keys(local_allowed)
    for attribution in _citation_attribution_literals(replacement):
        if _citation_attribution_key(attribution) not in allowed_attribution_keys:
            errors.append(
                "transaction imported citation attribution from outside its evidence: "
                + attribution
            )
    return errors


def _retryable_patch_rejections(rejected: dict[str, list[str]]) -> bool:
    """Retry only local structural defects, never semantic fact loss."""
    if not rejected:
        return False
    allowed = (
        "target overlaps an accepted transaction",
        "markdown table count decreased",
        "markdown table header/order changed",
        "required table structure was lost",
        "keep directive lost markdown tables",
    )
    errors = [error.casefold() for values in rejected.values() for error in values]
    if not errors:
        return False
    if all(any(marker in error for marker in allowed) for error in errors):
        return True
    table_retry = any(
        marker in error
        for error in errors
        for marker in (
            "table fact was lost",
            "markdown table count decreased",
            "markdown table header/order changed",
        )
    )
    preservation_errors = (
        "urls removed without",
        "numeric fact or unit was lost",
        "citation attribution was lost",
        "table fact was lost",
        "markdown table count decreased",
        "markdown table header/order changed",
    )
    return table_retry and all(
        any(marker in error for marker in preservation_errors) for error in errors
    )


def _accept_patch_transactions(
    before: str,
    raw: str,
    expected: SectionSpec,
    directives: str,
    allowed_source: str,
) -> tuple[str, list[SectionPatchTransaction], dict[str, list[str]]]:
    blocks = _section_patch_blocks(before, expected)
    transactions = _parse_patch_transactions(raw, blocks)
    accepted: list[SectionPatchTransaction] = []
    rejected: dict[str, list[str]] = {}
    occupied: set[str] = set()
    for transaction in transactions:
        if occupied.intersection(transaction.targets):
            rejected[transaction.transaction_id] = ["target overlaps an accepted transaction"]
            continue
        errors = (
            _transaction_regression_errors(
                transaction, blocks, expected, directives, allowed_source
            )
            if current_harness_config().integrity_policy == "lossless"
            else []
        )
        if errors:
            rejected[transaction.transaction_id] = errors
            continue
        individual = _apply_patch_transactions(before, blocks, [transaction])
        errors = _validate_patched_section(
            before, individual, expected, directives, allowed_source
        )
        if errors:
            rejected[transaction.transaction_id] = errors
            continue
        combined = _apply_patch_transactions(before, blocks, accepted + [transaction])
        errors = _validate_patched_section(
            before, combined, expected, directives, allowed_source
        )
        if errors:
            rejected[transaction.transaction_id] = errors
            continue
        accepted.append(transaction)
        occupied.update(transaction.targets)
    return _apply_patch_transactions(before, blocks, accepted), accepted, rejected


def _canonical_patch_output(transactions: list[SectionPatchTransaction]) -> str:
    rows = []
    for transaction in transactions:
        row: dict[str, Any] = {
            "id": transaction.transaction_id,
            "op": transaction.operation,
            "targets": list(transaction.targets),
        }
        if transaction.operation == "replace":
            row["content"] = transaction.content
        rows.append(row)
    return (
        "<!-- PATCHES -->\n"
        + json.dumps({"transactions": rows}, ensure_ascii=False, separators=(",", ":"))
        + "\n<!-- END PATCHES -->"
    )


def _referenced_editorial_context(
    full_draft: str, directives: str, current_section_id: str
) -> str:
    referenced = set(re.findall(r"\bS\d+\.\d+\b", directives)) - {current_section_id}
    if not referenced:
        return ""
    sections = {section.section_id: section.markdown for section in _subsection_blocks(full_draft)}
    return "\n\n".join(
        f"# Allowed evidence from {section_id}\n\n{sections[section_id]}"
        for section_id in sorted(referenced)
        if section_id in sections
    )


def _propose_section_patches(
    question: str,
    full_draft: str,
    section: SectionSpec,
    current_body: str,
    relevant_rowbook: str,
    local_directives: str,
    *,
    iteration_feedback: str = "",
    round_index: int = 1,
) -> str:
    blocks = _section_patch_blocks(current_body, section)
    allowed_context = _referenced_editorial_context(
        full_draft, local_directives, section.section_id
    )
    feedback_block = (
        "\n# Iteration Feedback\n\n" + iteration_feedback + "\n"
        if iteration_feedback
        else ""
    )
    user = f"""# Original Question

{question}

# Assigned Section Contract

{section.markdown}

# Local Directives

{local_directives}

# Immutable Assigned Blocks

{_format_patch_blocks(blocks)}

# Allowed Cross-Section Evidence

{allowed_context or "None. Do not introduce material from other sections."}

# Relevant Fact Ledger

{relevant_rowbook or "None."}
{feedback_block}
"""
    agent = ResearchAgent(
        system_prompt=LOCAL_PATCH_EDITOR_SYSTEM_PROMPT,
        label=f"patch-editor-{section.section_id}",
        call_type="local_editor_patch",
        actor=f"patch-editor-{section.section_id}",
        trajectory_id=f"local-editor-patch-{section.section_id}",
    )
    body = agent.complete(
        user,
        max_tokens=12000,
        empty_error=f"patch-editor-{section.section_id} returned empty content",
        turn_index=2 * round_index - 1,
    )
    if PATCH_OUTPUT_RE.search(body):
        return body
    return agent.complete(
        user
        + f"""

# Format Repair

The previous answer did not contain the required PATCHES JSON envelope. Convert
only its intended edits into valid block transactions. Do not output a complete
section or explanatory prose.

Previous malformed answer:

{body}
""",
        max_tokens=8000,
        empty_error=f"patch-editor-{section.section_id} format repair returned empty content",
        turn_index=2 * round_index,
    )


def _patch_iteration_feedback(
    accepted: list[SectionPatchTransaction],
    rejected: dict[str, list[str]],
    validation_error: str = "",
) -> str:
    lines = [
        "This is the final correction round. The current blocks already include "
        "every accepted first-round edit. Do not repeat or undo accepted edits.",
        "Retry only the unresolved edit using a smaller transaction. Preserve all "
        "tables unless the directive explicitly names the exact table repair.",
    ]
    if accepted:
        lines.append(
            "Already accepted transaction IDs: "
            + ", ".join(item.transaction_id for item in accepted)
        )
    for transaction_id, errors in rejected.items():
        lines.append(f"Rejected {transaction_id}: " + "; ".join(errors))
    if validation_error:
        lines.append("Round validation error: " + validation_error)
    return "\n".join(lines)


def _section_diff(section_id: str, before: str, after: str) -> str:
    return "".join(
        difflib.unified_diff(
            before.splitlines(keepends=True),
            after.splitlines(keepends=True),
            fromfile=f"{section_id}.before.md",
            tofile=f"{section_id}.after.md",
        )
    )


def _report_metrics(markdown: str) -> dict[str, int]:
    urls = re.findall(r"\[[^\]]*\]\((https?://[^)]+)\)", markdown)
    return {
        "characters": len(markdown),
        "words": len(re.findall(r"\b\w+\b", markdown)),
        "citations": len(urls),
        "unique_urls": len(set(urls)),
    }


def _write_editorial_summary(run_dir: Path, before: str, after: str) -> None:
    old = _report_metrics(before)
    new = _report_metrics(after)
    reduction = 100 * (1 - new["characters"] / max(1, old["characters"]))
    summary = f"""# Editorial Summary

| Metric | Before | After |
|---|---:|---:|
| Characters | {old['characters']:,} | {new['characters']:,} |
| Words | {old['words']:,} | {new['words']:,} |
| Citation occurrences | {old['citations']:,} | {new['citations']:,} |
| Unique URLs | {old['unique_urls']:,} | {new['unique_urls']:,} |

Character reduction: **{reduction:.1f}%**
"""
    _atomic_write(run_dir / "editorial_summary.md", summary)


def _apply_editorial_plan(
    question: str,
    research_spec: str,
    report_spec: str,
    sections: list[SectionSpec],
    facts: dict[str, str],
    bodies: dict[str, str],
    draft: str,
    run_dir: Path,
    *,
    concurrency: int,
) -> dict[str, str]:
    plan, directives = _plan_editorial_changes(question, research_spec, draft)
    valid_ids = {section.section_id for section in sections}
    unknown_ids = set(directives) - valid_ids
    if unknown_ids:
        _atomic_write(
            run_dir / "editorial_plan_warnings.txt",
            f"Ignored directives for absent section IDs: {sorted(unknown_ids)}\n",
        )
        directives = {
            section_id: note
            for section_id, note in directives.items()
            if section_id in valid_ids
        }
    _atomic_write(run_dir / "editorial_plan.md", plan)

    editorial_dir = run_dir / "editorial"
    edited = dict(bodies)
    revision_rounds = current_harness_config().planning.editor_revision_rounds
    retry_sections: dict[str, tuple[SectionSpec, str, str, str]] = {}

    def mark_final(section: SectionSpec, before: str, error: str = "") -> None:
        body = edited[section.section_id]
        if body == before:
            _atomic_write(
                editorial_dir / f"{section.section_id}.error.txt",
                (error or "ValueError: no safe patch transaction was accepted") + "\n",
            )
            return
        _atomic_write(editorial_dir / f"{section.section_id}.after.md", body)
        _atomic_write(
            editorial_dir / f"{section.section_id}.diff",
            _section_diff(section.section_id, before, body),
        )

    futures: dict[Any, tuple[SectionSpec, str, str]] = {}
    with ThreadPoolExecutor(max_workers=max(1, min(concurrency, len(sections)))) as executor:
        for section in sections:
            local = directives.get(section.section_id)
            if not local:
                continue
            before = bodies[section.section_id]
            _atomic_write(editorial_dir / f"{section.section_id}.before.md", before)
            _atomic_write(editorial_dir / f"{section.section_id}.directives.md", local)
            context = copy_context()
            future = executor.submit(
                context.run,
                _propose_section_patches,
                question,
                draft,
                section,
                before,
                _relevant_rowbook(section.section_id, facts),
                local,
            )
            futures[future] = (section, before, local)

        for future in as_completed(futures):
            section, before, _local = futures[future]
            try:
                raw = future.result()
            except Exception as exc:
                _atomic_write(
                    editorial_dir / f"{section.section_id}.error.txt",
                    f"{type(exc).__name__}: {exc}\n",
                )
                continue
            _atomic_write(editorial_dir / f"{section.section_id}.raw.md", raw)
            parsed = False
            try:
                rowbook = _relevant_rowbook(section.section_id, facts)
                allowed_context = _referenced_editorial_context(
                    draft, _local, section.section_id
                )
                body, accepted_transactions, rejected_transactions = (
                    _accept_patch_transactions(
                        before,
                        raw,
                        section,
                        _local,
                        allowed_context + "\n" + rowbook,
                    )
                )
                parsed = True
                _atomic_write(
                    editorial_dir / f"{section.section_id}.transactions.json",
                    json.dumps(
                        {
                            "accepted": [
                                transaction.transaction_id
                                for transaction in accepted_transactions
                            ],
                            "rejected": rejected_transactions,
                        },
                        ensure_ascii=False,
                        indent=2,
                    )
                    + "\n",
                )
                if revision_rounds == 2 and _retryable_patch_rejections(
                    rejected_transactions
                ):
                    feedback = _patch_iteration_feedback(
                        accepted_transactions, rejected_transactions
                    )
                    retry_sections[section.section_id] = (
                        section,
                        before,
                        _local,
                        feedback,
                    )
                    _atomic_write(
                        editorial_dir / f"{section.section_id}.round2.feedback.md",
                        feedback + "\n",
                    )
                if not accepted_transactions or body == before:
                    raise ValueError("no safe patch transaction was accepted")
                regressions = _validate_patched_section(
                    before,
                    body,
                    section,
                    _local,
                    allowed_context + "\n" + rowbook,
                )
                if regressions:
                    raise ValueError("; ".join(regressions))
                edited[section.section_id] = body
            except ValueError as exc:
                if revision_rounds == 2 and section.section_id not in retry_sections:
                    if not parsed:
                        feedback = _patch_iteration_feedback([], {}, str(exc))
                        retry_sections[section.section_id] = (
                            section,
                            before,
                            _local,
                            feedback,
                        )
                        _atomic_write(
                            editorial_dir / f"{section.section_id}.round2.feedback.md",
                            feedback + "\n",
                        )
                if section.section_id in retry_sections:
                    _atomic_write(
                        editorial_dir / f"{section.section_id}.round1.error.txt",
                        f"{type(exc).__name__}: {exc}\n",
                    )
                else:
                    mark_final(section, before, f"{type(exc).__name__}: {exc}")
                continue
            if section.section_id not in retry_sections:
                mark_final(section, before)

    if retry_sections:
        retry_futures: dict[Any, tuple[SectionSpec, str, str]] = {}
        with ThreadPoolExecutor(
            max_workers=max(1, min(concurrency, len(retry_sections)))
        ) as executor:
            for section, before, local, feedback in retry_sections.values():
                current = edited[section.section_id]
                context = copy_context()
                future = executor.submit(
                    context.run,
                    _propose_section_patches,
                    question,
                    draft,
                    section,
                    current,
                    _relevant_rowbook(section.section_id, facts),
                    local,
                    iteration_feedback=feedback,
                    round_index=2,
                )
                retry_futures[future] = (section, before, local)

            for future in as_completed(retry_futures):
                section, before, local = retry_futures[future]
                current = edited[section.section_id]
                error = ""
                try:
                    raw = future.result()
                    _atomic_write(
                        editorial_dir / f"{section.section_id}.round2.raw.md", raw
                    )
                    rowbook = _relevant_rowbook(section.section_id, facts)
                    allowed_context = _referenced_editorial_context(
                        draft, local, section.section_id
                    )
                    body, accepted_transactions, rejected_transactions = (
                        _accept_patch_transactions(
                            current,
                            raw,
                            section,
                            local,
                            allowed_context + "\n" + rowbook,
                        )
                    )
                    cumulative_errors = _validate_patched_section(
                        before,
                        body,
                        section,
                        local,
                        allowed_context + "\n" + rowbook,
                    )
                    if cumulative_errors:
                        rejected_transactions["ROUND2_CUMULATIVE"] = cumulative_errors
                        accepted_transactions = []
                        body = current
                    _atomic_write(
                        editorial_dir
                        / f"{section.section_id}.round2.transactions.json",
                        json.dumps(
                            {
                                "accepted": [
                                    item.transaction_id
                                    for item in accepted_transactions
                                ],
                                "rejected": rejected_transactions,
                            },
                            ensure_ascii=False,
                            indent=2,
                        )
                        + "\n",
                    )
                    if accepted_transactions and body != current:
                        edited[section.section_id] = body
                    if not accepted_transactions:
                        error = "ValueError: no safe round-2 patch was accepted"
                except Exception as exc:
                    error = f"{type(exc).__name__}: {exc}"
                    _atomic_write(
                        editorial_dir / f"{section.section_id}.round2.error.txt",
                        error + "\n",
                    )
                mark_final(section, before, error)
    changed = sum(edited[key] != bodies[key] for key in edited)
    _atomic_write(
        run_dir / "editorial_patch_summary.md",
        f"# Sparse Editorial Patch Summary\n\n"
        f"- Sections in draft: {len(sections)}\n"
        f"- Sections with actionable directives: {len(directives)}\n"
        f"- Successfully edited sections: {changed}\n"
        f"- Rejected or failed edits: {len(list(editorial_dir.glob('*.error.txt')))}\n"
        f"- Passed through byte-for-byte: {len(sections) - changed}\n",
    )
    return edited


def _apply_standard_editorial_plan(
    question: str,
    research_spec: str,
    report_spec: str,
    sections: list[SectionSpec],
    facts: dict[str, str],
    bodies: dict[str, str],
    draft: str,
    run_dir: Path,
    *,
    concurrency: int,
) -> dict[str, str]:
    plan, directives = _plan_editorial_changes(question, research_spec, draft)
    valid_ids = {section.section_id for section in sections}
    unknown_ids = set(directives) - valid_ids
    if unknown_ids:
        _atomic_write(
            run_dir / "editorial_plan_warnings.txt",
            f"Ignored directives for absent section IDs: {sorted(unknown_ids)}\n",
        )
        directives = {
            section_id: note
            for section_id, note in directives.items()
            if section_id in valid_ids
        }
    _atomic_write(run_dir / "editorial_plan.md", plan)

    editorial_dir = run_dir / "editorial"
    edited = dict(bodies)
    futures: dict[Any, tuple[SectionSpec, str, str]] = {}
    with ThreadPoolExecutor(max_workers=max(1, min(concurrency, len(sections)))) as executor:
        for section in sections:
            local = directives.get(section.section_id)
            if not local:
                continue
            before = bodies[section.section_id]
            _atomic_write(editorial_dir / f"{section.section_id}.before.md", before)
            _atomic_write(editorial_dir / f"{section.section_id}.directives.md", local)
            context = copy_context()
            future = executor.submit(
                context.run,
                _edit_one_section,
                question,
                report_spec,
                draft,
                plan,
                section,
                before,
                _relevant_rowbook(section.section_id, facts),
                local,
            )
            futures[future] = (section, before, local)

        for future in as_completed(futures):
            section, before, _local = futures[future]
            try:
                raw = future.result()
            except Exception as exc:
                _atomic_write(
                    editorial_dir / f"{section.section_id}.error.txt",
                    f"{type(exc).__name__}: {exc}\n",
                )
                continue
            _atomic_write(editorial_dir / f"{section.section_id}.raw.md", raw)
            try:
                try:
                    body = _parse_section_output(raw, section, allow_unwrapped=True)
                except ValueError:
                    recovered = (
                        _recover_section_submission(raw, section)
                        if _uses_auto_markdown_recovery()
                        else None
                    )
                    if recovered is None:
                        raise
                    body = _parse_section_output(recovered, section)
                regressions = _standard_patch_regression_errors(
                    before, body, section.markdown, _local
                )
                if regressions:
                    raise ValueError("; ".join(regressions))
            except ValueError as exc:
                _atomic_write(
                    editorial_dir / f"{section.section_id}.error.txt",
                    f"{type(exc).__name__}: {exc}\n",
                )
                continue
            _atomic_write(editorial_dir / f"{section.section_id}.after.md", body)
            _atomic_write(
                editorial_dir / f"{section.section_id}.diff",
                _section_diff(section.section_id, before, body),
            )
            edited[section.section_id] = body
    changed = sum(edited[key] != bodies[key] for key in edited)
    _atomic_write(
        run_dir / "editorial_patch_summary.md",
        f"# Sparse Editorial Patch Summary\n\n"
        f"- Sections in draft: {len(sections)}\n"
        f"- Sections with actionable directives: {len(directives)}\n"
        f"- Successfully edited sections: {changed}\n"
        f"- Rejected or failed edits: {len(list(editorial_dir.glob('*.error.txt')))}\n"
        f"- Passed through byte-for-byte: {len(sections) - changed}\n",
    )
    return edited


def apply_editorial_strategy(
    question: str,
    research_spec: str,
    report_spec: str,
    sections: list[SectionSpec],
    facts: dict[str, str],
    bodies: dict[str, str],
    draft: str,
    run_dir: Path,
    *,
    concurrency: int,
    mode: str = "lossless_patch",
) -> dict[str, str]:
    """Select one Local Editor implementation behind a stable stage contract."""
    if mode == "off":
        return dict(bodies)
    if mode == "standard":
        edited = _apply_standard_editorial_plan(
            question,
            research_spec,
            report_spec,
            sections,
            facts,
            bodies,
            draft,
            run_dir,
            concurrency=concurrency,
        )
        if current_harness_config().integrity_policy == "standard":
            return edited
        plan = (run_dir / "editorial_plan.md").read_text(encoding="utf-8")
        directives = _parse_section_directives(plan)
        guarded = dict(edited)
        for section in sections:
            before = bodies[section.section_id]
            after = edited[section.section_id]
            if before == after:
                continue
            local = directives.get(section.section_id, "")
            allowed_source = _referenced_editorial_context(
                draft, local, section.section_id
            ) + "\n" + _relevant_rowbook(section.section_id, facts)
            errors = _patch_regression_errors(
                before,
                after,
                section.markdown,
                local,
                allowed_source=allowed_source,
            )
            if errors:
                guarded[section.section_id] = before
                _atomic_write(
                    run_dir / "editorial" / f"{section.section_id}.lossless-rejected.txt",
                    "\n".join(errors).rstrip() + "\n",
                )
        return guarded
    if mode == "lossless_patch":
        return _apply_editorial_plan(
            question,
            research_spec,
            report_spec,
            sections,
            facts,
            bodies,
            draft,
            run_dir,
            concurrency=concurrency,
        )
    raise ValueError(f"invalid editor mode: {mode}")
