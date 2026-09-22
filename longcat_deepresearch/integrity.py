from __future__ import annotations

import re
import unicodedata
from urllib.parse import urlsplit

def _high_priority_lead_errors(critique: str, final_spec: str) -> list[str]:
    """Require an explicit disposition for every Critic HP lead.

    The Reviser may place an ID in a subsection's entity/source blocks or in
    the optional rejected-leads section. This keeps the contract inspectable
    without introducing JSON or another pipeline stage.
    """
    lead_ids = sorted(
        set(re.findall(r"\bHP\d+\b", critique, flags=re.IGNORECASE)),
        key=lambda item: int(re.search(r"\d+", item).group()),
    )
    present = {item.upper() for item in re.findall(r"\bHP\d+\b", final_spec, re.IGNORECASE)}
    return [
        f"high-priority Critic lead {lead_id.upper()} was neither absorbed nor explicitly rejected"
        for lead_id in lead_ids
        if lead_id.upper() not in present
    ]


def _lossless_normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", str(text or "")).casefold()
    text = re.sub(r"[‐‑‒–—−]", "-", text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s*([/%°:+×x-])\s*", r"\1", text)
    return text.strip(" \t\r\n.,;:!?，。；：！？`*_[]()")


def _http_literals(text: str) -> list[str]:
    urls: list[str] = []
    for raw in re.findall(
        r"https?://[^\s<>()\[\]{}\"'`“”‘’、，。；]+",
        text,
        flags=re.IGNORECASE,
    ):
        value = raw.rstrip(".,;:!?，。；：！？\"'`”’」』】》")
        if value and value not in urls:
            urls.append(value)
    return urls


def _numeric_fact_literals(text: str) -> list[str]:
    """Extract exact high-information numeric facts, excluding section/lead IDs."""
    scrubbed = re.sub(r"https?://[^\s<>()]+", " ", text, flags=re.IGNORECASE)
    scrubbed = re.sub(r"(?im)^#{1,6}\s+.*$", " ", scrubbed)
    scrubbed = re.sub(r"\b(?:HP|S|F)\d+(?:\.\d+)?\b", " ", scrubbed)
    unit = (
        r"(?:%|‰|°c|℃|hz|khz|mhz|ghz|thz|km/h|m/s|km|cm|mm|μm|µm|nm|pm|"
        r"kg|mg|μg|µg|g|pa|kpa|mpa|gpa|v|mv|kv|a|ma|μa|µa|w|mw|kw|"
        r"ev|mev|kev|s|ms|μs|µs|ns|ps|fps|rpm|db|tb|gb|mb|kb|bps|bit/s|"
        r"mol|mmol|μmol|µmol|m|mm|μm|µm|wt%|at%|k|"
        r"ω(?:[\s·⋅*]*[μµu]?m)?|ohms?|m²|cm²|km²|"
        r"million|billion|trillion|万|亿)"
    )
    pattern = re.compile(
        rf"(?<![\w.])(?:[~≈<>≤≥±]\s*)?"
        rf"(?:[$€£¥₹]\s*)?"
        rf"(?:19\d{{2}}|20\d{{2}}|\d+(?:[.,]\d+)?(?:e[+-]?\d+)?)"
        rf"(?:\s*(?:[‐‑‒–—−-]|:|±)\s*\d+(?:[.,]\d+)?)?"
        rf"(?:\s*{unit})?(?=$|[^\w])",
        re.IGNORECASE,
    )
    values: list[str] = []
    for match in pattern.finditer(scrubbed):
        raw = match.group(0).strip()
        normalized_spaced = _lossless_normalize(raw)
        has_unit = bool(re.search(unit + r"$", normalized_spaced, flags=re.IGNORECASE))
        has_currency = bool(re.match(r"^[$€£¥₹]", raw))
        normalized = re.sub(r"\s+", "", normalized_spaced)
        is_year = bool(re.fullmatch(r"(?:19|20)\d{2}", normalized))
        is_range_or_decimal = bool(re.search(r"[-.,:]|e[+-]?\d+|±", normalized))
        if not (has_unit or has_currency or is_year or is_range_or_decimal):
            continue
        if normalized and normalized not in values:
            values.append(normalized)
    return values


def _citation_attribution_literals(text: str) -> list[str]:
    """Protect explicit author-year attributions while allowing prose rewrites."""
    patterns = (
        r"\b[A-Z][A-Za-zÀ-ɏ'\-’]+(?:\s+(?:et\s+al\.?|and|&)\s*"
        r"[A-Z]?[A-Za-zÀ-ɏ'\-’.]*)?\s*(?:\(|,\s*)(?:19|20)\d{2}[a-z]?\)?",
        r"\[[^\]\n]{2,100}(?:19|20)\d{2}[a-z]?\]\(https?://[^)]+\)",
    )
    values: list[str] = []
    for pattern in patterns:
        for match in re.finditer(pattern, text):
            raw = match.group(0)
            if raw.startswith("["):
                raw = raw[1 : raw.index("]")]
            normalized = _lossless_normalize(raw)
            if normalized and normalized not in values:
                values.append(normalized)
    return values


def _citation_attribution_key(literal: str) -> str:
    """Collapse harmless punctuation variants to an author/year identity."""
    normalized = _lossless_normalize(literal)
    year = re.search(r"(?:19|20)\d{2}[a-z]?", normalized)
    author = re.match(r"[a-zà-ɏ'\-’]+", normalized)
    return f"{author.group(0)}:{year.group(0)}" if year and author else normalized


def _citation_attribution_keys(text: str) -> set[str]:
    return {
        _citation_attribution_key(literal)
        for literal in _citation_attribution_literals(text)
    }


def _required_entity_literals(spec: str) -> list[str]:
    blocks = re.split(
        r"(?=^###\s+S\d+\.\d+\s*[｜|:])", spec, flags=re.MULTILINE
    )[1:]
    values: list[str] = []
    for block in blocks:
        for entity in _entity_labels(block):
            normalized = _lossless_normalize(entity)
            if normalized and normalized not in values:
                values.append(normalized)
    return values


def _table_fact_literals(markdown: str) -> list[str]:
    """Protect row keys plus compact factual cells, not presentation wording."""
    values: list[str] = []
    lines = markdown.splitlines()
    for index, line in enumerate(lines):
        if not re.match(r"^\s*\|.*\|\s*$", line):
            continue
        if index + 1 >= len(lines) or not re.match(
            r"^\s*\|?(?:\s*:?-{3,}:?\s*\|)+\s*$", lines[index + 1]
        ):
            continue
        row_index = index + 2
        while row_index < len(lines) and re.match(
            r"^\s*\|.*\|\s*$", lines[row_index]
        ):
            cells = [cell.strip() for cell in lines[row_index].strip().strip("|").split("|")]
            for cell_index, cell in enumerate(cells):
                plain = re.sub(r"\[([^]]+)\]\([^)]+\)", r"\1", cell)
                normalized = _lossless_normalize(plain)
                factual = (
                    cell_index == 0
                    or bool(_numeric_fact_literals(plain))
                    or bool(_http_literals(cell))
                    or (len(normalized) <= 40 and bool(re.search(r"[A-Z\d\u3400-\u9fff]", plain)))
                )
                if factual and len(normalized) >= 2 and normalized not in values:
                    values.append(normalized)
            row_index += 1
    return values


def _delete_lines(directives: str) -> str:
    return "\n".join(
        line
        for line in directives.splitlines()
        if re.match(r"^\s*[-*]\s*(?:\*\*)?DELETE(?:\*\*)?\s*:", line, re.IGNORECASE)
    )


def _deletion_authorizes(literal: str, directives: str, *, url: bool = False) -> bool:
    delete_text = _lossless_normalize(_delete_lines(directives))
    if not delete_text:
        return False
    target = _lossless_normalize(literal)
    if target and target in delete_text:
        return True
    if target and re.sub(r"\s+", "", target) in re.sub(r"\s+", "", delete_text):
        return True
    if url:
        try:
            hostname = urlsplit(literal).hostname or ""
        except ValueError:
            hostname = ""
        return bool(hostname and _lossless_normalize(hostname) in delete_text)
    return False


def _planning_semantic_regression_errors(before: str, after: str) -> list[str]:
    """Reject a Reviser result that drops explicit planning facts."""
    errors: list[str] = []
    after_normalized = _lossless_normalize(after)
    after_compact = re.sub(r"\s+", "", after_normalized)
    for url in _http_literals(before):
        if url not in _http_literals(after):
            errors.append(f"lossless planning guard lost URL: {url}")
    for literal in _numeric_fact_literals(before):
        if literal not in after_compact:
            errors.append(f"lossless planning guard lost numeric fact: {literal}")
    for entity in _required_entity_literals(before):
        if entity not in after_normalized:
            errors.append(f"lossless planning guard lost required entity/case: {entity}")
    for attribution in _citation_attribution_literals(before):
        if attribution not in after_normalized:
            errors.append(
                f"lossless planning guard lost citation attribution: {attribution}"
            )
    return errors



def _entity_labels(section_spec: str) -> list[str]:
    match = re.search(
        r"^####\s+(?:Required Entities / Cases|必需实体\s*/?\s*案例|必需实体与案例)\s*$\n"
        r"(.*?)(?=^####\s+|^###\s+|^##\s+|\Z)",
        section_spec,
        re.IGNORECASE | re.MULTILINE | re.DOTALL,
    )
    if not match:
        return []
    labels: list[str] = []
    for line in match.group(1).splitlines():
        item = re.sub(r"^\s*[-*+]\s+", "", line).strip()
        if not item or re.search(r"none identified|暂无|待确认", item, re.IGNORECASE):
            continue
        linked = re.match(r"\[([^]]+)\]\([^)]+\)", item)
        if linked:
            item = linked.group(1)
        item = re.sub(r"[*_`]", "", item)
        item = re.split(r"\s+(?:—|–|-|:|：)\s+", item, maxsplit=1)[0].strip()
        if 2 <= len(item) <= 100:
            labels.append(item)
    return labels


def _normalized_text(text: str) -> str:
    return re.sub(r"[^\w]+", " ", text.casefold()).strip()


def _markdown_table_count(text: str) -> int:
    separator = re.compile(
        r"^\s*\|?\s*:?-{3,}:?\s*(?:\|\s*:?-{3,}:?\s*)+\|?\s*$"
    )
    return sum(separator.match(line) is not None for line in text.splitlines())


def _markdown_table_headers(text: str) -> list[tuple[str, ...]]:
    """Return normalized Markdown table headers in document order."""
    lines = text.splitlines()
    separator = re.compile(
        r"^\s*\|?\s*:?-{3,}:?\s*(?:\|\s*:?-{3,}:?\s*)+\|?\s*$"
    )
    headers: list[tuple[str, ...]] = []
    for index in range(len(lines) - 1):
        if not re.match(r"^\s*\|.*\|\s*$", lines[index]):
            continue
        if not separator.match(lines[index + 1]):
            continue
        cells = [
            _lossless_normalize(re.sub(r"\[([^]]+)\]\([^)]+\)", r"\1", cell))
            for cell in lines[index].strip().strip("|").split("|")
        ]
        headers.append(tuple(cell for cell in cells if cell))
    return headers


def _plain_table_cell(cell: str) -> str:
    return re.sub(r"\[([^]]+)\]\([^)]+\)", r"\1", cell).strip()


def _authorized_table_removal_text(text: str, directives: str) -> str:
    """Return cells explicitly covered by table-column or row DELETE lines."""
    delete_text = _lossless_normalize(_delete_lines(directives))
    if not delete_text:
        return ""
    lines = text.splitlines()
    separator = re.compile(
        r"^\s*\|?\s*:?-{3,}:?\s*(?:\|\s*:?-{3,}:?\s*)+\|?\s*$"
    )
    authorized: list[str] = []
    index = 0
    while index + 1 < len(lines):
        if not re.match(r"^\s*\|.*\|\s*$", lines[index]) or not separator.match(
            lines[index + 1]
        ):
            index += 1
            continue
        headers = [cell.strip() for cell in lines[index].strip().strip("|").split("|")]
        deleted_columns = {
            column
            for column, header in enumerate(headers)
            if _lossless_normalize(_plain_table_cell(header)) in delete_text
            and bool(
                re.search(
                    rf"(?mi)^\s*[-*]\s*(?:\*\*)?DELETE(?:\*\*)?\s*:.*"
                    rf"{re.escape(_plain_table_cell(header))}.*(?:column|列)",
                    directives,
                )
            )
        }
        authorized.extend(headers[column] for column in deleted_columns)
        row_index = index + 2
        while row_index < len(lines) and re.match(
            r"^\s*\|.*\|\s*$", lines[row_index]
        ):
            cells = [
                cell.strip()
                for cell in lines[row_index].strip().strip("|").split("|")
            ]
            row_key = _lossless_normalize(_plain_table_cell(cells[0])) if cells else ""
            row_alias = re.sub(
                r"\s+(?:et\s+al\.?\s+)?(?:19|20)\d{2}[a-z]?$", "", row_key
            )
            cell_markers = [
                _lossless_normalize(_plain_table_cell(cell)) for cell in cells
            ]
            named_status = any(
                len(marker) >= 8 and marker in delete_text
                for marker in cell_markers[1:]
            )
            if row_key and (
                row_key in delete_text
                or (len(row_alias) >= 4 and row_alias in delete_text)
                or named_status
            ):
                authorized.extend(cells)
            else:
                authorized.extend(
                    cells[column] for column in deleted_columns if column < len(cells)
                )
            row_index += 1
        index = row_index
    return "\n".join(authorized)


def _authorized_table_header_orders(directives: str) -> list[tuple[str, ...]]:
    orders: list[tuple[str, ...]] = []
    for value in re.findall(r"`([^`\n]*\|[^`\n]*)`", directives):
        cells = tuple(
            _lossless_normalize(cell) for cell in value.split("|") if cell.strip()
        )
        if len(cells) >= 2:
            orders.append(cells)
    return orders


def _table_structure_regression_errors(
    before: str, after: str, directives: str
) -> list[str]:
    """Protect table identity and header order unless explicitly edited."""
    before_headers = _markdown_table_headers(before)
    after_headers = _markdown_table_headers(after)
    if not before_headers:
        return []
    delete_table = bool(
        re.search(
            r"(?mi)^\s*[-*]\s*(?:\*\*)?DELETE(?:\*\*)?\s*:.*(?:\btable\b|表格)",
            directives,
        )
    )
    edit_structure = bool(
        re.search(
            r"(?is)(?:\btable\b|表格).{0,100}(?:column|header|order|reorder|列|表头|顺序)|"
            r"(?:column|header|order|reorder|列|表头|顺序).{0,100}(?:\btable\b|表格)",
            directives,
        )
    )
    authorized_orders = _authorized_table_header_orders(directives)
    errors: list[str] = []
    if len(after_headers) < len(before_headers) and not delete_table:
        errors.append(
            f"Markdown table count decreased without table DELETE: "
            f"{len(before_headers)} -> {len(after_headers)}"
        )
    remaining = list(after_headers)
    for header in before_headers:
        if header in remaining:
            remaining.remove(header)
            continue
        if edit_structure:
            exact = next(
                (candidate for candidate in remaining if candidate in authorized_orders),
                None,
            )
            if exact is not None:
                remaining.remove(exact)
                continue
            unordered = sorted(header)
            match = next(
                (candidate for candidate in remaining if sorted(candidate) == unordered),
                None,
            )
            if match is not None:
                remaining.remove(match)
                continue
        if not delete_table:
            errors.append(
                "Markdown table header/order changed without an exact table directive: "
                + " | ".join(header)
            )
    return errors


def _explicit_delete(directives: str) -> bool:
    return bool(re.search(r"(?mi)^\s*[-*]\s*DELETE\s*:", directives))


def _standard_patch_regression_errors(
    before: str,
    after: str,
    section_spec: str,
    directives: str,
) -> list[str]:
    """Reject unsafe local edits instead of guessing how to repair them."""
    errors: list[str] = []
    body = re.sub(r"(?m)^#{1,6}\s+.*$", "", after, count=1).strip()
    if len(before) >= 1000 and len(body) < 200:
        errors.append("edited body is implausibly short")

    editor_leak = re.search(
        r"Complete revised assigned report unit Markdown with citations\.|"
        r"^(?:Let me|Now let me|Actually\b|Wait\b|OK\b|I (?:should|think|need to|will)\b|"
        r"Looking at\b|The (?:local )?directives?\b|The current S\d+(?:\.\d+)? section\b)",
        body,
        re.IGNORECASE | re.MULTILINE,
    )
    if editor_leak:
        errors.append("editor commentary or prompt placeholder leaked into output")

    before_urls = set(re.findall(r"https?://[^)\s]+", before))
    after_urls = set(re.findall(r"https?://[^)\s]+", after))
    removed_urls = sorted(before_urls - after_urls)
    if removed_urls and not _explicit_delete(directives):
        errors.append(
            "URLs removed without an explicit DELETE directive: "
            + ", ".join(removed_urls)
        )

    normalized_before = _normalized_text(before)
    normalized_after = _normalized_text(after)
    for entity in _entity_labels(section_spec):
        normalized_entity = _normalized_text(entity)
        if normalized_entity in normalized_before and normalized_entity not in normalized_after:
            errors.append(f"required entity/case was lost: {entity}")

    table_rows_before = len(re.findall(r"(?m)^\s*\|.*\|\s*$", before))
    table_rows_after = len(re.findall(r"(?m)^\s*\|.*\|\s*$", after))
    presentation_requires_table = bool(
        re.search(
            r"^####\s+(?:Presentation|呈现要求)\s*$.*?\btable\b|"
            r"^####\s+呈现要求\s*$.*?表格",
            section_spec,
            re.IGNORECASE | re.MULTILINE | re.DOTALL,
        )
    )
    if presentation_requires_table and table_rows_before and not table_rows_after:
        errors.append("required table structure was lost")

    keep_table = bool(
        re.search(
            r"(?mi)^\s*[-*]\s*(?:\*\*)?KEEP(?:\*\*)?\s*:.*(?:\btable\b|表格)",
            directives,
        )
    )
    before_tables = _markdown_table_count(before)
    after_tables = _markdown_table_count(after)
    if keep_table and after_tables < before_tables:
        errors.append(
            f"KEEP directive lost Markdown tables: {before_tables} -> {after_tables}"
        )
    return errors


def _patch_regression_errors(
    before: str,
    after: str,
    section_spec: str,
    directives: str,
    allowed_source: str = "",
) -> list[str]:
    """Reject unsafe local edits instead of guessing how to repair them."""
    errors: list[str] = []
    body = re.sub(r"(?m)^#{1,6}\s+.*$", "", after, count=1).strip()
    if len(before) >= 1000 and len(body) < 200:
        errors.append("edited body is implausibly short")

    editor_leak = re.search(
        r"Complete revised assigned report unit Markdown with citations\.|"
        r"^(?:Let me|Now let me|Actually\b|Wait\b|OK\b|I (?:should|think|need to|will)\b|"
        r"Looking at\b|The (?:local )?directives?\b|The current S\d+(?:\.\d+)? section\b)",
        body,
        re.IGNORECASE | re.MULTILINE,
    )
    if editor_leak:
        errors.append("editor commentary or prompt placeholder leaked into output")

    authorized_table_removal = _authorized_table_removal_text(before, directives)
    authorized_table_normalized = _lossless_normalize(authorized_table_removal)
    authorized_table_urls = set(_http_literals(authorized_table_removal))
    authorized_table_numeric = set(_numeric_fact_literals(authorized_table_removal))
    authorized_table_attributions = _citation_attribution_keys(
        authorized_table_removal
    )

    before_urls = set(_http_literals(before))
    after_urls = set(_http_literals(after))
    removed_urls = sorted(before_urls - after_urls)
    unauthorized_urls = [
        url
        for url in removed_urls
        if url not in authorized_table_urls
        and not _deletion_authorizes(url, directives, url=True)
    ]
    if unauthorized_urls:
        errors.append(
            "URLs removed without an exact DELETE directive: "
            + ", ".join(unauthorized_urls)
        )

    allowed_urls = set(_http_literals(before + "\n" + allowed_source))
    invented_urls = sorted(after_urls - allowed_urls)
    if invented_urls:
        errors.append("URLs added without source evidence: " + ", ".join(invented_urls))

    normalized_before = _normalized_text(before)
    normalized_after = _normalized_text(after)
    for entity in _entity_labels(section_spec):
        normalized_entity = _normalized_text(entity)
        if (
            normalized_entity in normalized_before
            and normalized_entity not in normalized_after
            and not _deletion_authorizes(entity, directives)
        ):
            errors.append(f"required entity/case was lost: {entity}")

    lossless_after = _lossless_normalize(after)
    compact_after = re.sub(r"\s+", "", lossless_after)
    for literal in _numeric_fact_literals(before):
        if (
            literal not in compact_after
            and literal not in authorized_table_numeric
            and not _deletion_authorizes(literal, directives)
        ):
            errors.append(f"numeric fact or unit was lost: {literal}")

    allowed_numeric = set(_numeric_fact_literals(before + "\n" + allowed_source))
    invented_numeric = [
        literal for literal in _numeric_fact_literals(after) if literal not in allowed_numeric
    ]
    if invented_numeric:
        errors.append(
            "numeric facts added without source evidence: " + ", ".join(invented_numeric)
        )

    for attribution in _citation_attribution_literals(before):
        if (
            attribution not in lossless_after
            and _citation_attribution_key(attribution)
            not in authorized_table_attributions
            and not _deletion_authorizes(attribution, directives)
        ):
            errors.append(f"citation attribution was lost: {attribution}")

    allowed_attributions = set(
        _citation_attribution_literals(before + "\n" + allowed_source)
    )
    invented_attributions = [
        literal
        for literal in _citation_attribution_literals(after)
        if literal not in allowed_attributions
    ]
    if invented_attributions:
        errors.append(
            "citation attributions added without source evidence: "
            + ", ".join(invented_attributions)
        )

    for literal in _table_fact_literals(before):
        if (
            literal not in lossless_after
            and literal not in authorized_table_normalized
            and not _deletion_authorizes(literal, directives)
        ):
            errors.append(f"table fact was lost: {literal}")

    table_rows_before = len(re.findall(r"(?m)^\s*\|.*\|\s*$", before))
    table_rows_after = len(re.findall(r"(?m)^\s*\|.*\|\s*$", after))
    presentation_requires_table = bool(
        re.search(
            r"^####\s+(?:Presentation|呈现要求)\s*$.*?\btable\b|"
            r"^####\s+呈现要求\s*$.*?表格",
            section_spec,
            re.IGNORECASE | re.MULTILINE | re.DOTALL,
        )
    )
    if presentation_requires_table and table_rows_before and not table_rows_after:
        errors.append("required table structure was lost")

    keep_table = bool(
        re.search(
            r"(?mi)^\s*[-*]\s*(?:\*\*)?KEEP(?:\*\*)?\s*:.*(?:\btable\b|表格)",
            directives,
        )
    )
    before_tables = _markdown_table_count(before)
    after_tables = _markdown_table_count(after)
    if keep_table and after_tables < before_tables:
        errors.append(
            f"KEEP directive lost Markdown tables: {before_tables} -> {after_tables}"
        )
    errors.extend(_table_structure_regression_errors(before, after, directives))
    return errors
