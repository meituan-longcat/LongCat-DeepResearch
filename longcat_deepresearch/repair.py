from __future__ import annotations

import json
import re
from dataclasses import dataclass
from difflib import SequenceMatcher


PATCH_RE = re.compile(
    r"<!--\s*PATCHES\s*-->(.*?)<!--\s*END PATCHES\s*-->",
    re.IGNORECASE | re.DOTALL,
)


@dataclass(frozen=True)
class ExactPatch:
    patch_id: str
    old: str
    new: str
    expected_count: int


def apply_exact_patches(before: str, raw_patch: str, *, max_patches: int = 12) -> str:
    """Apply one fail-closed exact-match patch transaction."""
    match = PATCH_RE.search(str(raw_patch or ""))
    if not match:
        raise ValueError("repair output is missing PATCHES markers")
    try:
        payload = json.loads(match.group(1).strip())
    except json.JSONDecodeError as exc:
        raise ValueError(f"repair patch is not valid JSON: {exc}") from exc
    rows = payload.get("patches") if isinstance(payload, dict) else None
    if not isinstance(rows, list) or not rows or len(rows) > max_patches:
        raise ValueError(f"repair patch must contain 1..{max_patches} patches")
    patches: list[ExactPatch] = []
    seen_old: set[str] = set()
    for index, row in enumerate(rows, 1):
        if not isinstance(row, dict):
            raise ValueError(f"patch {index} is not an object")
        patch_id = str(row.get("id") or f"P{index}")
        old = str(row.get("old") or "")
        new = str(row.get("new") or "")
        if not old or old == new:
            raise ValueError(f"{patch_id} must make one non-empty exact replacement")
        if old in seen_old:
            raise ValueError(f"{patch_id} duplicates an earlier anchor")
        expected_count = int(row.get("expected_count") or 1)
        if expected_count < 1 or expected_count > 100:
            raise ValueError(f"{patch_id} expected_count must be between 1 and 100")
        if before.count(old) != expected_count:
            raise ValueError(
                f"{patch_id} anchor count does not match expected_count={expected_count}"
            )
        seen_old.add(old)
        patches.append(ExactPatch(patch_id, old, new, expected_count))
    result = before
    for patch in patches:
        if result.count(patch.old) != patch.expected_count:
            raise ValueError(f"{patch.patch_id} anchor changed before application")
        result = result.replace(patch.old, patch.new, patch.expected_count)
    if result == before:
        raise ValueError("repair patch made no change")
    max_growth = max(2048, len(before) // 5)
    if len(result) - len(before) > max_growth:
        raise ValueError("repair patch expands the artifact beyond the sparse-repair limit")
    similarity = SequenceMatcher(None, before, result, autojunk=False).ratio()
    if len(before) >= 1000 and similarity < 0.90:
        raise ValueError(
            f"repair patch changes too much content for a sparse transaction: {similarity:.4f}"
        )
    return result
