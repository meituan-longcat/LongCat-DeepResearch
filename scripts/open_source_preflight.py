#!/usr/bin/env python3
"""Fail closed on common open-source release hygiene mistakes."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SELF = Path(__file__).resolve()
REQUIRED_FILES = {
    "README.md",
    "README_zh-CN.md",
    "LICENSE",
    "technical_report/LongCat-DeepResearch.pdf",
    "pyproject.toml",
    ".gitignore",
    "assets/benchmark_overview.png",
    "assets/longcat_logo.png",
    "assets/harness_overview.png",
    "assets/data_construction.png",
}
MAX_FILE_BYTES = 5 * 1024 * 1024
APPROVED_PNGS = {
    "assets/benchmark_overview.png",
    "assets/longcat_logo.png",
    "assets/harness_overview.png",
    "assets/data_construction.png",
}
APPROVED_PDFS = {"technical_report/LongCat-DeepResearch.pdf"}

# Concatenation keeps the scanner from reporting its own rule definitions.
FORBIDDEN_LITERALS = (
    "/Users" + "/",
    "/mnt/" + "hdfs/",
    "/mnt/" + "dolphinfs/",
)
SECRET_PATTERNS = (
    ("private key", re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----")),
    ("GitHub token", re.compile(r"\bgh[opsu]_[A-Za-z0-9]{30,}\b")),
    ("AWS access key", re.compile(r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b")),
    ("OpenAI-style key", re.compile(r"\bsk-[A-Za-z0-9_-]{20,}\b")),
    (
        "credential in URL",
        re.compile(r"https?://[^\s/:]+:[^\s/@]+@", re.IGNORECASE),
    ),
)


def release_paths(root: Path = ROOT) -> list[Path]:
    command = [
        "git",
        "ls-files",
        "--cached",
        "--others",
        "--exclude-standard",
        "-z",
    ]
    payload = subprocess.check_output(command, cwd=root)
    return [
        root / item.decode("utf-8")
        for item in payload.split(b"\0")
        if item and (root / item.decode("utf-8")).is_file()
    ]


def scan_text(relative: str, text: str) -> list[str]:
    findings: list[str] = []
    for literal in FORBIDDEN_LITERALS:
        if literal.lower() in text.lower():
            findings.append(f"{relative}: contains forbidden release literal {literal!r}")
    for label, pattern in SECRET_PATTERNS:
        if pattern.search(text):
            findings.append(f"{relative}: contains a possible {label}")
    return findings


def is_approved_binary(relative: str, payload: bytes) -> bool:
    return (
        relative in APPROVED_PNGS and payload.startswith(b"\x89PNG\r\n\x1a\n")
    ) or (relative in APPROVED_PDFS and payload.startswith(b"%PDF-"))


def scan_repository(root: Path = ROOT) -> list[str]:
    findings: list[str] = []
    existing = {path.relative_to(root).as_posix() for path in release_paths(root)}
    missing = sorted(REQUIRED_FILES - existing)
    findings.extend(f"missing required release file: {name}" for name in missing)

    for path in release_paths(root):
        relative = path.relative_to(root).as_posix()
        if path.is_symlink():
            findings.append(f"{relative}: symbolic links are not allowed in the release")
            continue
        size = path.stat().st_size
        if size > MAX_FILE_BYTES:
            findings.append(f"{relative}: file is larger than {MAX_FILE_BYTES} bytes")
            continue
        payload = path.read_bytes()
        if b"\0" in payload:
            if not is_approved_binary(relative, payload):
                findings.append(f"{relative}: unexpected binary file")
            continue
        try:
            text = payload.decode("utf-8")
        except UnicodeDecodeError:
            findings.append(f"{relative}: file is not UTF-8 text")
            continue
        if path.resolve() != SELF:
            findings.extend(scan_text(relative, text))

    tracked_env = sorted(
        name for name in existing if (Path(name).name == ".env" or Path(name).suffix == ".pem")
    )
    findings.extend(f"{name}: sensitive file type must not be tracked" for name in tracked_env)
    return findings


def main() -> int:
    findings = scan_repository()
    if findings:
        print("open-source preflight failed:")
        for finding in findings:
            print(f"- {finding}")
        return 1
    print(f"open-source preflight passed ({len(release_paths())} files checked)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
