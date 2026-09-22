from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "open_source_preflight.py"
SPEC = importlib.util.spec_from_file_location("open_source_preflight", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class OpenSourcePreflightTests(unittest.TestCase):
    def test_clean_text_passes(self) -> None:
        self.assertEqual(
            MODULE.scan_text("config.example", "API_KEY=\nhttps://example.org"),
            [],
        )

    def test_private_key_is_rejected(self) -> None:
        findings = MODULE.scan_text(
            "secret.pem",
            "-----BEGIN " + "PRIVATE KEY-----\nnot-a-real-key",
        )
        self.assertTrue(any("private key" in finding for finding in findings))

    def test_internal_location_is_rejected(self) -> None:
        findings = MODULE.scan_text("config.py", "/Users" + "/example/project")
        self.assertTrue(any("forbidden release literal" in finding for finding in findings))

    def test_only_named_png_assets_are_allowed(self) -> None:
        png = b"\x89PNG\r\n\x1a\n\x00"
        self.assertTrue(MODULE.is_approved_binary("assets/benchmark_overview.png", png))
        self.assertTrue(MODULE.is_approved_binary("assets/longcat_logo.png", png))
        self.assertFalse(MODULE.is_approved_binary("assets/extra.png", png))

    def test_only_named_pdf_is_allowed(self) -> None:
        pdf = b"%PDF-1.7\n"
        self.assertTrue(
            MODULE.is_approved_binary("technical_report/LongCat-DeepResearch.pdf", pdf)
        )
        self.assertFalse(MODULE.is_approved_binary("private-report.pdf", pdf))


if __name__ == "__main__":
    unittest.main()
