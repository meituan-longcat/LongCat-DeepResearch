from __future__ import annotations

import json
import re
import tempfile
import threading
import unittest
from pathlib import Path

from longcat_deepresearch import LongCatDeepResearch
from longcat_deepresearch.prompts import (
    EDITORIAL_PLANNER_SYSTEM_PROMPT,
    LOCAL_EDITOR_SYSTEM_PROMPT,
    PLANNING_CRITIC_PROMPT,
    PLANNING_JUDGE_PROMPT,
    PLANNING_REVISER_PROMPT,
    PLANNING_SYSTEM_PROMPT,
    RESEARCHER_SYSTEM_PROMPT,
)


def planning_body(filename: str) -> str:
    sections = []
    for index in range(1, 7):
        sections.append(
            f"""### S1.{index}｜Part {index}
#### What to Cover
Cover part {index}.
#### Research Questions
- What is fact {index}?
#### Required Entities / Cases
- Entity {index}
#### Source Leads
- https://example.org/{index} — primary source
"""
        )
    return (
        f"<!-- FILE: {filename} -->\n# ReportSpec\n## S1｜Topic\n"
        + "\n".join(sections)
        + "<!-- END FILE -->"
    )


class ReplayBackend:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.requests: list[dict] = []

    def llm(self, request: dict) -> dict:
        with self.lock:
            self.requests.append(request)
        messages = request["messages"]
        system = messages[0]["content"]
        tool_results = [message for message in messages if message.get("role") == "tool"]

        if system == PLANNING_SYSTEM_PROMPT:
            message = (
                self.tool_calls("writer", 3)
                if not tool_results
                else {"role": "assistant", "content": planning_body("research_spec.write.md")}
            )
        elif system == PLANNING_JUDGE_PROMPT:
            message = (
                self.tool_calls("judge", 3, fetches=1)
                if not tool_results
                else {"role": "assistant", "content": planning_body("research_spec.write.md")}
            )
        elif system == PLANNING_CRITIC_PROMPT:
            message = (
                self.tool_calls("critic", 3, fetches=1)
                if not tool_results
                else {
                    "role": "assistant",
                    "content": "# ResearchSpec Critique\n\nNo material correction required.",
                }
            )
        elif system == PLANNING_REVISER_PROMPT:
            message = {
                "role": "assistant",
                "content": planning_body("research_spec.md"),
            }
        elif system == RESEARCHER_SYSTEM_PROMPT:
            matches = re.findall(r"### (S1\.\d+)｜Part (\d+)", messages[1]["content"])
            if not matches:
                raise AssertionError("researcher prompt lacks a section")
            section_id, number = matches[-1]
            message = {
                "role": "assistant",
                "content": (
                    f"<!-- SECTION -->\n### {section_id}｜Part {number}\n"
                    f"Evidence for part {number}.[source](https://example.org/{number})\n"
                    "<!-- END SECTION -->"
                ),
            }
        elif system == EDITORIAL_PLANNER_SYSTEM_PROMPT:
            message = {
                "role": "assistant",
                "content": (
                    "<!-- EDIT_PLAN -->\n# Global Editorial Plan\n"
                    "## S1.1\n- Retain the cited source.\n"
                    "<!-- END EDIT_PLAN -->"
                ),
            }
        elif system == LOCAL_EDITOR_SYSTEM_PROMPT:
            message = {
                "role": "assistant",
                "content": (
                    "<!-- SECTION -->\n### S1.1｜Part 1\n"
                    "Evidence for part 1.[source](https://example.org/1)\n"
                    "<!-- END SECTION -->"
                ),
            }
        else:
            raise AssertionError(f"unexpected system prompt: {system[:80]!r}")

        return {
            "choices": [{"message": message, "finish_reason": "stop"}],
            "usage": {
                "prompt_tokens": 1,
                "completion_tokens": 1,
                "total_tokens": 2,
            },
        }

    def web_search(self, request: dict) -> dict:
        return {
            "results": [
                {
                    "title": request["query"],
                    "url": "https://example.org/source",
                    "snippet": "fixed result",
                    "published_at": None,
                }
            ]
        }

    def web_fetch(self, request: dict) -> dict:
        return {
            "pages": [
                {
                    "url": url,
                    "title": "Source",
                    "text": "fixed page",
                    "error": None,
                }
                for url in request["urls"]
            ]
        }

    @staticmethod
    def tool_calls(prefix: str, searches: int, fetches: int = 0) -> dict:
        calls = [
            {
                "id": f"{prefix}-search-{index}",
                "type": "function",
                "function": {
                    "name": "web_search",
                    "arguments": json.dumps({"query": f"{prefix} {index}"}),
                },
            }
            for index in range(searches)
        ]
        calls.extend(
            {
                "id": f"{prefix}-fetch-{index}",
                "type": "function",
                "function": {
                    "name": "web_fetch",
                    "arguments": json.dumps({"urls": ["https://example.org/source"]}),
                },
            }
            for index in range(fetches)
        )
        return {"role": "assistant", "content": None, "tool_calls": calls}


class PipelineTests(unittest.TestCase):
    def test_full_pipeline_runs_with_only_the_backend_interface(self) -> None:
        backend = ReplayBackend()
        with tempfile.TemporaryDirectory() as raw:
            result = LongCatDeepResearch(backend=backend).run(
                "Compare six parts.",
                run_root=Path(raw),
            )
            report = result.final_path.read_text(encoding="utf-8")
        self.assertIn("Part 1", report)
        self.assertIn("https://example.org/1", report)
        self.assertGreater(len(backend.requests), 10)


if __name__ == "__main__":
    unittest.main()
