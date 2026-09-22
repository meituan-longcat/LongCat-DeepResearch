from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from longcat_deepresearch import runtime
from longcat_deepresearch.backend import backend_scope


class FakeBackend:
    def __init__(self) -> None:
        self.search_requests: list[dict] = []
        self.fetch_requests: list[dict] = []

    def llm(self, request: dict) -> dict:
        raise AssertionError("LLM is not used in these tests")

    def web_search(self, request: dict) -> dict:
        self.search_requests.append(request)
        return {
            "results": [
                {
                    "title": "Primary source",
                    "url": "https://example.org/source",
                    "snippet": "  useful   evidence  ",
                    "published_at": "2026-01-01",
                }
            ]
        }

    def web_fetch(self, request: dict) -> dict:
        self.fetch_requests.append(request)
        return {
            "pages": [
                {
                    "url": url,
                    "title": f"Page {index}",
                    "text": "body ![image](https://images.example/a.png)",
                    "error": None,
                }
                for index, url in enumerate(request["urls"], start=1)
            ]
        }


class WebToolTests(unittest.TestCase):
    def test_tool_names_are_stable(self) -> None:
        self.assertEqual(
            [tool["function"]["name"] for tool in runtime.TOOLS],
            ["web_search", "web_fetch"],
        )
        self.assertEqual(
            runtime.TOOLS[1]["function"]["parameters"]["required"],
            ["urls"],
        )

    def test_web_search_normalizes_results_for_the_model(self) -> None:
        backend = FakeBackend()
        with backend_scope(backend):
            result = runtime._search_web("query", 20)
        self.assertEqual(backend.search_requests, [{"query": "query", "top_k": 10}])
        self.assertIn("Primary source", result)
        self.assertIn("https://example.org/source", result)
        self.assertIn("Snippet: useful evidence", result)

    def test_web_fetch_validates_order_cleans_images_and_caches(self) -> None:
        backend = FakeBackend()
        urls = ["https://example.org/a", "https://example.org/b"]
        with tempfile.TemporaryDirectory() as raw, backend_scope(backend):
            runtime._activate_web_fetch_cache(Path(raw))
            first = json.loads(runtime._web_fetch(urls))
            second = json.loads(runtime._web_fetch(urls))
        self.assertEqual(len(backend.fetch_requests), 1)
        self.assertEqual(first, second)
        self.assertEqual([page["url"] for page in first["pages"]], urls)
        self.assertEqual(first["pages"][0]["text"], "body ![](IMG_URL)")

    def test_web_fetch_rejects_invalid_input_before_backend(self) -> None:
        backend = FakeBackend()
        with backend_scope(backend):
            result = runtime._web_fetch(["https://example.org", "file:///etc/passwd"])
        self.assertIn("rejected", result.lower())
        self.assertEqual(backend.fetch_requests, [])

    def test_tool_dispatch_uses_web_fetch_name(self) -> None:
        backend = FakeBackend()
        call = {
            "id": "fetch-1",
            "function": {
                "name": "web_fetch",
                "arguments": json.dumps({"urls": ["https://example.org/a"]}),
            },
        }
        with tempfile.TemporaryDirectory() as raw, backend_scope(backend):
            runtime._activate_web_fetch_cache(Path(raw))
            call_id, result = runtime._run_tool_call(call)
        self.assertEqual(call_id, "fetch-1")
        self.assertIn("https://example.org/a", result)


if __name__ == "__main__":
    unittest.main()
