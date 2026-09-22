from __future__ import annotations

import sys
import types
import unittest

from longcat_deepresearch import runtime
from longcat_deepresearch.backend import (
    backend_scope,
    current_backend,
    load_backend,
    reset_default_backend,
)


class FakeBackend:
    def __init__(self) -> None:
        self.llm_requests: list[dict] = []

    def llm(self, request: dict) -> dict:
        self.llm_requests.append(request)
        return {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "done",
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 1,
                "completion_tokens": 1,
                "total_tokens": 2,
            },
        }

    def web_search(self, request: dict) -> dict:
        return {"results": []}

    def web_fetch(self, request: dict) -> dict:
        return {"pages": []}


class BackendInterfaceTests(unittest.TestCase):
    def test_llm_uses_openai_function_calling_contract(self) -> None:
        backend = FakeBackend()
        with backend_scope(backend):
            message = runtime._chat(
                [{"role": "user", "content": "hello"}],
                tools=True,
                max_tokens=123,
            )
        self.assertEqual(message, {"role": "assistant", "content": "done"})
        request = backend.llm_requests[0]
        self.assertEqual(request["model"], runtime.MODEL)
        self.assertEqual(request["messages"][0]["content"], "hello")
        self.assertEqual(request["max_tokens"], 123)
        self.assertFalse(request["stream"])
        self.assertEqual(
            [tool["function"]["name"] for tool in request["tools"]],
            ["web_search", "web_fetch"],
        )
        self.assertEqual(request["tool_choice"], "auto")

    def test_module_loader_accepts_factory(self) -> None:
        module_name = "_longcat_test_backend"
        module = types.ModuleType(module_name)
        expected = FakeBackend()
        module.create_backend = lambda: expected
        sys.modules[module_name] = module
        try:
            self.assertIs(load_backend(module_name), expected)
        finally:
            sys.modules.pop(module_name, None)
            reset_default_backend()

    def test_backend_scope_is_used_by_runtime(self) -> None:
        backend = FakeBackend()
        with backend_scope(backend):
            self.assertIs(current_backend(), backend)


if __name__ == "__main__":
    unittest.main()
