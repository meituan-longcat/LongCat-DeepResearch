# Backend interface

The harness imports no model, search, or crawling SDK. A user implementation provides all three external capabilities through `longcat_deepresearch.backend.HarnessBackend`.

A module named by `DR_BACKEND_MODULE` must export either `create_backend()` or `BACKEND`. The resulting object must implement `llm`, `web_search`, and `web_fetch`.

## LLM

### Input

`llm(request)` receives an OpenAI Chat Completions-compatible object:

```json
{
  "model": "model-name",
  "messages": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "..."}
  ],
  "stream": false,
  "max_tokens": 12000,
  "tools": [
    {
      "type": "function",
      "function": {
        "name": "web_search",
        "description": "...",
        "parameters": {"type": "object", "properties": {}}
      }
    }
  ],
  "tool_choice": "auto"
}
```

`tools` and `tool_choice` are omitted for tool-free calls. The backend may copy the object and add provider-specific fields, but it must not mutate the input object.

### Output

The method returns a full Chat Completions-compatible response:

```json
{
  "choices": [
    {
      "message": {
        "role": "assistant",
        "content": null,
        "tool_calls": [
          {
            "id": "call_1",
            "type": "function",
            "function": {
              "name": "web_search",
              "arguments": "{\"query\":\"example\",\"top_k\":5}"
            }
          }
        ]
      },
      "finish_reason": "tool_calls"
    }
  ],
  "usage": {
    "prompt_tokens": 100,
    "completion_tokens": 20,
    "total_tokens": 120
  }
}
```

For a terminal answer, `message.content` is a string and `tool_calls` is absent. `usage` is optional but recommended. Exceptions propagate to the caller; retry and timeout policy belongs to the backend.

## web_search

### Input

```json
{"query": "search terms", "top_k": 8}
```

`top_k` is clamped to the range 1–10.

### Output

```json
{
  "results": [
    {
      "title": "Source title",
      "url": "https://example.org/source",
      "snippet": "Short result summary",
      "published_at": "2026-01-01"
    }
  ]
}
```

`title`, `url`, and `snippet` are strings. `published_at` is a string or null. URLs must use HTTP or HTTPS.

## web_fetch

The model-facing tool name is always `web_fetch`. It supports a batch so one tool call can fetch several independent sources.

### Input

```json
{"urls": ["https://example.org/a", "https://example.org/b"]}
```

The list contains 1–10 HTTP(S) URLs.

### Output

```json
{
  "pages": [
    {
      "url": "https://example.org/a",
      "title": "Page A",
      "text": "Readable page text",
      "error": null
    },
    {
      "url": "https://example.org/b",
      "title": "",
      "text": "",
      "error": "not readable"
    }
  ]
}
```

The backend must return exactly one page per requested URL, in the same order. `url`, `title`, and `text` are strings; `error` is a string or null. The harness removes image destinations, truncates long text, and caches valid responses within the run.

## Concurrency and security

One backend instance is shared across concurrent planning and research workers. Implementations must therefore be thread-safe.

The backend is also the security boundary for network access. It must handle authentication, timeouts, retries, rate limits, redirects, DNS resolution, private/link-local address blocking, response-size limits, and service-specific logging. Retrieved content is untrusted and may contain prompt injection.
