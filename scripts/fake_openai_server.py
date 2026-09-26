"""A minimal OpenAI-compatible /chat/completions server for local end-to-end runs.

Used by scripts/gauntlet.sh to exercise router.py over real HTTP, without
touching a paid endpoint.

    python scripts/fake_openai_server.py 8099
"""

from __future__ import annotations

import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

ROUTER_MODEL = "router-model"


def task_of(prompt: str) -> str:
    for line in prompt.splitlines():
        if line.startswith("Task: "):
            return line.removeprefix("Task: ")
    return prompt


def answer_for(prompt: str, model: str, port: int) -> str:
    task = task_of(prompt)
    if model == ROUTER_MODEL:
        return "B" if "architecture" in task.lower() else "A"
    return f"[{model}@{port}] answer for: {task}"


def top_logprobs(chosen: str) -> list[dict[str, Any]]:
    other = "B" if chosen == "A" else "A"

    return [
        {"token": chosen, "logprob": -0.2, "bytes": None},
        {"token": other, "logprob": -3.5, "bytes": None},
    ]


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def do_POST(self) -> None:
        if self.path != "/v1/chat/completions":
            self.send_error(404, "unknown path")
            return

        length = int(self.headers.get("Content-Length", "0"))
        body = json.loads(self.rfile.read(length) or b"{}")
        model = body.get("model", "")
        prompt = "".join(
            str(part.get("content", "")) for part in body.get("messages", [])
        )
        port = int(self.server.server_address[1])
        content = answer_for(prompt, model, port)
        want_logprobs = bool(body.get("logprobs")) and model == ROUTER_MODEL

        payload = json.dumps(self.payload(model, content, want_logprobs)).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def payload(self, model: str, content: str, want_logprobs: bool) -> dict[str, Any]:
        return {
            "id": "chatcmpl-fake",
            "object": "chat.completion",
            "created": 0,
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": content},
                    "finish_reason": "stop",
                    "logprobs": (
                        {
                            "content": [
                                {
                                    "token": content,
                                    "logprob": -0.2,
                                    "top_logprobs": top_logprobs(
                                        content.strip() or "A"
                                    ),
                                }
                            ]
                        }
                        if want_logprobs
                        else None
                    ),
                }
            ],
        }

    def log_message(self, format: str, *args: Any) -> None:
        pass


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8099
    ThreadingHTTPServer(("127.0.0.1", port), Handler).serve_forever()
