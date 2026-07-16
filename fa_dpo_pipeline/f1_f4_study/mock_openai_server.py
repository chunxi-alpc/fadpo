#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, List, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Tiny OpenAI-compatible mock server for end-to-end F1-F4 pipeline smoke tests."
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18080)
    parser.add_argument("--model-id", default="mock-responder")
    return parser.parse_args()


def extract_message_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return "\n".join(part for part in parts if part)
    return ""


def pick_first_option_label(prompt: str) -> str:
    matches = re.findall(r"(?m)^\s*([A-Z])\.\s+", prompt)
    return matches[0] if matches else "A"


def build_generation_content(prompt: str) -> str:
    if "\nOptions:\n" in prompt or "\nOptions:" in prompt:
        label = pick_first_option_label(prompt)
        return (
            "I review the question and options, then provide a placeholder choice so the "
            "evaluation pipeline can be tested end to end.\n\n"
            f"Final Answer: {label}"
        )
    return (
        "I review the clinical case and provide a placeholder diagnostic impression so the "
        "generation and judging pipeline can be tested end to end.\n\n"
        "Final Answer: Mock diagnosis"
    )


def build_judge_content() -> str:
    payload = {
        "dominant_error_type": "NONE",
        "has_f1": False,
        "has_f2": False,
        "has_f3": False,
        "has_f4": False,
        "single_dominant_error": False,
        "answer_supported_by_reasoning": True,
        "clinical_plausibility_score": 3,
        "confidence_score": 5,
        "short_rationale": "Mock judge response for pipeline smoke testing.",
    }
    return json.dumps(payload, ensure_ascii=False)


def classify_request(messages: List[Dict[str, Any]]) -> Tuple[str, str]:
    system_text = "\n".join(extract_message_text(msg.get("content")) for msg in messages if msg.get("role") == "system")
    user_text = "\n".join(extract_message_text(msg.get("content")) for msg in messages if msg.get("role") == "user")
    if "strict clinical reasoning auditor" in system_text.lower():
        return "judge", build_judge_content()
    return "generation", build_generation_content(user_text)


class MockHandler(BaseHTTPRequestHandler):
    server_version = "MockOpenAI/0.1"

    @property
    def model_id(self) -> str:
        return getattr(self.server, "model_id", "mock-responder")

    def _send_json(self, payload: Dict[str, Any], status: int = HTTPStatus.OK) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: Any) -> None:
        return

    def do_GET(self) -> None:
        if self.path.rstrip("/") == "/v1/models":
            self._send_json(
                {
                    "object": "list",
                    "data": [
                        {
                            "id": self.model_id,
                            "object": "model",
                            "created": int(time.time()),
                            "owned_by": "mock",
                        }
                    ],
                }
            )
            return
        self._send_json({"error": {"message": "Not found"}}, status=HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:
        if self.path.rstrip("/") != "/v1/chat/completions":
            self._send_json({"error": {"message": "Not found"}}, status=HTTPStatus.NOT_FOUND)
            return

        length = int(self.headers.get("Content-Length", "0") or 0)
        raw_body = self.rfile.read(length).decode("utf-8") if length > 0 else "{}"
        try:
            payload = json.loads(raw_body)
        except json.JSONDecodeError:
            self._send_json({"error": {"message": "Invalid JSON"}}, status=HTTPStatus.BAD_REQUEST)
            return

        messages = payload.get("messages")
        if not isinstance(messages, list):
            self._send_json({"error": {"message": "`messages` must be a list"}}, status=HTTPStatus.BAD_REQUEST)
            return

        _, content = classify_request(messages)
        response = {
            "id": f"chatcmpl-mock-{int(time.time() * 1000)}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": self.model_id,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": content},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
            },
        }
        self._send_json(response)


def main() -> None:
    args = parse_args()
    server = ThreadingHTTPServer((args.host, args.port), MockHandler)
    server.model_id = args.model_id
    print(f"Mock OpenAI server listening on http://{args.host}:{args.port}/v1")
    server.serve_forever()


if __name__ == "__main__":
    main()
