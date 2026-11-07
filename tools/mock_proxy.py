"""Мок LLM-прокси (OpenAI-совместимый /chat/completions) для локальных прогонов без ключа.

Не решает задачи «по-настоящему»: отвечает заготовками, чтобы проверить сантехнику
харнесса (вызовы, учёт токенов, реакцию на исчерпание бюджета).

    python3 tools/mock_proxy.py --port 8765 --budget 5000
    LLM_BASE_URL=http://127.0.0.1:8765/v1 LLM_API_KEY=x python3 run_agent.py --tasks-dir ...
"""
import argparse
import json
import re
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

LOCK = threading.Lock()
STATE = {"spent": 0.0, "budget": None, "calls": 0}


def fake_answer(prompt: str) -> str:
    # Для coding-промпта возвращаем исходный файл без изменений (тесты не пройдут),
    # для остальных - программу, печатающую 42.
    if "mylib.py" in prompt and "```python" in prompt:
        m = re.search(r"```python\n(.*?)```", prompt, re.DOTALL)
        body = m.group(1) if m else "pass\n"
        return "```python\n" + body + "```"
    return "```python\nprint(42)\n```"


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_POST(self):
        n = int(self.headers.get("Content-Length", "0"))
        req = json.loads(self.rfile.read(n) or b"{}")
        msgs = req.get("messages", [])
        prompt = "\n".join(str(m.get("content", "")) for m in msgs)
        text = fake_answer(prompt)
        pt, ct = max(1, len(prompt) // 4), max(1, len(text) // 4)
        with LOCK:
            cost = ct + pt / 10
            if STATE["budget"] is not None and STATE["spent"] + cost > STATE["budget"]:
                self.send_response(429)
                self.end_headers()
                self.wfile.write(b'{"error":{"message":"token budget exhausted"}}')
                return
            STATE["spent"] += cost
            STATE["calls"] += 1
        body = {
            "id": "mock", "object": "chat.completion", "model": req.get("model", "mock"),
            "choices": [{"index": 0, "finish_reason": "stop",
                         "message": {"role": "assistant", "content": text}}],
            "usage": {"prompt_tokens": pt, "completion_tokens": ct, "total_tokens": pt + ct},
        }
        if req.get("stream"):
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            chunk = dict(body, object="chat.completion.chunk",
                         choices=[{"index": 0, "delta": {"role": "assistant", "content": text},
                                   "finish_reason": "stop"}])
            self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
            self.wfile.write(b"data: [DONE]\n\n")
            return
        raw = json.dumps(body).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self):
        raw = json.dumps(STATE).encode()
        self.send_response(200)
        self.end_headers()
        self.wfile.write(raw)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--budget", type=float, default=None)
    a = ap.parse_args()
    STATE["budget"] = a.budget
    ThreadingHTTPServer(("127.0.0.1", a.port), H).serve_forever()
