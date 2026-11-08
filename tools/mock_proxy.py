"""Мок LLM-прокси (OpenAI-совместимый /chat/completions) для локальных прогонов без ключа.

Не решает задачи «по-настоящему»: отвечает заготовками, чтобы проверить сантехнику
харнесса (вызовы, учёт токенов, реакцию на исчерпание бюджета).

    python3 tools/mock_proxy.py --port 8765 --budget 5000 [--exhaust-mode reset]
    LLM_BASE_URL=http://127.0.0.1:8765/v1 LLM_API_KEY=x python3 run_agent.py --tasks-dir ...
"""
import argparse
import json
import re
import socket
import struct
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

LOCK = threading.Lock()
STATE = {"spent": 0.0, "budget": None, "calls": 0, "inflight": 0, "max_conc": None, "rejected": 0,
         "exhaust_mode": "http429", "after_exhausted": 0}


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

    def exhausted(self):
        """Что прокси делает после конца бюджета: организаторы этого не сообщили, проверяем всё."""
        STATE["after_exhausted"] += 1
        mode = STATE["exhaust_mode"]
        if mode == "hang":
            time.sleep(3600)
        elif mode == "reset":
            self.connection.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, struct.pack("ii", 1, 0))
            self.connection.close()
            return
        code, text = {"http429": (429, b'{"error":{"message":"token budget exhausted"}}'),
                      "silent429": (429, b'{"error":{"message":"limit reached"}}'),
                      "http402": (402, b'{"error":{"message":"payment required"}}'),
                      "http500": (500, b'{"error":{"message":"internal error"}}'),
                      "http403": (403, b'{"error":{"message":"forbidden"}}')}[mode]
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(text)))
        self.end_headers()
        self.wfile.write(text)

    def do_POST(self):
        n = int(self.headers.get("Content-Length", "0"))
        req = json.loads(self.rfile.read(n) or b"{}")
        msgs = req.get("messages", [])
        prompt = "\n".join(str(m.get("content", "")) for m in msgs)
        with LOCK:
            if STATE["max_conc"] and STATE["inflight"] >= STATE["max_conc"]:
                STATE["rejected"] += 1
                self.send_response(429)
                self.end_headers()
                self.wfile.write("Слишком много одновременных запросов с аккаунта".encode())
                return
            STATE["inflight"] += 1
        try:
            time.sleep(0.3)  # имитация задержки модели
            text = fake_answer(prompt)
        finally:
            with LOCK:
                STATE["inflight"] -= 1
        pt, ct = max(1, len(prompt) // 4), max(1, len(text) // 4)
        with LOCK:
            cost = ct + pt / 10
            over = STATE["budget"] is not None and STATE["spent"] + cost > STATE["budget"]
            if not over:
                STATE["spent"] += cost
                STATE["calls"] += 1
        if over:
            self.exhausted()
            return
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
    ap.add_argument("--max-concurrent", type=int, default=None)
    ap.add_argument("--exhaust-mode", default="http429",
                    choices=["http429", "silent429", "http402", "http500", "http403", "hang", "reset"])
    a = ap.parse_args()
    STATE["budget"] = a.budget
    STATE["max_conc"] = a.max_concurrent
    STATE["exhaust_mode"] = a.exhaust_mode
    ThreadingHTTPServer(("127.0.0.1", a.port), H).serve_forever()
