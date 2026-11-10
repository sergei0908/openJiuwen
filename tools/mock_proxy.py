"""Мок LLM-прокси (OpenAI-совместимый /chat/completions) для локальных прогонов без ключа.

Не решает задачи «по-настоящему»: отвечает заготовками, чтобы проверить сантехнику
харнесса (вызовы, учёт токенов, реакцию на исчерпание бюджета).

    python3 tools/mock_proxy.py --port 8765 --budget 5000 [--exhaust-mode reset]
    python3 tools/mock_proxy.py --delay 5-10 --hang-calls 7 --hang-s 200   # медленный прокси
    LLM_BASE_URL=http://127.0.0.1:8765/v1 LLM_API_KEY=x python3 run_agent.py --tasks-dir ...
"""
import argparse
import json
import random
import re
import socket
import struct
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

LOCK = threading.Lock()
STATE = {"spent": 0.0, "budget": None, "calls": 0, "inflight": 0, "max_conc": None, "rejected": 0,
         "exhaust_mode": "http429", "after_exhausted": 0, "delay": (0.3, 0.3), "hang_calls": set(),
         "hang_s": 200.0, "received": 0, "hung": 0}


def parse_delay(text):
    """«5» - ровно 5 с, «5-10» - случайно от 5 до 10 с."""
    lo, _, hi = str(text).partition("-")
    lo = float(lo)
    hi = float(hi) if hi else lo
    if lo < 0 or hi < lo:
        raise argparse.ArgumentTypeError(f"bad delay: {text}")
    return lo, hi


def response_delay(n):
    """Сколько ждать перед ответом на n-й полученный запрос: зависание или обычная задержка модели."""
    if n in STATE["hang_calls"]:
        STATE["hung"] += 1
        return STATE["hang_s"]
    lo, hi = STATE["delay"]
    return random.uniform(lo, hi)


def fake_answer(req):
    """Имитирует агентный ответ: сначала вызов инструмента, потом 'FINAL: <вывод инструмента>'.

    Возвращает (content, tool_calls). Задачи «по-настоящему» не решает: python печатает 42,
    update_code присылает первую функцию из промпта без изменений (тесты не пройдут) - цель
    в проверке цикла агента, учёта токенов и лимитов.
    """
    msgs = req.get("messages", [])
    tools = [t.get("function", {}).get("name") for t in (req.get("tools") or [])]
    if any(m.get("role") == "tool" for m in msgs):
        out = str([m for m in msgs if m.get("role") == "tool"][-1].get("content", "")).strip().splitlines()
        return "FINAL: " + (re.sub(r"^FINAL:\s*", "", out[0]) if out else "?"), None
    if "python" in tools:
        # если системный промпт просит печатать ответ скриптом - печатаем FINAL, как настоящая модель
        code = "print('FINAL:', 42)" if "print('FINAL:'" in str(msgs[0].get("content", "")) else "print(42)"
        return "", [_call("python", {"code": code})]
    if "update_code" in tools:
        prompt = "\n".join(str(m.get("content", "")) for m in msgs)
        m = re.search(r"```python\n(def .*?)(?:\n\S|```)", prompt, re.DOTALL)
        return "", [_call("update_code", {"code": (m.group(1) if m else "def f():\n    pass") + "\n"})]
    return "FINAL: 42", None


def _call(name, args):
    return {"id": "call_1", "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}


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
        prompt = "\n".join(str(m.get("content", "")) for m in msgs) + json.dumps(req.get("tools") or [])
        with LOCK:
            if STATE["max_conc"] and STATE["inflight"] >= STATE["max_conc"]:
                STATE["rejected"] += 1
                self.send_response(429)
                self.end_headers()
                self.wfile.write("Слишком много одновременных запросов с аккаунта".encode())
                return
            STATE["inflight"] += 1
            STATE["received"] += 1
            wait = response_delay(STATE["received"])
        try:
            time.sleep(wait)  # задержка модели или зависание
            text, tool_calls = fake_answer(req)
        finally:
            with LOCK:
                STATE["inflight"] -= 1
        pt, ct = max(1, len(prompt) // 4), max(1, (len(text) + len(json.dumps(tool_calls or []))) // 4)
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
            "choices": [{"index": 0, "finish_reason": "tool_calls" if tool_calls else "stop",
                         "message": dict({"role": "assistant", "content": text},
                                         **({"tool_calls": tool_calls} if tool_calls else {}))}],
            "usage": {"prompt_tokens": pt, "completion_tokens": ct, "total_tokens": pt + ct},
        }
        if req.get("stream"):
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            chunk = dict(body, object="chat.completion.chunk",
                         choices=[{"index": 0, "delta": dict({"role": "assistant", "content": text},
                                   **({"tool_calls": [dict(tc, index=0) for tc in tool_calls]} if tool_calls else {})),
                                   "finish_reason": "tool_calls" if tool_calls else "stop"}])
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
        raw = json.dumps(STATE, default=sorted).encode()
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
    ap.add_argument("--delay", type=parse_delay, default=(0.3, 0.3),
                    help="задержка ответа в секундах: 5 или диапазон 5-10 (по умолчанию 0.3)")
    ap.add_argument("--hang-calls", type=int, nargs="*", default=[],
                    help="номера запросов (с 1), на которых прокси зависает на --hang-s секунд")
    ap.add_argument("--hang-s", type=float, default=200.0)
    a = ap.parse_args()
    STATE["delay"] = a.delay
    STATE["hang_calls"] = set(a.hang_calls)
    STATE["hang_s"] = a.hang_s
    STATE["budget"] = a.budget
    STATE["max_conc"] = a.max_concurrent
    STATE["exhaust_mode"] = a.exhaust_mode
    ThreadingHTTPServer(("127.0.0.1", a.port), H).serve_forever()
