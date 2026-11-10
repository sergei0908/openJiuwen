"""Мок-прокси умеет быть медленным: задержка ответа и зависание на выбранных запросах."""
import argparse
import importlib.util
import json
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("mock_proxy", ROOT / "tools" / "mock_proxy.py")
mock_proxy = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mock_proxy)


def test_parse_delay():
    assert mock_proxy.parse_delay("5") == (5.0, 5.0)
    assert mock_proxy.parse_delay("5-10") == (5.0, 10.0)
    for bad in ("10-5", "-1"):
        with pytest.raises((argparse.ArgumentTypeError, ValueError)):
            mock_proxy.parse_delay(bad)


def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _post(port):
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", method="POST",
                                 data=json.dumps({"messages": [{"role": "user", "content": "hi"}]}).encode(),
                                 headers={"Content-Type": "application/json"})
    t = time.time()
    with urllib.request.urlopen(req, timeout=10) as r:
        r.read()
    return time.time() - t


def test_delay_and_hang():
    port = _free_port()
    proc = subprocess.Popen([sys.executable, str(ROOT / "tools" / "mock_proxy.py"), "--port", str(port),
                             "--delay", "0.4", "--hang-calls", "2", "--hang-s", "1.5"],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        for _ in range(50):
            try:
                socket.create_connection(("127.0.0.1", port), timeout=0.2).close()
                break
            except OSError:
                time.sleep(0.1)
        first, hung, third = _post(port), _post(port), _post(port)
        assert 0.4 <= first < 1.2 and 0.4 <= third < 1.2
        assert hung >= 1.5
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/", timeout=5) as r:
            state = json.loads(r.read())
        assert state["received"] == 3 and state["hung"] == 1
    finally:
        proc.terminate()
        proc.wait(5)
