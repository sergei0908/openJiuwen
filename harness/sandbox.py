"""Запуск кода и тестов в подпроцессе с таймаутом."""
import os
import re
import subprocess
import sys
import tempfile

PY = sys.executable or "python3"
_ENV = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", PYTHONIOENCODING="utf-8")
# Ключ прокси не нужен коду, который пишет модель.
_ENV.pop("LLM_API_KEY", None)


def run(cmd, cwd, timeout=20, stdin=None):
    try:
        p = subprocess.run(cmd, cwd=str(cwd), capture_output=True, text=True, check=False,
                           timeout=timeout, env=_ENV, input=stdin)
        return p.returncode, p.stdout, p.stderr
    except subprocess.TimeoutExpired as e:
        return -9, (e.stdout or "") if isinstance(e.stdout, str) else "", "TIMEOUT"
    except Exception as e:  # noqa: BLE001
        return -1, "", str(e)


def run_python(code, cwd, timeout=20):
    fd, path = tempfile.mkstemp(suffix=".py", dir=os.environ.get("TMPDIR", "/tmp"))
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(code)
    try:
        return run([PY, path], cwd, timeout)
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass


CODE_RE = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.DOTALL)


def extract_code(text):
    blocks = CODE_RE.findall(text or "")
    if blocks:
        return max(blocks, key=len)
    t = (text or "").strip()
    if t.startswith("```"):
        t = t.strip("`")
        t = t.split("\n", 1)[1] if "\n" in t else t
    return t


def last_line(s):
    lines = [l.strip() for l in (s or "").strip().splitlines() if l.strip()]
    return lines[-1] if lines else ""
