"""Регламент (встреча 9.11): в рамках одной задачи используется одна модель - эскалации по умолчанию нет."""
import importlib


def test_reasoning_escalation_is_off_by_default(monkeypatch):
    monkeypatch.delenv("ALLOW_REASONING", raising=False)
    from harness import config
    try:
        assert importlib.reload(config).ALLOW_REASONING is False
    finally:
        importlib.reload(config)


def test_can_be_enabled_explicitly(monkeypatch):
    monkeypatch.setenv("ALLOW_REASONING", "1")
    from harness import config
    try:
        assert importlib.reload(config).ALLOW_REASONING is True
    finally:
        monkeypatch.delenv("ALLOW_REASONING")
        importlib.reload(config)
