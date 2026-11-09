"""Локальная защита develop: хук pre-push не пускает push и удаление защищённых веток."""
import os
import shutil
import subprocess
from pathlib import Path

import pytest

HOOK = Path(__file__).resolve().parent.parent / "tools" / "hooks" / "pre-push"
SHA, ZERO = "1" * 40, "0" * 40

pytestmark = pytest.mark.skipif(shutil.which("sh") is None, reason="needs sh")


def push(remote_ref, local_sha=SHA, **env):
    line = f"refs/heads/x {local_sha} {remote_ref} {SHA}\n"
    return subprocess.run(["sh", str(HOOK), "origin", "url"], input=line, text=True, capture_output=True, check=False,
                          env={**os.environ, **env})


def test_feature_branch_is_allowed():
    assert push("refs/heads/feat/x").returncode == 0


@pytest.mark.parametrize("ref", ["refs/heads/develop", "refs/heads/main"])
def test_protected_branch_is_rejected(ref):
    r = push(ref)
    assert r.returncode == 1 and "open a PR" in r.stderr


def test_delete_is_rejected():
    r = push("refs/heads/develop", local_sha=ZERO)
    assert r.returncode == 1 and "deleting" in r.stderr


def test_similar_name_is_not_protected():
    assert push("refs/heads/develop-old").returncode == 0


def test_override():
    assert push("refs/heads/develop", ALLOW_PROTECTED_PUSH="1").returncode == 0
