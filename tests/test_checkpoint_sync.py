"""Checkpoint-sync tests.

No AWS. What is tested is the logic that decides *which* checkpoint to resume
from and what happens when a sync fails — the two places where a bug costs
hours of GPU time and announces nothing.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

import pytest
from transformers import TrainerControl

from legalmind.train import checkpoint_sync
from legalmind.train.checkpoint_sync import (
    ENV_VAR,
    S3CheckpointCallback,
    latest_remote_checkpoint,
    pull_latest_checkpoint,
    s3_uri_from_env,
)


class FakeCompleted:
    def __init__(self, stdout: str = "", stderr: str = "", returncode: int = 0) -> None:
        self.stdout = stdout
        self.stderr = stderr
        self.returncode = returncode


@pytest.fixture(autouse=True)
def fake_aws_cli(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(checkpoint_sync.shutil, "which", lambda _: "/usr/bin/aws")


def test_env_var_is_optional_and_stripped(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(ENV_VAR, raising=False)
    assert s3_uri_from_env() is None
    monkeypatch.setenv(ENV_VAR, "  s3://bucket/run/  ")
    assert s3_uri_from_env() == "s3://bucket/run"
    monkeypatch.setenv(ENV_VAR, "   ")
    assert s3_uri_from_env() is None


def test_missing_aws_cli_fails_at_startup(monkeypatch: pytest.MonkeyPatch) -> None:
    """Refusing to start beats discovering at the first save that nothing is
    being backed up — by then an hour of GPU time is already spent."""
    monkeypatch.setattr(checkpoint_sync.shutil, "which", lambda _: None)
    with pytest.raises(RuntimeError, match="AWS CLI is not on PATH"):
        S3CheckpointCallback("s3://bucket/run")


LISTING = """\
                           PRE checkpoint-100/
                           PRE checkpoint-200/
                           PRE checkpoint-900/
                           PRE checkpoint-1000/
                           PRE some-other-dir/
"""


def test_latest_checkpoint_is_chosen_numerically(monkeypatch: pytest.MonkeyPatch) -> None:
    """The bug this exists to prevent: sorted as strings, "checkpoint-900" comes
    after "checkpoint-1000", so a resume would silently rewind 100 steps and
    throw away the training in between without erroring."""
    monkeypatch.setattr(checkpoint_sync, "_run", lambda argv: FakeCompleted(stdout=LISTING))
    assert latest_remote_checkpoint("s3://bucket/run") == "checkpoint-1000"


def test_non_checkpoint_directories_are_ignored(monkeypatch: pytest.MonkeyPatch) -> None:
    listing = "                           PRE logs/\n                           PRE wandb/\n"
    monkeypatch.setattr(checkpoint_sync, "_run", lambda argv: FakeCompleted(stdout=listing))
    assert latest_remote_checkpoint("s3://bucket/run") is None


def test_empty_prefix_means_start_from_scratch(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(checkpoint_sync, "_run", lambda argv: FakeCompleted(returncode=1))
    assert pull_latest_checkpoint("s3://bucket/run", tmp_path) is None


def test_pull_failure_raises_rather_than_starting_over(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A download that fails must not quietly fall through to a fresh run. That
    turns a recoverable interruption into a silent restart from step zero."""
    calls: list[list[str]] = []

    def fake_run(argv: list[str]) -> FakeCompleted:
        calls.append(argv)
        if argv[1:3] == ["s3", "ls"]:
            return FakeCompleted(stdout=LISTING)
        return FakeCompleted(stderr="connection reset", returncode=1)

    monkeypatch.setattr(checkpoint_sync, "_run", fake_run)
    with pytest.raises(RuntimeError, match="failed to pull"):
        pull_latest_checkpoint("s3://bucket/run", tmp_path)


def _args(output_dir: Path) -> Any:
    return type("Args", (), {"output_dir": str(output_dir)})()


def _state(step: int) -> Any:
    return type("State", (), {"global_step": step})()


def test_save_uploads_the_checkpoint_just_written(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    (tmp_path / "checkpoint-100").mkdir()
    calls: list[list[str]] = []

    def record(argv: list[str]) -> FakeCompleted:
        calls.append(argv)
        return FakeCompleted()

    monkeypatch.setattr(checkpoint_sync, "_run", record)

    callback = S3CheckpointCallback("s3://bucket/run")
    callback.on_save(_args(tmp_path), _state(100), TrainerControl())

    assert callback.uploads == 1
    assert calls[0][-2:] == [str(tmp_path / "checkpoint-100"), "s3://bucket/run/checkpoint-100"]


def test_sync_failure_warns_but_does_not_kill_the_run(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Losing the backup is strictly better than losing the training. What must
    not happen is silence — a run that believes it is backed up and is not is
    worse than one that knows it is not."""
    (tmp_path / "checkpoint-100").mkdir()
    monkeypatch.setattr(
        checkpoint_sync, "_run", lambda argv: FakeCompleted(stderr="AccessDenied", returncode=1)
    )

    callback = S3CheckpointCallback("s3://bucket/run")
    callback.on_save(_args(tmp_path), _state(100), TrainerControl())

    assert callback.failures == 1
    assert callback.uploads == 0
    assert "WARNING: checkpoint sync failed" in capsys.readouterr().err


def test_missing_local_checkpoint_is_a_no_op(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        checkpoint_sync,
        "_run",
        lambda argv: pytest.fail("must not shell out when there is nothing to upload"),
    )
    callback = S3CheckpointCallback("s3://bucket/run")
    callback.on_save(_args(tmp_path), _state(100), TrainerControl())
    assert callback.uploads == 0


def test_run_helper_does_not_raise_on_nonzero(monkeypatch: pytest.MonkeyPatch) -> None:
    """`check=False` is deliberate: every caller inspects returncode itself and
    decides whether the failure is fatal."""
    result = checkpoint_sync._run([__import__("sys").executable, "-c", "raise SystemExit(3)"])
    assert isinstance(result, subprocess.CompletedProcess)
    assert result.returncode == 3
