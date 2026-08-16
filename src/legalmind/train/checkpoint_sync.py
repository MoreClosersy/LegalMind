"""Mirror checkpoints to S3 so an interrupted run is resumable.

Set `LEGALMIND_S3_CHECKPOINT_URI` and every checkpoint the trainer writes is
pushed to object storage; on startup, the newest checkpoint found there is
pulled back down and training resumes from it.

The obvious motivation is Spot reclamation, which arrives with two minutes'
notice. But Spot is not the reason this is worth having. The instance can also
be stopped by an SSH drop, an OOM, a CUDA fault, a full disk, or somebody
closing a laptop — and the local disk of a terminated instance goes with it. A
run that only checkpoints locally is one unlucky event away from having spent
three hours of GPU time on nothing. This guards every one of those, not just
the one with a name.

Deliberately shelling out to the AWS CLI rather than taking a boto3 dependency.
`aws s3 sync` already does concurrent multipart transfer and skips unchanged
files, the CLI is present on any Deep Learning AMI, and the training image has
no reason to grow an SDK for one operation. A missing CLI is reported as a
configuration error at startup, not discovered at the first save.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments

ENV_VAR = "LEGALMIND_S3_CHECKPOINT_URI"
_CHECKPOINT_PREFIX = "checkpoint-"


def s3_uri_from_env() -> str | None:
    uri = os.getenv(ENV_VAR, "").strip().rstrip("/")
    return uri or None


def _require_aws_cli() -> str:
    aws = shutil.which("aws")
    if aws is None:
        raise RuntimeError(
            f"{ENV_VAR} is set but the AWS CLI is not on PATH. Checkpoint sync would "
            "fail silently at the first save, three hours into a run — refusing to "
            "start instead. Install the CLI or unset the variable."
        )
    return aws


def _run(argv: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(argv, capture_output=True, text=True, check=False)


def latest_remote_checkpoint(uri: str) -> str | None:
    """Newest `checkpoint-N` under `uri`, by step number.

    Sorted numerically, not lexically: `checkpoint-900` sorts after
    `checkpoint-1000` as a string, which would silently resume from an earlier
    step and quietly discard an hour of training.
    """
    aws = _require_aws_cli()
    result = _run([aws, "s3", "ls", f"{uri}/"])
    if result.returncode != 0:
        return None

    steps: list[int] = []
    for line in result.stdout.splitlines():
        name = line.split()[-1].rstrip("/")
        if name.startswith(_CHECKPOINT_PREFIX):
            suffix = name[len(_CHECKPOINT_PREFIX) :]
            if suffix.isdigit():
                steps.append(int(suffix))
    if not steps:
        return None
    return f"{_CHECKPOINT_PREFIX}{max(steps)}"


def pull_latest_checkpoint(uri: str, output_dir: Path) -> Path | None:
    """Download the newest remote checkpoint, returning its local path."""
    name = latest_remote_checkpoint(uri)
    if name is None:
        print(f"no checkpoint under {uri}; starting from scratch", file=sys.stderr)
        return None

    aws = _require_aws_cli()
    local = output_dir / name
    local.mkdir(parents=True, exist_ok=True)
    print(f"resuming: pulling {uri}/{name} -> {local}", file=sys.stderr)
    result = _run([aws, "s3", "sync", f"{uri}/{name}", str(local)])
    if result.returncode != 0:
        raise RuntimeError(f"failed to pull {uri}/{name}: {result.stderr.strip()}")
    return local


class S3CheckpointCallback(TrainerCallback):
    """Push each checkpoint to S3 as it is written.

    Runs on `on_save`, synchronously. That blocks training for the length of an
    upload, which for a LoRA adapter is a second or two — the alternative is a
    background upload that has not finished when the instance is reclaimed,
    which is the same as no upload at all while looking like one.
    """

    def __init__(self, uri: str) -> None:
        self.uri = uri.rstrip("/")
        self._aws = _require_aws_cli()
        self.uploads = 0
        self.failures = 0

    def on_save(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: object,
    ) -> None:
        if not args.output_dir:
            return
        local = Path(args.output_dir) / f"{_CHECKPOINT_PREFIX}{state.global_step}"
        if not local.is_dir():
            return

        result = _run([self._aws, "s3", "sync", str(local), f"{self.uri}/{local.name}"])
        if result.returncode == 0:
            self.uploads += 1
            print(f"synced {local.name} -> {self.uri}/{local.name}", file=sys.stderr)
            return

        # A failed upload is reported loudly but does not kill the run: local
        # checkpoints still exist, and losing the sync is strictly better than
        # losing the training. Silence is the thing to avoid — a run that
        # believes it is backed up and is not is worse than one that knows it
        # is not.
        self.failures += 1
        print(
            f"WARNING: checkpoint sync failed for {local.name} "
            f"({self.failures} failure(s) so far): {result.stderr.strip()}",
            file=sys.stderr,
        )
