"""The shipped configs must actually parse — a typo in YAML should fail in CI,
not three hours into a Spot instance."""

from __future__ import annotations

from pathlib import Path

import pytest

from legalmind.config import load_train_config

CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"


@pytest.mark.parametrize("name", ["train_qwen3_8b.yaml", "train_smoke_0.6b.yaml"])
def test_shipped_configs_parse(name: str) -> None:
    cfg = load_train_config(CONFIG_DIR / name)
    assert cfg.model.name.startswith("Qwen/Qwen3")
    assert cfg.data.max_seq_length > 0
    assert cfg.output_dir


def test_completion_only_masking_is_on_everywhere() -> None:
    """Training on prompt tokens is the silent mistake this project is explicit
    about avoiding. If a config ever turns it off, that must be deliberate."""
    for name in ("train_qwen3_8b.yaml", "train_smoke_0.6b.yaml"):
        cfg = load_train_config(CONFIG_DIR / name)
        assert cfg.data.completion_only_loss, f"{name} trains on prompt tokens"


def test_thinking_disabled_everywhere() -> None:
    """Qwen3's generation prompt differs by thinking mode: with it off, the
    prompt already ends with an empty `<think></think>` block. Train and serve
    must render identically, so both configs pin it off."""
    for name in ("train_qwen3_8b.yaml", "train_smoke_0.6b.yaml"):
        cfg = load_train_config(CONFIG_DIR / name)
        assert cfg.model.enable_thinking is False


def test_smoke_config_uses_committed_fixtures() -> None:
    """CI runs the smoke config with no network and no API key, so its data has
    to be in the repo."""
    cfg = load_train_config(CONFIG_DIR / "train_smoke_0.6b.yaml")
    root = CONFIG_DIR.parent
    assert (root / cfg.data.train_path).exists()
    assert (root / cfg.data.val_path).exists()


def test_smoke_config_needs_no_secrets() -> None:
    """CI runs this config with no W&B key available."""
    cfg = load_train_config(CONFIG_DIR / "train_smoke_0.6b.yaml")
    assert cfg.tracking.report_to == "none"
    assert cfg.quantization.load_in_4bit is False  # bitsandbytes has no macOS/CI wheel


def test_missing_required_section_raises(tmp_path: Path) -> None:
    bad = tmp_path / "bad.yaml"
    bad.write_text("model:\n  name: Qwen/Qwen3-8B\n")
    with pytest.raises(KeyError, match="data"):
        load_train_config(bad)
