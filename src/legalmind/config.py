"""Typed loader for the YAML training configs in `configs/`.

Every training knob lives in YAML, not in argparse flags. Two reasons: the
config file is what gets logged to W&B as the run's config (so a run is
reproducible from its own record), and the CI smoke test executes the exact
same code path as the production run with a different file.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class ModelConfig:
    name: str
    enable_thinking: bool = False
    attn_implementation: str = "eager"
    torch_dtype: str = "bfloat16"


@dataclass(frozen=True)
class QuantizationConfig:
    load_in_4bit: bool = False
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_use_double_quant: bool = True
    bnb_4bit_compute_dtype: str = "bfloat16"


@dataclass(frozen=True)
class LoraConfig:
    r: int = 16
    alpha: int = 32
    dropout: float = 0.05
    target_modules: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class DataConfig:
    train_path: str
    val_path: str
    max_seq_length: int = 2048
    packing: bool = True
    # Named for the actual TRL 0.18 knob. TRL has no `assistant_only_loss` at
    # this version; `completion_only_loss` is the equivalent and requires a
    # prompt/completion dataset, which is what `train/sft.py` builds.
    completion_only_loss: bool = True


@dataclass(frozen=True)
class TrackingConfig:
    report_to: str = "none"
    run_name: str = "legalmind"


@dataclass(frozen=True)
class TrainConfig:
    model: ModelConfig
    quantization: QuantizationConfig
    lora: LoraConfig
    data: DataConfig
    tracking: TrackingConfig
    # Passed through to `transformers.TrainingArguments` verbatim. Keeping this
    # as a dict rather than a dataclass means new HF arguments can be used from
    # YAML without touching this file.
    training: dict[str, Any]

    @property
    def output_dir(self) -> str:
        return str(self.training["output_dir"])


def load_train_config(path: str | Path) -> TrainConfig:
    """Parse a training YAML into a `TrainConfig`.

    Raises `KeyError` on a missing required section rather than silently
    defaulting — a config that omits `model` or `data` is a mistake, not a
    request for defaults.
    """
    raw = yaml.safe_load(Path(path).read_text())
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: expected a YAML mapping at the top level")

    for required in ("model", "data", "training"):
        if required not in raw:
            raise KeyError(f"{path}: missing required section '{required}'")

    return TrainConfig(
        model=ModelConfig(**raw["model"]),
        quantization=QuantizationConfig(**raw.get("quantization", {})),
        lora=LoraConfig(**raw.get("lora", {})),
        data=DataConfig(**raw["data"]),
        tracking=TrackingConfig(**raw.get("tracking", {})),
        training=dict(raw["training"]),
    )
