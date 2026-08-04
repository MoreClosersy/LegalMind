"""LegalMind — domain-adapted legal LLM.

Package layout:
    data/   synthetic instruction generation, filtering, decontamination
    train/  QLoRA SFT with completion-only loss masking
    eval/   three-arm comparison harness (base / base+prompt / fine-tuned)
    serve/  vLLM-backed gateway with a deterministic UPL compliance layer
"""

__version__ = "0.1.0"
