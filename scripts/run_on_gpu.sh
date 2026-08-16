#!/usr/bin/env bash
#
# Everything that has to happen on the GPU box, in the order it has to happen.
#
#   ./scripts/run_on_gpu.sh preflight   # checks only, no GPU time spent
#   ./scripts/run_on_gpu.sh train
#   ./scripts/run_on_gpu.sh serve
#   ./scripts/run_on_gpu.sh evaluate
#   ./scripts/run_on_gpu.sh all
#
# GPU time is billed by the hour and every failure mode below has cost somebody
# a run. The design rule for this script is that anything checkable without a
# GPU is checked in `preflight`, before the expensive part starts — a run that
# dies at hour three on a missing environment variable is the most avoidable
# way to waste money on this project.
#
# Recommended: run under tmux. An SSH drop otherwise takes the training with it.
#
#   tmux new -s legalmind
#   ./scripts/run_on_gpu.sh all 2>&1 | tee run.log

set -euo pipefail

CONFIG="${CONFIG:-configs/train_qwen3_8b.yaml}"
ADAPTER_DIR="${ADAPTER_DIR:-outputs/qwen3-8b-legalmind}"
BASE_MODEL="${BASE_MODEL:-Qwen/Qwen3-8B}"
VLLM_PORT="${VLLM_PORT:-8000}"
GATEWAY_PORT="${GATEWAY_PORT:-8080}"

log()  { printf '\n\033[1m== %s\033[0m\n' "$*"; }
fail() { printf '\033[31mFAIL: %s\033[0m\n' "$*" >&2; exit 1; }
ok()   { printf '  ok   %s\n' "$*"; }
warn() { printf '  warn %s\n' "$*"; }

preflight() {
  log "Preflight (no GPU time spent)"

  command -v nvidia-smi >/dev/null || fail "no nvidia-smi; this is not a GPU instance"
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader \
    | while read -r line; do ok "GPU: $line"; done

  # bf16 and FlashAttention-2 need Ampere (SM 8.0+). A T4 is SM 7.5 and will
  # fail deep inside the first forward pass with an unhelpful message, so it is
  # caught here instead. This is the concrete reason the config specifies g5
  # rather than the cheaper g4dn.
  local cap
  cap=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1)
  awk -v c="$cap" 'BEGIN { exit !(c+0 >= 8.0) }' \
    || fail "compute capability $cap < 8.0 (Ampere). bf16 and FlashAttention-2 need g5/g6, not g4dn"
  ok "compute capability $cap supports bf16 + FlashAttention-2"

  [ -f "$CONFIG" ] || fail "missing $CONFIG"
  ok "config $CONFIG"

  # The data is gitignored, so a fresh clone has none of it. Discovering that
  # after the model has loaded wastes the load.
  for f in data/train.jsonl data/val.jsonl; do
    [ -s "$f" ] || fail "missing $f — run the synthesis pipeline, or scp it over"
    ok "$(wc -l < "$f" | tr -d ' ') rows in $f"
  done

  if [ -n "${LEGALMIND_S3_CHECKPOINT_URI:-}" ]; then
    command -v aws >/dev/null || fail "LEGALMIND_S3_CHECKPOINT_URI is set but the AWS CLI is missing"
    aws s3 ls "${LEGALMIND_S3_CHECKPOINT_URI}/" >/dev/null 2>&1 \
      || fail "cannot list ${LEGALMIND_S3_CHECKPOINT_URI} — check the bucket and the instance role"
    ok "checkpoint sync -> ${LEGALMIND_S3_CHECKPOINT_URI}"
  else
    warn "LEGALMIND_S3_CHECKPOINT_URI unset: checkpoints stay on local disk only,"
    warn "     which the instance takes with it if it is terminated"
  fi

  if [ -n "${WANDB_API_KEY:-}" ]; then
    ok "W&B key present"
  else
    warn "WANDB_API_KEY unset: the run will train but log nowhere"
  fi

  # Disk, not an afterthought. Base weights ~16GB plus three checkpoints plus
  # the HF cache; a full disk mid-run corrupts the checkpoint being written.
  local avail
  avail=$(df -BG --output=avail . | tail -1 | tr -dc '0-9')
  [ "$avail" -ge 60 ] || fail "only ${avail}GB free; want 60GB+ for weights, cache, and checkpoints"
  ok "${avail}GB free disk"

  # The cheapest possible proof that the training path works end to end. Four
  # minutes on CPU against three hours of GPU time is a trade worth making
  # every single run, not just the first.
  log "Smoke test: 10 steps on the committed fixtures"
  uv run python -m legalmind.train.sft --config configs/train_smoke_0.6b.yaml
  uv run python scripts/verify_masking.py --config configs/train_smoke_0.6b.yaml
  ok "training path verified before spending GPU time on it"
}

train() {
  log "Training: QLoRA on $BASE_MODEL"
  uv run python -m legalmind.train.sft --config "$CONFIG"
  [ -f "$ADAPTER_DIR/adapter_model.safetensors" ] \
    || fail "training finished but no adapter at $ADAPTER_DIR"
  ok "adapter at $ADAPTER_DIR"
}

serve() {
  log "Serving: vLLM + gateway"
  [ -d "$ADAPTER_DIR" ] || fail "no adapter at $ADAPTER_DIR — train first"

  # One process, base weights and the adapter both addressable by model name.
  # That is what lets all three evaluation arms run against identical weights
  # in a single session, which matters for fairness as much as for cost:
  # restarting between arms lets memory fragmentation and kernel autotuning
  # drift line up with arm identity.
  uv run vllm serve "$BASE_MODEL" \
    --port "$VLLM_PORT" \
    --enable-lora \
    --lora-modules "legalmind=$ADAPTER_DIR" \
    --max-lora-rank 32 \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.90 \
    > vllm.log 2>&1 &
  echo $! > .vllm.pid

  printf '  waiting for vLLM'
  for _ in $(seq 120); do
    if curl -fsS "http://localhost:${VLLM_PORT}/health" >/dev/null 2>&1; then break; fi
    printf '.'; sleep 5
  done
  printf '\n'
  curl -fsS "http://localhost:${VLLM_PORT}/health" >/dev/null \
    || fail "vLLM did not come up in 10 minutes — see vllm.log"
  ok "vLLM on :$VLLM_PORT"

  LEGALMIND_UPSTREAM="http://localhost:${VLLM_PORT}/v1" \
  LEGALMIND_MODEL="$BASE_MODEL" \
    uv run uvicorn legalmind.serve.gateway:app \
      --host 0.0.0.0 --port "$GATEWAY_PORT" > gateway.log 2>&1 &
  echo $! > .gateway.pid
  sleep 3
  curl -fsS "http://localhost:${GATEWAY_PORT}/healthz" | tee /dev/null \
    || fail "gateway did not come up — see gateway.log"
  ok "gateway on :$GATEWAY_PORT"
}

evaluate() {
  log "Evaluation: three arms"
  curl -fsS "http://localhost:${VLLM_PORT}/health" >/dev/null \
    || fail "vLLM is not running — run '$0 serve' first"

  # Order matters for cost, not correctness: forgetting refuses to report on a
  # contaminated benchmark, and finding that out first costs six hundred
  # generations rather than twenty-three thousand.
  uv run python -m legalmind.eval.forgetting  --out eval_results/forgetting.json
  uv run python -m legalmind.eval.redteam     --out eval_results/redteam.json
  uv run python -m legalmind.eval.legalbench  --out eval_results/legalbench.json
  uv run python scripts/bench_latency.py --gateway "http://localhost:${GATEWAY_PORT}" \
    --out eval_results/latency.json
  ok "results under eval_results/ — commit them, every README number cites one"
}

stop() {
  for name in gateway vllm; do
    if [ -f ".${name}.pid" ]; then
      kill "$(cat ".${name}.pid")" 2>/dev/null && ok "stopped $name" || true
      rm -f ".${name}.pid"
    fi
  done
}

case "${1:-}" in
  preflight) preflight ;;
  train)     train ;;
  serve)     serve ;;
  evaluate)  evaluate ;;
  stop)      stop ;;
  all)       preflight; train; serve; evaluate ;;
  *)         echo "usage: $0 {preflight|train|serve|evaluate|stop|all}" >&2; exit 2 ;;
esac
