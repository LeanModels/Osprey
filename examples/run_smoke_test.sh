#!/bin/bash
# Two-minute smoke test: train a from-scratch EAGLE-3 draft against Qwen3-8B on
# the ~1k-sample dataset committed at data/train.small.jsonl. Proves the
# environment, the SGLang target backend, and the training loop all work before
# you commit a GPU-day to a real run. It does not produce a usable drafter.
#
#   bash examples/run_smoke_test.sh [NUM_GPUS]

set -euo pipefail

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
ROOT_DIR=$(dirname "$SCRIPT_DIR")

export TORCHINDUCTOR_CACHE_DIR=${TORCHINDUCTOR_CACHE_DIR:-$ROOT_DIR/cache/compiled_kernels}
export SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1
mkdir -p "$TORCHINDUCTOR_CACHE_DIR"

NUM_GPUS=${1:-2}
OUTPUT_DIR=${OUTPUT_DIR:-$ROOT_DIR/outputs/smoke-test}

torchrun \
    --standalone \
    --nproc_per_node "$NUM_GPUS" \
    "$ROOT_DIR/scripts/train_eagle3.py" \
    --target-model-path Qwen/Qwen3-8B \
    --target-model-backend sglang \
    --draft-model-config "$ROOT_DIR/configs/qwen3-8b-eagle3.json" \
    --train-data-path "$ROOT_DIR/data/train.small.jsonl" \
    --output-dir "$OUTPUT_DIR" \
    --cache-dir "$ROOT_DIR/cache/smoke-test" \
    --num-epochs 1 \
    --batch-size 1 \
    --learning-rate 1e-4 \
    --max-length 4096 \
    --ttt-length 5 \
    --chat-template qwen3-thinking \
    --attention-backend flex_attention \
    --tp-size 1 \
    --log-interval 1 \
    --embedding-key model.embed_tokens.weight \
    --report-to none
