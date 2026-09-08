#!/bin/bash
# Stage 4 on meta-llama/Llama-3.3-70B-Instruct: adapt the converted drafter on
# the target-regenerated Open-PerfectBlend split (paper Table 1).
#
#   bash examples/llama33-70b/train.sh
#
# Required environment:
#   OSPREY_DRAFT_INIT   converted drafter directory, from
#                       scripts/osprey/convert_checkpoint.py --target llama33-70b
#   OSPREY_LLAMA_DATA   perfectblend_train_100k_llama33_70b.jsonl
#   OSPREY_OUTPUT       output root
#
# Optional:
#   NUM_GPUS            default 8. Each GPU is one draft data-parallel rank, so
#                       an epoch over the 100,000-sample split is 12,500 steps
#                       and the paper's 3 epochs are 37,500.
#   TP_SIZE             target tensor parallel, default 4 (bf16 70B fits in
#                       4x80 GB with room for the draft).
#   LEARNING_RATE       default 1e-4
#   SAVE_INTERVAL       default 5000
#   SGLANG_ATTN_BACKEND default fa3 (Hopper); flashinfer on Blackwell
#   DRY_RUN             set to 1 to print the torchrun command only
#
# The target is served in-process by SGLang at tp=4 to produce hidden states
# and logits on the fly; nothing is precomputed. The model is gated on the Hub:
# accept the license and log in with `hf auth login` first.

set -euo pipefail

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
ROOT_DIR=$(dirname "$(dirname "$SCRIPT_DIR")")

export TORCHINDUCTOR_CACHE_DIR=${TORCHINDUCTOR_CACHE_DIR:-$ROOT_DIR/cache/compiled_kernels}
export SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
mkdir -p "$TORCHINDUCTOR_CACHE_DIR"

NUM_GPUS=${NUM_GPUS:-8}
TP_SIZE=${TP_SIZE:-4}
LEARNING_RATE=${LEARNING_RATE:-1e-4}
SAVE_INTERVAL=${SAVE_INTERVAL:-5000}
SGLANG_ATTN_BACKEND=${SGLANG_ATTN_BACKEND:-fa3}

for var in OSPREY_DRAFT_INIT OSPREY_LLAMA_DATA OSPREY_OUTPUT; do
    if [ -z "${!var:-}" ]; then
        echo "ERROR: $var is not set. See the header of this script." >&2
        exit 2
    fi
done
if [ ! -d "$OSPREY_DRAFT_INIT" ]; then
    echo "ERROR: OSPREY_DRAFT_INIT is not a directory: $OSPREY_DRAFT_INIT" >&2
    echo "       Run scripts/osprey/convert_checkpoint.py --target llama33-70b first." >&2
    exit 2
fi
if [ ! -f "$OSPREY_LLAMA_DATA" ]; then
    echo "ERROR: missing training split: $OSPREY_LLAMA_DATA" >&2
    exit 2
fi

echo "=== Stage 4: Llama-3.3-70B-Instruct, tp=$TP_SIZE, $NUM_GPUS GPUs ==="
python "$ROOT_DIR/scripts/osprey/run_adaptation.py" \
    --target llama33-70b \
    --draft-checkpoint "$OSPREY_DRAFT_INIT" \
    --train-data "$OSPREY_LLAMA_DATA" \
    --output-dir "$OSPREY_OUTPUT/llama33-70b/osprey/perfectblend" \
    --cache-dir "${OSPREY_CACHE_DIR:-$ROOT_DIR/cache/osprey-llama33-70b}" \
    --num-gpus "$NUM_GPUS" \
    --tp-size "$TP_SIZE" \
    --learning-rate "$LEARNING_RATE" \
    --save-interval "$SAVE_INTERVAL" \
    --sglang-attention-backend "$SGLANG_ATTN_BACKEND" \
    ${DRY_RUN:+--dry-run}
