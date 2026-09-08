#!/bin/bash
# Stage 4 on Qwen/Qwen3-8B: adapt one drafter arm on each of the five
# single-domain splits. Running all five fills the training rows of the
# cross-domain matrix (paper Figure 2 for DRAFT=osprey, Table 6 for the
# eagle3-* arms).
#
#   bash examples/run_osprey_qwen3_8b_adaptation.sh [NUM_GPUS]
#
# Required environment:
#   OSPREY_DATA          directory holding {domain}_train_65k_qwen3_8B_4096.jsonl
#   OSPREY_OUTPUT        output root; each domain writes to
#                        $OSPREY_OUTPUT/qwen3-8b/$DRAFT/{domain}
#   OSPREY_DRAFT_INIT    converted drafter directory from
#                        scripts/osprey/convert_checkpoint.py.
#                        Required for DRAFT=osprey only.
#
# Optional:
#   DRAFT            osprey (default) | eagle3-1layer | eagle3-2layer
#                    osprey        - warm-started body, paper's main result
#                    eagle3-1layer - from-scratch 1-layer EAGLE-3 baseline
#                    eagle3-2layer - parameter-matched 2-layer baseline (Table 6)
#   LEARNING_RATE    1e-4 (default, paper's main setting) or 1e-5 (Table 3/8)
#   OSPREY_DOMAINS   space-separated subset of the five domains
#   NUM_GPUS         default 2, which gives the paper's 97,500 steps
#                    (65k samples x 3 epochs / 2). Changing it changes the
#                    effective batch and the step count.
#   EVAL_ALL_DOMAINS set to 1 to evaluate held-out loss on all five domains
#                    during training (wandb only; the reported accepted
#                    lengths come from the separate eval script)
#   OSPREY_CACHE_DIR tokenizer cache. Give concurrent launches separate dirs.
#   SAVE_INTERVAL    steps between checkpoints (default 5000). Each one is
#                    ~6 GB, so 97,500 steps at the default is ~120 GB per
#                    domain; raise this if disk is tight. The final step is
#                    always saved regardless.
#   DRY_RUN          set to 1 to print each torchrun command without running
#
# This script runs its domains sequentially. To train several domains at once,
# launch one invocation per domain with its own CUDA_VISIBLE_DEVICES,
# OSPREY_DOMAINS and OSPREY_CACHE_DIR; `torchrun --standalone` picks a free
# rendezvous port per job, so concurrent launches do not collide.
#
# Everything else (3 epochs, TTT length 5, sequence length 4096, warmup 0.04,
# qwen3-thinking template, flex attention) lives in
# scripts/osprey/run_adaptation.py.

set -euo pipefail

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
ROOT_DIR=$(dirname "$SCRIPT_DIR")

export TORCHINDUCTOR_CACHE_DIR=${TORCHINDUCTOR_CACHE_DIR:-$ROOT_DIR/cache/compiled_kernels}
export SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1
export PYTORCH_ALLOC_CONF=${PYTORCH_ALLOC_CONF:-expandable_segments:True}
mkdir -p "$TORCHINDUCTOR_CACHE_DIR"

NUM_GPUS=${1:-2}
DRAFT=${DRAFT:-osprey}
LEARNING_RATE=${LEARNING_RATE:-1e-4}
SAVE_INTERVAL=${SAVE_INTERVAL:-5000}
DOMAINS=${OSPREY_DOMAINS:-"chat code commonsense finance math"}

for var in OSPREY_DATA OSPREY_OUTPUT; do
    if [ -z "${!var:-}" ]; then
        echo "ERROR: $var is not set. See the header of this script." >&2
        exit 2
    fi
done

case "$DRAFT" in
    osprey)
        if [ -z "${OSPREY_DRAFT_INIT:-}" ] || [ ! -d "${OSPREY_DRAFT_INIT}" ]; then
            echo "ERROR: DRAFT=osprey needs OSPREY_DRAFT_INIT pointing at a" >&2
            echo "       converted drafter directory. Run" >&2
            echo "       scripts/osprey/convert_checkpoint.py first." >&2
            exit 2
        fi
        DRAFT_ARGS=(--draft-checkpoint "$OSPREY_DRAFT_INIT")
        ;;
    eagle3-1layer)
        DRAFT_ARGS=(--from-scratch --draft-checkpoint "$ROOT_DIR/configs/qwen3-8b-eagle3.json")
        ;;
    eagle3-2layer)
        DRAFT_ARGS=(--from-scratch --draft-checkpoint "$ROOT_DIR/configs/qwen3-8b-eagle3-2layer.json")
        ;;
    *)
        echo "ERROR: DRAFT must be osprey, eagle3-1layer or eagle3-2layer" >&2
        exit 2
        ;;
esac

EVAL_ARGS=()
if [ "${EVAL_ALL_DOMAINS:-0}" = "1" ]; then
    for domain in chat code commonsense finance math; do
        EVAL_ARGS+=(--eval-data "$OSPREY_DATA/${domain}_eval_512_qwen3_8B_4096.jsonl")
    done
fi

for domain in $DOMAINS; do
    train_data="$OSPREY_DATA/${domain}_train_65k_qwen3_8B_4096.jsonl"
    if [ ! -f "$train_data" ]; then
        echo "ERROR: missing training split: $train_data" >&2
        exit 2
    fi

    echo "=== Stage 4 [$DRAFT] lr=$LEARNING_RATE domain=$domain ==="
    python "$ROOT_DIR/scripts/osprey/run_adaptation.py" \
        --target qwen3-8b \
        "${DRAFT_ARGS[@]}" \
        --train-data "$train_data" \
        --output-dir "$OSPREY_OUTPUT/qwen3-8b/$DRAFT/$domain" \
        --cache-dir "${OSPREY_CACHE_DIR:-$ROOT_DIR/cache/osprey}" \
        --num-gpus "$NUM_GPUS" \
        --learning-rate "$LEARNING_RATE" \
        --save-interval "$SAVE_INTERVAL" \
        "${EVAL_ARGS[@]}" \
        ${DRY_RUN:+--dry-run}
done
