#!/bin/bash
# Five-domain evaluation of one adapted checkpoint against Qwen/Qwen3-8B.
# Each run produces one row of the cross-domain matrix (paper Figure 2); run it
# for all five checkpoints of an arm to fill the 5x5 matrix.
#
#   bash examples/run_osprey_qwen3_8b_eval.sh <CHECKPOINT_DIR>
#
# CHECKPOINT_DIR is a single epoch/step directory written by Stage 4, e.g.
#   $OSPREY_OUTPUT/qwen3-8b/osprey/math/epoch_2_step_97500
#
# Required environment:
#   OSPREY_DATA     directory holding {domain}_eval_512_qwen3_8B_4096.jsonl
#   OSPREY_OUTPUT   output root; results go to $OSPREY_OUTPUT/results/<name>
#
# Optional:
#   OSPREY_DOMAINS  space-separated subset of the five domains
#   NUM_PROMPTS     prompts per domain (default 512, the whole eval split).
#                   This is what the paper reports. Accepted length falls as
#                   the prompt count rises on every domain except code, so a
#                   subsample reads high: on commonsense a 32-prompt run gives
#                   4.70 where the full split gives 4.34. Lower it only for a
#                   quick smoke test, and never compare a subsampled number
#                   against the paper's tables.
#   RESULT_NAME     subdirectory under results/ (default: derived from the path)
#   TP_SIZE         tensor parallelism for the target (default 1; 4 for Llama-3.3-70B)
#   PORT            SGLang port (default 30000). Give each concurrent run its
#                   own port and its own CUDA_VISIBLE_DEVICES to evaluate
#                   several checkpoints at once.
#
# Serving configuration is the paper's: (batch, steps, topk, draft_tokens) =
# (1, 5, 1, 6) at max_tokens 4096, greedy, tp=1 on a single GPU.
#
# Requires the SGLang patch (bash sglang_patches/apply_patch.sh). Without it a
# multi-layer Osprey draft cannot be served and acceptance collapses to ~1.0.

set -euo pipefail

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
ROOT_DIR=$(dirname "$SCRIPT_DIR")

if [ $# -lt 1 ]; then
    echo "usage: bash examples/run_osprey_qwen3_8b_eval.sh <CHECKPOINT_DIR>" >&2
    exit 2
fi

DRAFT_CKPT=$1
NUM_PROMPTS=${NUM_PROMPTS:-512}
PORT=${PORT:-30000}
DOMAINS=${OSPREY_DOMAINS:-"chat code commonsense finance math"}
TARGET_MODEL=${OSPREY_TARGET_MODEL:-Qwen/Qwen3-8B}

for var in OSPREY_DATA OSPREY_OUTPUT; do
    if [ -z "${!var:-}" ]; then
        echo "ERROR: $var is not set. See the header of this script." >&2
        exit 2
    fi
done

if [ ! -d "$DRAFT_CKPT" ]; then
    echo "ERROR: not a checkpoint directory: $DRAFT_CKPT" >&2
    exit 2
fi

BENCHMARKS=()
for domain in $DOMAINS; do
    eval_data="$OSPREY_DATA/${domain}_eval_512_qwen3_8B_4096.jsonl"
    if [ ! -f "$eval_data" ]; then
        echo "ERROR: missing evaluation split: $eval_data" >&2
        exit 2
    fi
    BENCHMARKS+=("${domain}_eval:${NUM_PROMPTS}")
done

# Default result name: <arm>-<train-domain>, read off the checkpoint path.
if [ -z "${RESULT_NAME:-}" ]; then
    train_domain=$(basename "$(dirname "$DRAFT_CKPT")")
    arm=$(basename "$(dirname "$(dirname "$DRAFT_CKPT")")")
    RESULT_NAME="${arm}-${train_domain}"
fi

echo "=== Evaluating $RESULT_NAME ==="
echo "    draft:  $DRAFT_CKPT"
echo "    target: $TARGET_MODEL"

python "$ROOT_DIR/benchmarks/bench_eagle3.py" \
    --model-path "$TARGET_MODEL" \
    --speculative-draft-model-path "$DRAFT_CKPT" \
    --config-list 1,5,1,6 \
    --port "$PORT" \
    --benchmark-list "${BENCHMARKS[@]}" \
    --domain-eval-data-dir "$OSPREY_DATA" \
    --domain-eval-suffix _4096 \
    --max-tokens 4096 \
    --tp-size "${TP_SIZE:-1}" \
    --mem-fraction-static "${MEM_FRACTION_STATIC:-0.8}" \
    --dtype bfloat16 \
    --name "$RESULT_NAME" \
    --output-dir "$OSPREY_OUTPUT/results/$RESULT_NAME"
