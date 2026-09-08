#!/bin/bash
# Evaluate a drafter against MiniMaxAI/MiniMax-M2.5 (229B FP8 MoE).
#
#   bash examples/run_osprey_minimax_m25_eval.sh <DRAFT_CHECKPOINT> [SUITE]
#
# DRAFT_CHECKPOINT is either a Stage-4 epoch/step directory, or the public
# baseline `thoughtworks/MiniMax-M2.5-Eagle3` for the comparison arm.
#
# SUITE selects which table to produce:
#   public       paper Table 1 - humaneval, math500, livecodebench, mtbench,
#                commonsense_eval at 64 prompts each
#   multilingual paper Table 2 - mgsm (64), global_mmlu (84),
#                global_mmlu_native (84)
#   all          both (default)
#
# Required environment:
#   OSPREY_OUTPUT   output root; results go to $OSPREY_OUTPUT/results/<name>
#   OSPREY_DATA     needed by the commonsense_eval row of the public suite
#
# Optional:
#   TP_SIZE           default 4, the paper's setting
#   EP_SIZE           default 4
#   SGLANG_ATTN_BACKEND  default flashinfer (Blackwell); fa3 on Hopper
#   RESULT_NAME       subdirectory under results/
#
# This script launches SGLang itself, because an FP8 MoE target needs server
# flags (ep-size, moe backend, custom all-reduce) that the benchmark runner
# does not forward, and then drives it with --skip-launch-server. Serving
# configuration is the paper's: (batch, steps, topk, draft_tokens) =
# (1, 5, 1, 6) at max_tokens 4096, greedy.
#
# Requires the SGLang patch (bash sglang_patches/apply_patch.sh).

set -euo pipefail

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
ROOT_DIR=$(dirname "$SCRIPT_DIR")

if [ $# -lt 1 ]; then
    echo "usage: bash examples/run_osprey_minimax_m25_eval.sh <DRAFT_CHECKPOINT> [public|multilingual|all]" >&2
    exit 2
fi

DRAFT_CKPT=$1
SUITE=${2:-all}
TARGET_MODEL=${OSPREY_TARGET_MODEL:-MiniMaxAI/MiniMax-M2.5}
TP_SIZE=${TP_SIZE:-4}
EP_SIZE=${EP_SIZE:-4}
PORT=${PORT:-30000}
SGLANG_ATTN_BACKEND=${SGLANG_ATTN_BACKEND:-flashinfer}
MEM_FRACTION_STATIC=${MEM_FRACTION_STATIC:-0.85}

if [ -z "${OSPREY_OUTPUT:-}" ]; then
    echo "ERROR: OSPREY_OUTPUT is not set." >&2
    exit 2
fi

BENCHMARKS=()
case "$SUITE" in
    public|all)
        if [ -z "${OSPREY_DATA:-}" ]; then
            echo "ERROR: the public suite needs OSPREY_DATA for commonsense_eval." >&2
            exit 2
        fi
        BENCHMARKS+=(humaneval:64 math500:64 livecodebench:64 mtbench:64 commonsense_eval:64)
        ;;
esac
case "$SUITE" in
    multilingual|all)
        BENCHMARKS+=(mgsm:64 global_mmlu:84 global_mmlu_native:84)
        ;;
esac
if [ ${#BENCHMARKS[@]} -eq 0 ]; then
    echo "ERROR: SUITE must be public, multilingual or all" >&2
    exit 2
fi

RESULT_NAME=${RESULT_NAME:-$(basename "$DRAFT_CKPT")-$SUITE}
export SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1

SERVER_PID=""
cleanup() {
    if [ -n "$SERVER_PID" ] && kill -0 "$SERVER_PID" 2>/dev/null; then
        echo ">>> Stopping SGLang server (pid $SERVER_PID)"
        kill "$SERVER_PID" 2>/dev/null || true
        wait "$SERVER_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT

echo "=== Launching MiniMax-M2.5 with EAGLE3 speculative decoding ==="
echo "    target: $TARGET_MODEL  (tp=$TP_SIZE ep=$EP_SIZE)"
echo "    draft:  $DRAFT_CKPT"
python -m sglang.launch_server \
    --model-path "$TARGET_MODEL" \
    --speculative-algorithm EAGLE3 \
    --speculative-draft-model-path "$DRAFT_CKPT" \
    --speculative-num-steps 5 \
    --speculative-eagle-topk 1 \
    --speculative-num-draft-tokens 6 \
    --tp-size "$TP_SIZE" \
    --ep-size "$EP_SIZE" \
    --attention-backend "$SGLANG_ATTN_BACKEND" \
    --mem-fraction-static "$MEM_FRACTION_STATIC" \
    --cuda-graph-max-bs 1 \
    --max-running-requests 1 \
    --disable-custom-all-reduce \
    --trust-remote-code \
    --host 127.0.0.1 \
    --port "$PORT" &
SERVER_PID=$!

echo -n ">>> Waiting for the server "
START=$SECONDS
until curl -sf -o /dev/null "http://127.0.0.1:$PORT/v1/models"; do
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
        echo " server exited during startup" >&2
        exit 1
    fi
    if (( SECONDS - START > ${LAUNCH_TIMEOUT:-1800} )); then
        echo " timeout" >&2
        exit 1
    fi
    sleep 5
done
echo "ready ($(( SECONDS - START ))s)"

DOMAIN_ARGS=()
if [ -n "${OSPREY_DATA:-}" ]; then
    DOMAIN_ARGS=(--domain-eval-data-dir "$OSPREY_DATA" --domain-eval-suffix _4096)
fi

echo "=== Benchmarking: ${BENCHMARKS[*]} ==="
python "$ROOT_DIR/benchmarks/bench_eagle3.py" \
    --model-path "$TARGET_MODEL" \
    --port "$PORT" \
    --config-list 1,5,1,6 \
    --benchmark-list "${BENCHMARKS[@]}" \
    "${DOMAIN_ARGS[@]}" \
    --max-tokens 4096 \
    --trust-remote-code \
    --skip-launch-server \
    --name "$RESULT_NAME" \
    --output-dir "$OSPREY_OUTPUT/results/$RESULT_NAME"
