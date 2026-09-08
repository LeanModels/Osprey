#!/bin/bash
# Regenerate a prompt file's assistant turns with the target model, producing
# the on-policy data that Stage 4 distills against.
#
# Launches NUM_SERVERS SGLang servers on this node, streams every prompt in
# INPUT_FILE through them, and writes {"id", "conversations"} records to
# OUTPUT_FILE. Servers are torn down on exit.
#
#   INPUT_FILE=prompts.jsonl OUTPUT_FILE=out.jsonl \
#     bash scripts/osprey/regenerate_data.sh
#
# Environment:
#   INPUT_FILE     (required) prompt-side JSONL: {"id", "conversations": [...]}
#   OUTPUT_FILE    (required) destination; --resume skips ids already present
#   MODEL          target model (default Qwen/Qwen3-8B)
#   TP_SIZE        tensor parallel per server (default 1)
#   NUM_SERVERS    servers to launch (default 8, one per GPU at TP=1)
#   MAX_TOKENS     generation cap (default 4096, the paper's value)
#   TEMPERATURE    sampling temperature (default 1.0)
#   CONCURRENCY    in-flight requests per server (default 32)
#   REASONING_PARSER  SGLang reasoning parser (default qwen3; use
#                     minimax-append-think for MiniMax-M2.5)
#
# The paper's Qwen3-8B splits are five runs of this script, one per domain, at
# MAX_TOKENS=4096. The MiniMax-M2.5 split is one run over the code prompts with
# MODEL/TP_SIZE/REASONING_PARSER pointed at that target.

set -euo pipefail

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
ROOT_DIR=$(dirname "$(dirname "$SCRIPT_DIR")")

MODEL=${MODEL:-Qwen/Qwen3-8B}
INPUT_FILE=${INPUT_FILE:?set INPUT_FILE to the prompt-side JSONL}
OUTPUT_FILE=${OUTPUT_FILE:?set OUTPUT_FILE to the destination JSONL}

TP_SIZE=${TP_SIZE:-1}
NUM_SERVERS=${NUM_SERVERS:-8}
BASE_PORT=${BASE_PORT:-60010}
CONCURRENCY=${CONCURRENCY:-32}
MAX_TOKENS=${MAX_TOKENS:-4096}
TEMPERATURE=${TEMPERATURE:-1.0}
CUDA_GRAPH_MAX_BS=${CUDA_GRAPH_MAX_BS:-64}
CONTEXT_LENGTH=${CONTEXT_LENGTH:-$((MAX_TOKENS * 2 + 2048))}
REASONING_PARSER=${REASONING_PARSER:-qwen3}
HEALTH_TIMEOUT=${HEALTH_TIMEOUT:-900}

ulimit -n 65536 2>/dev/null || true

if [ ! -f "$INPUT_FILE" ]; then
    echo "ERROR: input file not found: $INPUT_FILE" >&2
    exit 1
fi
mkdir -p "$(dirname "$OUTPUT_FILE")"

SERVER_PIDS=()
cleanup() {
    echo ""
    echo ">>> Shutting down SGLang servers ..."
    for pid in "${SERVER_PIDS[@]:-}"; do
        if kill -0 "$pid" 2>/dev/null; then
            kill "$pid" 2>/dev/null || true
            wait "$pid" 2>/dev/null || true
        fi
    done
}
trap cleanup EXIT

wait_for_server() {
    local url=$1 timeout=$2 start=$SECONDS
    echo -n "    Waiting for $url ..."
    while ! curl -sf -o /dev/null "$url/v1/models" 2>/dev/null; do
        if (( SECONDS - start > timeout )); then
            echo " TIMEOUT after ${timeout}s"
            return 1
        fi
        sleep 2
    done
    echo " ready ($(( SECONDS - start ))s)"
}

echo ">>> Launching $NUM_SERVERS SGLang servers for $MODEL (TP=$TP_SIZE each)"
SERVER_ADDRS=()
for (( i = 0; i < NUM_SERVERS; i++ )); do
    port=$(( BASE_PORT + i * 10 ))
    first_gpu=$(( i * TP_SIZE ))
    gpus=$(seq -s, "$first_gpu" $(( first_gpu + TP_SIZE - 1 )))
    echo "    server $i: GPU(s) $gpus -> port $port"
    CUDA_VISIBLE_DEVICES="$gpus" python3 -m sglang.launch_server \
        --model "$MODEL" \
        --mem-fraction-static 0.9 \
        --tp "$TP_SIZE" \
        --trust-remote-code \
        --cuda-graph-max-bs "$CUDA_GRAPH_MAX_BS" \
        --context-length "$CONTEXT_LENGTH" \
        --reasoning-parser "$REASONING_PARSER" \
        --dtype bfloat16 \
        --host 0.0.0.0 \
        --port "$port" &
    SERVER_PIDS+=($!)
    SERVER_ADDRS+=("localhost:$port")
done

echo ""
for addr in "${SERVER_ADDRS[@]}"; do
    wait_for_server "http://$addr" "$HEALTH_TIMEOUT"
done

echo ""
echo ">>> Regenerating with $MODEL"
echo "    in:  $INPUT_FILE"
echo "    out: $OUTPUT_FILE"
echo "    max_tokens=$MAX_TOKENS temperature=$TEMPERATURE"
echo "    in-flight requests: $(( CONCURRENCY * NUM_SERVERS ))"
echo ""

python "$ROOT_DIR/scripts/regenerate_train_data.py" \
    --model "$MODEL" \
    --concurrency "$CONCURRENCY" \
    --max-tokens "$MAX_TOKENS" \
    --temperature "$TEMPERATURE" \
    --server-address "${SERVER_ADDRS[@]}" \
    --input-file-path "$INPUT_FILE" \
    --output-file-path "$OUTPUT_FILE" \
    --is-reasoning-model \
    --resume

echo ">>> Done: $OUTPUT_FILE ($(wc -l < "$OUTPUT_FILE") records)"
