#!/bin/bash
# Five-domain evaluation of an Osprey drafter against meta-llama/Llama-3.3-70B-Instruct
# (paper Table 1). The benchmarks read only the prompts, so this reuses the
# Qwen3-8B evaluation script with the target swapped and tp raised to 4.
#
#   bash examples/llama33-70b/eval.sh <CHECKPOINT_DIR>
#
# Required environment: OSPREY_DATA, OSPREY_OUTPUT (see examples/qwen3-8b/eval.sh).
# Needs 4 GPUs; the target is served in bf16 at tp=4.

set -euo pipefail
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
OSPREY_TARGET_MODEL=${OSPREY_TARGET_MODEL:-meta-llama/Llama-3.3-70B-Instruct} \
TP_SIZE=${TP_SIZE:-4} \
MEM_FRACTION_STATIC=${MEM_FRACTION_STATIC:-0.75} \
  exec bash "$SCRIPT_DIR/../qwen3-8b/eval.sh" "$@"
