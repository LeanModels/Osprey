#!/bin/bash
# Apply (or check, or revert) the Osprey SGLang patch on the sglang installed in
# the active Python environment.
#
#   bash sglang_patches/apply_patch.sh              # apply
#   bash sglang_patches/apply_patch.sh --check      # dry-run only, no writes
#   bash sglang_patches/apply_patch.sh --revert     # restore the backups
#
# The patch is authored against sglang 0.5.9, the version pinned in
# pyproject.toml. It is required for *serving* an Osprey drafter (evaluation);
# training does not use it.

set -euo pipefail

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
PATCH_FILE="$SCRIPT_DIR/sglang.patch"
MODE=${1:-apply}

EXPECTED_VERSION=0.5.9
TOUCHED_FILES=(
    sglang/srt/models/llama.py
    sglang/srt/models/llama_eagle3.py
    sglang/srt/models/qwen3.py
)
NEW_FILE=sglang/srt/models/qwen3_eagle3.py

SITE_PACKAGES=$(python -c 'import os, sglang; print(os.path.dirname(os.path.dirname(sglang.__file__)))')
VERSION=$(python -c 'import sglang; print(sglang.__version__)')

echo "sglang $VERSION at $SITE_PACKAGES"
if [ "$VERSION" != "$EXPECTED_VERSION" ]; then
    echo "WARNING: patch was authored against sglang $EXPECTED_VERSION." >&2
    echo "         It may not apply to $VERSION." >&2
fi

cd "$SITE_PACKAGES"

if [ "$MODE" = "--revert" ]; then
    for f in "${TOUCHED_FILES[@]}"; do
        if [ -f "$f.osprey_orig" ]; then
            cp "$f.osprey_orig" "$f"
            echo "reverted $f"
        else
            echo "no backup for $f, skipping" >&2
        fi
    done
    rm -f "$NEW_FILE"
    find sglang/srt/models/__pycache__ -name 'llama*' -o -name 'qwen3*' 2>/dev/null | xargs -r rm -f
    echo "Revert complete."
    exit 0
fi

# The patch paths are python/sglang/srt/...; site-packages is rooted at
# sglang/srt/..., so strip two leading components.
if [ -f "$NEW_FILE" ] && grep -q "Qwen3ForCausalLMEagle3" "$NEW_FILE" 2>/dev/null; then
    echo "Already patched ($NEW_FILE exists). Nothing to do."
    echo "Use --revert first if you want to re-apply."
    exit 0
fi

echo "--- dry run ---"
patch -p2 --dry-run < "$PATCH_FILE"

if [ "$MODE" = "--check" ]; then
    echo "Dry run clean. Re-run without --check to apply."
    exit 0
fi

for f in "${TOUCHED_FILES[@]}"; do
    cp -n "$f" "$f.osprey_orig"
done

echo "--- applying ---"
patch -p2 < "$PATCH_FILE"

# Drop stale bytecode for the modules we just rewrote.
find sglang/srt/models/__pycache__ \
    \( -name 'llama*.cpython*' -o -name 'qwen3*.cpython*' \) -delete 2>/dev/null || true

python - <<'EOF'
from sglang.srt.models.qwen3_eagle3 import Qwen3ForCausalLMEagle3  # noqa: F401
from sglang.srt.models.llama_eagle3 import LlamaForCausalLMEagle3  # noqa: F401

print("import check: Qwen3ForCausalLMEagle3 and LlamaForCausalLMEagle3 both load")
EOF

echo "Patch applied. Backups saved as *.osprey_orig next to each modified file."
