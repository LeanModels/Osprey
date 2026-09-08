# Osprey SGLang patch

Stock SGLang can only serve a **single-layer** EAGLE3 draft whose architecture is
`LlamaForCausalLMEagle3`, and it unconditionally replaces the draft's embedding
with the target's at load time. An Osprey drafter breaks all three assumptions:
it has 2 (Qwen3-4B) or 4 (LayerSkip-Llama) layers, it is registered as
`Qwen3ForCausalLMEagle3` for the Qwen3-4B backbone, and its `embed_tokens` is a
tokenizer-aligned table that must not be overwritten.

This patch adds exactly that support. **It is required to evaluate an Osprey
checkpoint.** Training does not use SGLang's draft path and works unpatched.

## Contents

| File | Change |
| --- | --- |
| `sglang/srt/models/qwen3_eagle3.py` | **New.** Qwen3 EAGLE3 draft head. Subclasses `Qwen3DecoderLayer` so the per-head Q/K RMSNorm is preserved, widens `qkv_proj` to the `2 * hidden_size` EAGLE3 concat, and supports a multi-layer body, `swap_h_and_emb`, and `target_hidden_size != hidden_size`. |
| `sglang/srt/models/llama_eagle3.py` | Multi-layer body via a `self.layers` ModuleList, `swap_h_and_emb` argument routing, and a load-time guard that fails loudly if a warm-started draft's checkpoint has no `embed_tokens.weight`. |
| `sglang/srt/models/llama.py` | `set_embed` skips the target-embedding overwrite when `swap_h_and_emb` is set. |
| `sglang/srt/models/qwen3.py` | Adds `get_embed` / `set_embed` with the same guard, since `Qwen3ForCausalLMEagle3` inherits them from `Qwen3ForCausalLM`. |

Every change is gated on a config flag that only Osprey checkpoints set
(`num_hidden_layers > 1`, `swap_h_and_emb`, `target_hidden_size`). A vanilla
1-layer EAGLE3 draft takes the same code path as unpatched SGLang, so the
from-scratch baselines in this repo are unaffected by whether the patch is
applied.

The MiniMax-M2.5 target needs no patch — SGLang 0.5.9 already implements
`set_eagle3_layers_to_capture` for `minimax_m2`.

## Applying

Authored against **sglang 0.5.9**, the version pinned in `pyproject.toml`.

```bash
source .venv/bin/activate
bash sglang_patches/apply_patch.sh --check    # dry run, writes nothing
bash sglang_patches/apply_patch.sh            # apply in place
```

The script backs up each modified file as `<name>.osprey_orig`, clears the stale
bytecode, and import-checks both draft heads. To undo:

```bash
bash sglang_patches/apply_patch.sh --revert
```

If you build SGLang from source instead, apply it at the repo root, where the
patch paths (`python/sglang/srt/...`) line up directly:

```bash
cd <sglang-repo> && git checkout v0.5.9
git apply /path/to/sglang_patches/sglang.patch
pip install -e python/
```

## Verifying it took effect

```bash
python -c "from sglang.srt.models.qwen3_eagle3 import Qwen3ForCausalLMEagle3; print('ok')"
```

If evaluation reports an accepted length near 1.0 for a checkpoint that trained
to a much higher one, the usual cause is an unpatched SGLang: the draft loaded
but silently ran with the target's embedding and only its first layer.
