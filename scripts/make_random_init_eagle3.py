#!/usr/bin/env python3
"""Build the body-random control for the "did pretraining help?" ablation.

Produces a drafter identical to a converted Osprey checkpoint in every respect
except the one under test: the config (`swap_h_and_emb`, layer count, hidden
sizes, vocabulary), the tokenizer-aligned `embed_tokens`, the zero `fc`, and
`t2d`/`d2t` are all copied from the base checkpoint, while the transformer body
and `lm_head` are randomly initialized. Both of those are products of Stage 2
pretraining, so the only variable against the warm-started run is the language
prior itself (paper Table 4).

The random init goes through `AutoEagle3DraftModel.from_config(config)`, i.e.
exactly what the trainer builds when given no checkpoint, so the architecture
matches its pretrained sibling exactly.

For the *parameter-matched* from-scratch EAGLE-3 baselines (paper Table 6), do
not use this script: train directly from `configs/qwen3-8b-eagle3.json` or
`configs/qwen3-8b-eagle3-2layer.json`, which is what
`examples/run_osprey_qwen3_8b_adaptation.sh` does with `DRAFT=eagle3-1layer` or
`DRAFT=eagle3-2layer`.
"""

import argparse
import json
import shutil
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from specforge.modeling.auto import AutoDraftModelConfig, AutoEagle3DraftModel

# lm_head is intentionally NOT preserved — it is a product of FineWeb-1T
# pretraining, so the from-scratch control must randomize it too. Only the
# embedding table, fc warm-start, and vocab maps are kept constant.
PRESERVE_ALWAYS = ("embed_tokens.weight", "t2d", "d2t")
TOKENIZER_FILES = (
    "tokenizer.json",
    "tokenizer.model",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "generation_config.json",
    "added_tokens.json",
    "merges.txt",
    "vocab.json",
    "chat_template.jinja",
)


def load_base_state(base_dir: Path) -> dict:
    out = {}
    with safe_open(
        str(base_dir / "model.safetensors"), framework="pt", device="cpu"
    ) as f:
        for k in f.keys():
            out[k] = f.get_tensor(k)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--base",
        required=True,
        help="Pretrained converted Eagle-3 ckpt dir (architecture source)",
    )
    ap.add_argument("--output", required=True)
    ap.add_argument(
        "--mode",
        default="body-random",
        choices=["body-random"],
        help="Only body-random is supported; the vanilla baseline is built "
        "checkpoint-free via the trainer's --num-draft-layers flag.",
    )
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    base = Path(args.base)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)

    base_cfg = json.load(open(base / "config.json"))
    cfg_dict = dict(base_cfg)

    cfg_path = out / "config.json"
    with open(cfg_path, "w") as f:
        json.dump(cfg_dict, f, indent=2)
        f.write("\n")

    # Build the random-init model via the trainer's own from_config path so
    # the architecture is guaranteed identical to what training would build.
    config = AutoDraftModelConfig.from_file(str(cfg_path))
    model = AutoEagle3DraftModel.from_config(config, torch_dtype=torch.bfloat16)
    sd = model.state_dict()

    base_state = load_base_state(base)
    hidden = int(base_cfg["hidden_size"])

    # Preserve only the embedding table, fc warm-start, and vocab maps.
    # lm_head + the transformer body are left random (they are products of
    # FineWeb-1T pretraining and are exactly what this control ablates).
    preserve = list(PRESERVE_ALWAYS) + ["fc.weight"]

    for k in preserve:
        if k in base_state and k in sd:
            sd[k] = base_state[k].to(sd[k].dtype).clone()

    # Zero the conditioning half of q/k/v to match the converted checkpoint's
    # fixed [0, W_pretrained] layout. This
    # keeps the swap architecture identical; only the h-half body weights
    # differ (random here vs pretrained in the base ckpt).
    zero_cols = slice(0, hidden)
    n_zeroed = 0
    for k in list(sd.keys()):
        if k.endswith(
            (
                "self_attn.q_proj.weight",
                "self_attn.k_proj.weight",
                "self_attn.v_proj.weight",
            )
        ):
            if sd[k].shape[1] == 2 * hidden:
                sd[k][:, zero_cols] = 0
                n_zeroed += 1
    print(f"[body-random] zeroed e-half of {n_zeroed} q/k/v projections")

    save_file(sd, str(out / "model.safetensors"), metadata={"format": "pt"})

    for name in TOKENIZER_FILES:
        src = base / name
        if src.exists():
            shutil.copy2(src, out / name)

    # Report
    print(f"Saved {args.mode} random-init Eagle-3 to {out}")
    print(
        f"  num_hidden_layers={cfg_dict['num_hidden_layers']}, hidden={hidden}, "
        f"target_hidden={cfg_dict.get('target_hidden_size')}, vocab={cfg_dict['vocab_size']}"
    )
    print(f"  swap_h_and_emb={cfg_dict.get('swap_h_and_emb', '(unset)')}")
    print(f"  preserved from base: {preserve}")
    with torch.no_grad():
        q = sd["layers.0.self_attn.q_proj.weight"].float()
        fc = sd["fc.weight"].float()
        print(
            f"  q_proj[0] std={q.std():.4e} (e-half zeros={int((q[:, :hidden]==0).all())})"
        )
        print(
            f"  fc std={fc.std():.4e}, embed preserved={'embed_tokens.weight' in preserve}"
        )


if __name__ == "__main__":
    main()
