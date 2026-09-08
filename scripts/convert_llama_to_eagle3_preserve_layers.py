#!/usr/bin/env python3
"""Convert a Llama causal LM checkpoint into a layer-preserving Eagle-3 draft.

The converted model keeps the first requested source decoder layers. For repo
Eagle-3's ``cat(input_emb, hidden_states)``
attention input and h-only residual path, Q/K/V are expanded with old weights on
the hidden-state half:

    [0, W_old]

This makes the initial attention residual follow ``h + attention_old(h)``.
"""

import argparse
import json
import shutil
from pathlib import Path
from typing import Dict

import torch
from safetensors import safe_open
from safetensors.torch import save_file


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, help="Source HF checkpoint dir")
    parser.add_argument("--output", required=True, help="Output checkpoint dir")
    parser.add_argument(
        "--draft-vocab-size",
        type=int,
        default=None,
        help="Draft vocab size. Defaults to full source vocab for best init parity.",
    )
    parser.add_argument(
        "--num-draft-layers",
        type=int,
        default=None,
        help="Number of source decoder layers to keep. Defaults to all source layers.",
    )
    parser.add_argument(
        "--aux-layer-ids",
        type=str,
        default=None,
        help=(
            "Comma-separated Eagle aux layer ids to pin in the draft config. "
            "If omitted, training uses the normal target-model default."
        ),
    )
    parser.add_argument(
        "--target-hidden-size",
        type=int,
        default=None,
        help=(
            "Hidden size of the target model that supplies Eagle-3 aux states. "
            "Defaults to the draft/source hidden size."
        ),
    )
    return parser.parse_args()


def load_config(source: Path) -> Dict:
    with open(source / "config.json", "r", encoding="utf-8") as f:
        return json.load(f)


def load_state_dict(source: Path) -> Dict[str, torch.Tensor]:
    safetensors_path = source / "model.safetensors"
    if safetensors_path.exists():
        out = {}
        with safe_open(str(safetensors_path), framework="pt", device="cpu") as f:
            for key in f.keys():
                out[key] = f.get_tensor(key)
        return out

    safetensors_index = source / "model.safetensors.index.json"
    if safetensors_index.exists():
        with open(safetensors_index, "r", encoding="utf-8") as fh:
            index = json.load(fh)
        shards = sorted(set(index["weight_map"].values()))
        out = {}
        for shard in shards:
            with safe_open(str(source / shard), framework="pt", device="cpu") as f:
                for key in f.keys():
                    out[key] = f.get_tensor(key)
        return out

    bin_path = source / "pytorch_model.bin"
    if bin_path.exists():
        return torch.load(bin_path, map_location="cpu")

    bin_index = source / "pytorch_model.bin.index.json"
    if bin_index.exists():
        with open(bin_index, "r", encoding="utf-8") as fh:
            index = json.load(fh)
        shards = sorted(set(index["weight_map"].values()))
        out = {}
        for shard in shards:
            out.update(torch.load(source / shard, map_location="cpu"))
        return out

    raise FileNotFoundError(
        f"No model.safetensors[.index.json] or pytorch_model.bin[.index.json] under {source}"
    )


def expand_qkv(old: torch.Tensor, hidden_size: int) -> torch.Tensor:
    new = old.new_zeros((old.shape[0], hidden_size * 2))
    new[:, hidden_size:] = old
    return new


def make_fc_weight(config: Dict) -> torch.Tensor:
    """Initialize ``fc``, which maps the three concatenated target aux hidden
    states (``3 * target_hidden_size``) to the draft hidden size.

    Identity on the last aux block, the paper's setting: at initialization
    ``fc(aux)`` is the deepest tapped target hidden state (truncated to the draft
    width). Function preservation does not depend on this -- it comes from the
    zero target-feature half of Q/K/V, which multiplies ``fc(aux)`` by zero. What
    the identity block buys is a gradient path: with a non-zero ``fc(aux)`` the
    zero Q/K/V half receives gradient from step 0 and the drafter starts reading
    the target. An all-zero ``fc`` together with that zero half is a stationary
    point: ``RMSNorm(0) = 0``, so both ``dL/dW_qkv[:, :h]`` and ``dL/dfc`` vanish
    identically and the target-feature path never trains.
    """
    hidden_size = config["hidden_size"]
    target_hidden_size = config.get("target_hidden_size", hidden_size)
    dtype = getattr(torch, config.get("torch_dtype", "bfloat16"), torch.bfloat16)
    fc = torch.zeros((hidden_size, target_hidden_size * 3), dtype=dtype)
    block = 2  # the deepest of the three tapped layers
    shared = min(hidden_size, target_hidden_size)
    start = block * target_hidden_size
    fc[:shared, start : start + shared] = torch.eye(shared, dtype=dtype)
    return fc


def parse_aux_layer_ids(value: str):
    if value is None:
        return None
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def convert_state(
    source_state: Dict[str, torch.Tensor],
    config: Dict,
    draft_vocab_size: int,
) -> Dict[str, torch.Tensor]:
    hidden_size = config["hidden_size"]
    num_layers = config["num_hidden_layers"]
    target_prefix = "layers" if num_layers > 1 else "midlayer"
    out: Dict[str, torch.Tensor] = {}

    embed = source_state["model.embed_tokens.weight"][:draft_vocab_size].contiguous()
    out["embed_tokens.weight"] = embed

    if "lm_head.weight" in source_state:
        source_head = source_state["lm_head.weight"]
    else:
        source_head = embed
    out["lm_head.weight"] = source_head[:draft_vocab_size].clone().contiguous()

    out["norm.weight"] = source_state["model.norm.weight"].contiguous()
    out["fc.weight"] = make_fc_weight(config)

    for layer_idx in range(num_layers):
        src = f"model.layers.{layer_idx}"
        dst = f"{target_prefix}.{layer_idx}" if num_layers > 1 else target_prefix

        out[f"{dst}.self_attn.q_proj.weight"] = expand_qkv(
            source_state[f"{src}.self_attn.q_proj.weight"], hidden_size
        )
        out[f"{dst}.self_attn.k_proj.weight"] = expand_qkv(
            source_state[f"{src}.self_attn.k_proj.weight"], hidden_size
        )
        out[f"{dst}.self_attn.v_proj.weight"] = expand_qkv(
            source_state[f"{src}.self_attn.v_proj.weight"], hidden_size
        )
        out[f"{dst}.self_attn.o_proj.weight"] = source_state[
            f"{src}.self_attn.o_proj.weight"
        ].contiguous()

        input_norm = source_state[f"{src}.input_layernorm.weight"].contiguous()
        out[f"{dst}.hidden_norm.weight"] = input_norm.clone()
        out[f"{dst}.input_layernorm.weight"] = input_norm.clone()
        out[f"{dst}.post_attention_layernorm.weight"] = source_state[
            f"{src}.post_attention_layernorm.weight"
        ].contiguous()

        for name in ("gate_proj", "up_proj", "down_proj"):
            out[f"{dst}.mlp.{name}.weight"] = source_state[
                f"{src}.mlp.{name}.weight"
            ].contiguous()

    t2d = torch.zeros(config["vocab_size"], dtype=torch.bool)
    d2t = torch.zeros(draft_vocab_size, dtype=torch.int64)
    t2d[:draft_vocab_size] = True
    out["t2d"] = t2d
    out["d2t"] = d2t

    return out


def make_eagle_config(
    config: Dict,
    draft_vocab_size: int,
    num_draft_layers: int,
    aux_layer_ids,
    target_hidden_size: int,
) -> Dict:
    eagle = dict(config)
    eagle["num_hidden_layers"] = num_draft_layers
    eagle["architectures"] = ["LlamaForCausalLMEagle3"]
    eagle["draft_vocab_size"] = draft_vocab_size
    eagle["tie_word_embeddings"] = False
    eagle["use_cache"] = True
    eagle.setdefault("pretraining_tp", 1)
    if target_hidden_size is not None:
        eagle["target_hidden_size"] = target_hidden_size
    if aux_layer_ids is not None:
        eagle["eagle_config"] = {
            "eagle_aux_hidden_state_layer_ids": aux_layer_ids,
            "use_aux_hidden_state": True,
        }
    eagle["swap_h_and_emb"] = True
    return eagle


def copy_tokenizer_files(source: Path, output: Path):
    for name in (
        "tokenizer.json",
        "tokenizer.model",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "added_tokens.json",
        "vocab.json",
        "merges.txt",
        "generation_config.json",
    ):
        src = source / name
        if src.exists():
            shutil.copy2(src, output / name)


def main():
    args = parse_args()
    source = Path(args.source)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    config = load_config(source)
    draft_vocab_size = args.draft_vocab_size or config["vocab_size"]
    if draft_vocab_size > config["vocab_size"]:
        raise ValueError(
            f"draft_vocab_size={draft_vocab_size} exceeds vocab_size={config['vocab_size']}"
        )

    state = load_state_dict(source)
    num_source_layers = config["num_hidden_layers"]
    num_draft_layers = args.num_draft_layers or num_source_layers
    if num_draft_layers > num_source_layers:
        raise ValueError(
            "Requested layers exceed source layer count: "
            f"num_draft_layers={num_draft_layers}, "
            f"source_layers={num_source_layers}"
        )
    aux_layer_ids = parse_aux_layer_ids(args.aux_layer_ids)
    if aux_layer_ids is not None and len(aux_layer_ids) != 3:
        raise ValueError("--aux-layer-ids must contain exactly 3 ids")
    if aux_layer_ids is not None and any(idx < 0 for idx in aux_layer_ids):
        raise ValueError("--aux-layer-ids must be non-negative target layer ids")

    eagle_config = make_eagle_config(
        config,
        draft_vocab_size,
        num_draft_layers=num_draft_layers,
        aux_layer_ids=aux_layer_ids,
        target_hidden_size=args.target_hidden_size,
    )
    eagle_state = convert_state(
        source_state=state,
        config=eagle_config,
        draft_vocab_size=draft_vocab_size,
    )

    with open(output / "config.json", "w", encoding="utf-8") as f:
        json.dump(eagle_config, f, indent=2)
        f.write("\n")
    save_file(eagle_state, output / "model.safetensors", metadata={"format": "pt"})
    copy_tokenizer_files(source, output)

    print(f"Saved layer-preserving Eagle-3 checkpoint to {output}")
    print(f"Source layers copied: 0..{num_draft_layers - 1}")
    print(f"draft_vocab_size: {draft_vocab_size}")
    print(
        "target_hidden_size: "
        f"{args.target_hidden_size or eagle_config['hidden_size']}"
    )
    print(
        "aux layer ids: "
        f"{aux_layer_ids if aux_layer_ids is not None else 'target default'}"
    )
    print(
        "Osprey init: identity fc on the last aux block, zero target-feature half of Q/K/V, embedding-rooted residual"
    )


if __name__ == "__main__":
    main()
