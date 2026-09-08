#!/usr/bin/env python3
"""Align an Osprey drafter's embedding and LM head to a target tokenizer."""

import argparse
import json
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file
from tqdm import tqdm
from transformers import AutoConfig, AutoTokenizer


def load_state(path: Path) -> dict[str, torch.Tensor]:
    with safe_open(str(path / "model.safetensors"), framework="pt") as handle:
        return {key: handle.get_tensor(key) for key in handle.keys()}


def target_vocab_size(model: str, tokenizer) -> int:
    size = len(tokenizer.get_vocab())
    try:
        config = AutoConfig.from_pretrained(model, trust_remote_code=True)
        text_config = getattr(config, "text_config", config)
        size = max(size, int(getattr(text_config, "vocab_size", 0) or 0))
    except Exception:
        pass
    return size


def space_marker(tokenizer) -> str | None:
    vocab = tokenizer.get_vocab()
    counts = {
        marker: sum(token.startswith(marker) for token in vocab)
        for marker in ("▁", "Ġ")
    }
    marker, count = max(counts.items(), key=lambda item: item[1])
    return marker if count / max(len(vocab), 1) > 0.2 else None


def _rewrite_generation_config(
    source: Path, output: Path, special_token_ids: dict[str, int | None]
) -> None:
    """Carry generation_config.json over with the target's special-token ids."""
    generation_config_path = source / "generation_config.json"
    if not generation_config_path.exists():
        return
    with open(generation_config_path, encoding="utf-8") as handle:
        generation_config = json.load(handle)
    for key, value in special_token_ids.items():
        if value is not None:
            generation_config[key] = value
        else:
            generation_config.pop(key, None)
    with open(output / "generation_config.json", "w", encoding="utf-8") as handle:
        json.dump(generation_config, handle, indent=2)
        handle.write("\n")


def align(source: Path, target_model: str, output: Path) -> dict[str, int]:
    output.mkdir(parents=True, exist_ok=True)
    state = load_state(source)
    with open(source / "config.json", encoding="utf-8") as handle:
        config = json.load(handle)

    source_tokenizer = AutoTokenizer.from_pretrained(source, trust_remote_code=True)
    target_tokenizer = AutoTokenizer.from_pretrained(
        target_model, trust_remote_code=True
    )
    source_vocab = source_tokenizer.get_vocab()
    size = target_vocab_size(target_model, target_tokenizer)
    embedding = state["embed_tokens.weight"].cpu()
    lm_head = state["lm_head.weight"].cpu()
    new_embedding = embedding.new_zeros((size, embedding.shape[1]))
    new_lm_head = lm_head.new_zeros((size, lm_head.shape[1]))
    source_marker = space_marker(source_tokenizer)
    target_marker = space_marker(target_tokenizer)
    stats = {"direct": 0, "single": 0, "averaged": 0, "unmapped": 0}

    for token, target_id in tqdm(
        target_tokenizer.get_vocab().items(), desc="align tokenizer"
    ):
        if target_id >= size:
            continue
        source_token = token
        if source_marker and target_marker and source_marker != target_marker:
            source_token = source_token.replace(target_marker, source_marker)
        source_id = source_vocab.get(source_token)
        if source_id is not None:
            ids = [source_id]
            stats["direct"] += 1
        else:
            text = (
                " "
                if source_token in {"<0x20>", ""}
                else target_tokenizer.convert_tokens_to_string([token])
            )
            ids = source_tokenizer(text, add_special_tokens=False)["input_ids"]
            if not ids:
                stats["unmapped"] += 1
                continue
            stats["single" if len(ids) == 1 else "averaged"] += 1
        new_embedding[target_id] = embedding[ids].mean(dim=0)
        new_lm_head[target_id] = lm_head[ids].mean(dim=0)

    state["embed_tokens.weight"] = new_embedding
    state["lm_head.weight"] = new_lm_head
    state["t2d"] = torch.ones(size, dtype=torch.bool)
    # d2t stores target-id offsets for the compressed draft vocabulary.
    # With a full 1:1 vocabulary every offset is zero.
    state["d2t"] = torch.zeros(size, dtype=torch.int64)
    config["vocab_size"] = size
    config["draft_vocab_size"] = size

    # Rewrite the special-token ids to the target's. Prefer the target's
    # AutoConfig, since a tokenizer often leaves these unset (Qwen3's
    # tokenizer.bos_token_id is None even though its config.json defines one).
    # A key the target does not define must be *deleted*, not left behind:
    # the source backbone's id would otherwise survive into a checkpoint with a
    # different vocabulary, and a stale pad_token_id silently freezes that row
    # of the draft's embedding.
    target_config = AutoConfig.from_pretrained(target_model, trust_remote_code=True)
    special_token_ids = {}
    for key in ("bos_token_id", "eos_token_id", "pad_token_id"):
        value = getattr(target_config, key, None)
        if value is None:
            value = getattr(target_tokenizer, key, None)
        special_token_ids[key] = value
        if value is not None:
            config[key] = value
        else:
            config.pop(key, None)

    save_file(state, str(output / "model.safetensors"), metadata={"format": "pt"})
    with open(output / "config.json", "w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2)
        handle.write("\n")
    _rewrite_generation_config(source, output, special_token_ids)
    target_tokenizer.save_pretrained(output)
    with open(output / "tokenizer_alignment.json", "w", encoding="utf-8") as handle:
        json.dump(
            {"target": target_model, "vocab_size": size, **stats}, handle, indent=2
        )
        handle.write("\n")
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--target-tokenizer", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(align(args.source, args.target_tokenizer, args.output))


if __name__ == "__main__":
    main()
