#!/usr/bin/env python3
"""Osprey Stage 3: convert a shallow pretrained backbone into an EAGLE-3 drafter.

Two steps, in order:

1. Keep the first `--num-layers` transformer blocks, expand every Q/K/V
   projection from `hidden_size` to `2 * hidden_size` with the pretrained
   weights on the hidden-state half and zeros on the new target-feature
   columns, and zero-initialize the target-feature projection `fc`.
2. When `--target-tokenizer` is set, remap the embedding and LM-head rows into
   the target's vocabulary (`scripts/osprey/align_tokenizer.py`).

`--target` fills in the target's hidden size, tapped layer ids, and tokenizer.
Pass them individually to convert against a target with no preset.

    python scripts/osprey/convert_checkpoint.py \\
        --architecture qwen3 --target qwen3-8b \\
        --source-checkpoint <shallow-ntp-dir> \\
        --output-dir <converted-dir>
"""

import argparse
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional


@dataclass(frozen=True)
class TargetPreset:
    """Interface a drafter must match to condition on this target."""

    hidden_size: int
    tokenizer: str
    #: Target layers whose hidden states are tapped (low / mid / high). None
    #: leaves the trainer's default for the target, which is what the paper's
    #: Qwen3-8B drafters use.
    aux_layer_ids: Optional[str] = None


TARGETS: Dict[str, TargetPreset] = {
    "qwen3-8b": TargetPreset(hidden_size=4096, tokenizer="Qwen/Qwen3-8B"),
    # 62-layer MoE; these are the ids the released MiniMax drafter was
    # converted and distilled with.
    "minimax-m25": TargetPreset(
        hidden_size=3072,
        tokenizer="MiniMaxAI/MiniMax-M2.5",
        aux_layer_ids="1,30,58",
    ),
}

# Layers kept from each source backbone (paper Sec. 3.1 / Appendix C).
DEFAULT_NUM_LAYERS = {"qwen3": 2, "layerskip": 4}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--architecture",
        choices=sorted(DEFAULT_NUM_LAYERS),
        required=True,
        help="qwen3 for the 2-layer Qwen3-4B backbone, layerskip for the "
        "4-layer LayerSkip-Llama-3.2-1B backbone.",
    )
    parser.add_argument("--source-checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--target",
        choices=sorted(TARGETS),
        default=None,
        help="Fills in --target-hidden-size, --aux-layer-ids and "
        "--target-tokenizer. Individual flags override it.",
    )
    parser.add_argument("--target-hidden-size", type=int, default=None)
    parser.add_argument(
        "--target-tokenizer",
        default=None,
        help="Target model or tokenizer whose vocabulary the embedding and "
        "LM head are remapped into. Omit to keep the backbone's own.",
    )
    parser.add_argument(
        "--aux-layer-ids",
        default=None,
        help="Comma-separated low,mid,high target layer ids to pin in the "
        "draft config. Omit to use the trainer's default for the target.",
    )
    parser.add_argument(
        "--num-layers",
        type=int,
        default=None,
        help="Transformer blocks to keep. Defaults to 2 (qwen3) or 4 "
        "(layerskip), matching the paper.",
    )
    parser.add_argument("--draft-vocab-size", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def resolve(args: argparse.Namespace) -> argparse.Namespace:
    """Apply the target preset to any flag the caller left unset."""
    preset = TARGETS[args.target] if args.target else None
    if args.target_hidden_size is None:
        if preset is None:
            raise SystemExit("--target-hidden-size is required without --target")
        args.target_hidden_size = preset.hidden_size
    if args.target_tokenizer is None and preset is not None:
        args.target_tokenizer = preset.tokenizer
    if args.aux_layer_ids is None and preset is not None:
        args.aux_layer_ids = preset.aux_layer_ids
    if args.num_layers is None:
        args.num_layers = DEFAULT_NUM_LAYERS[args.architecture]
    return args


def build_commands(args: argparse.Namespace) -> List[List[str]]:
    root = Path(__file__).resolve().parents[2]
    converter = (
        root / "scripts/convert_qwen3_to_eagle3_preserve_layers.py"
        if args.architecture == "qwen3"
        else root / "scripts/convert_llama_to_eagle3_preserve_layers.py"
    )
    command = [
        sys.executable,
        str(converter),
        "--source",
        args.source_checkpoint,
        "--output",
        args.output_dir,
        "--num-draft-layers",
        str(args.num_layers),
        "--target-hidden-size",
        str(args.target_hidden_size),
    ]
    if args.draft_vocab_size is not None:
        command += ["--draft-vocab-size", str(args.draft_vocab_size)]
    if args.aux_layer_ids:
        command += ["--aux-layer-ids", args.aux_layer_ids]

    commands = [command]
    if args.target_tokenizer:
        # Runs in place on the freshly converted directory.
        commands.append(
            [
                sys.executable,
                str(root / "scripts/osprey/align_tokenizer.py"),
                "--source",
                args.output_dir,
                "--target-tokenizer",
                args.target_tokenizer,
                "--output",
                args.output_dir,
            ]
        )
    return commands


def main() -> None:
    args = resolve(parse_args())
    for command in build_commands(args):
        print(" ".join(command))
        if not args.dry_run:
            subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
