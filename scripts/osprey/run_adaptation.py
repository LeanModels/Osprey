#!/usr/bin/env python3
"""Launch Osprey Stage 4: per-target on-policy EAGLE-3 distillation.

Holds the paper's hyperparameters so the launch scripts and the README cannot
drift from the trainer. Pick a target with `--target`; everything the paper
varies (draft checkpoint, training split, learning rate) is a flag.

    python scripts/osprey/run_adaptation.py --target qwen3-8b \\
        --draft-checkpoint <converted-dir> \\
        --train-data <domain>_train_65k_qwen3_8B_4096.jsonl \\
        --output-dir <out>

Pass `--dry-run` to print the torchrun command without launching it.
"""

import argparse
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List


@dataclass(frozen=True)
class TargetPreset:
    """Per-target settings from the paper (Sec. 4 and Appendix D)."""

    model: str
    chat_template: str
    #: GPUs for the run. The trainer gives every (tp_rank, dp_rank) pair a
    #: distinct prompt, so the effective batch equals num_gpus and the step
    #: count is num_samples * num_epochs / num_gpus.
    num_gpus: int
    #: Tensor parallel for the *target* model served in-process by SGLang.
    tp_size: int
    #: Expert parallel; only meaningful for an MoE target.
    ep_size: int = 1
    sglang_mem_fraction_static: float = 0.4
    extra_args: List[str] = field(default_factory=list)


TARGETS: Dict[str, TargetPreset] = {
    # Sec. 4.1 / Appendix D: bf16 Qwen3-8B on a single node, target tp=1.
    # 65k samples x 3 epochs / 2 GPUs = 97,500 steps.
    "qwen3-8b": TargetPreset(
        model="Qwen/Qwen3-8B",
        chat_template="qwen3-thinking",
        num_gpus=2,
        tp_size=1,
    ),
    # Sec. 4.2 / Appendix D: 229B FP8 MoE, tp=4 and ep=4 on one 4-GPU node.
    # trust-remote-code is required by the MiniMax checkpoint.
    "minimax-m25": TargetPreset(
        model="MiniMaxAI/MiniMax-M2.5",
        chat_template="minimax-m25",
        num_gpus=4,
        tp_size=4,
        ep_size=4,
        sglang_mem_fraction_static=0.78,
        extra_args=["--trust-remote-code"],
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", choices=sorted(TARGETS), default="qwen3-8b")
    parser.add_argument("--draft-checkpoint", required=True)
    parser.add_argument("--train-data", required=True)
    parser.add_argument(
        "--eval-data",
        action="append",
        default=[],
        help="Repeatable. Each split is evaluated separately at --eval-interval.",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--cache-dir", default="./cache/osprey")
    parser.add_argument(
        "--num-gpus",
        type=int,
        default=None,
        help="Overrides the target preset. Changes the effective batch, and "
        "therefore the step count, so leave it alone to match the paper.",
    )
    parser.add_argument("--tp-size", type=int, default=None)
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-4,
        help="Paper default 1e-4; pass 1e-5 for the learning-rate ablation.",
    )
    parser.add_argument("--num-epochs", type=int, default=3)
    parser.add_argument("--ttt-length", type=int, default=5)
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--eval-interval", type=int, default=1000)
    parser.add_argument("--save-interval", type=int, default=5000)
    parser.add_argument(
        "--sglang-attention-backend",
        default=None,
        help="Target-side attention backend. Defaults to fa3, which Hopper "
        "supports; Blackwell (B200) needs flashinfer.",
    )
    parser.add_argument(
        "--from-scratch",
        action="store_true",
        help="Train the EAGLE-3 baseline instead of Osprey: build the draft "
        "from --draft-checkpoint read as a config file, load the target's "
        "embedding, and keep it frozen.",
    )
    parser.add_argument(
        "--report-to", choices=["wandb", "tensorboard", "none"], default="none"
    )
    parser.add_argument("--wandb-project", default=None)
    parser.add_argument("--wandb-name", default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def build_command(args: argparse.Namespace) -> List[str]:
    root = Path(__file__).resolve().parents[2]
    preset = TARGETS[args.target]
    num_gpus = args.num_gpus if args.num_gpus is not None else preset.num_gpus
    tp_size = args.tp_size if args.tp_size is not None else preset.tp_size

    checkpoint = Path(args.draft_checkpoint)
    if args.from_scratch:
        # The baseline has no pretrained body: --draft-model-config is a plain
        # config JSON and there is no --ckpt-dir to resume from.
        draft_config = checkpoint
    else:
        draft_config = checkpoint / "config.json"

    command = [
        "torchrun",
        "--standalone",
        "--nproc_per_node",
        str(num_gpus),
        str(root / "scripts/train_eagle3.py"),
        "--target-model-path",
        preset.model,
        "--target-model-backend",
        "sglang",
        "--draft-model-config",
        str(draft_config),
        "--train-data-path",
        args.train_data,
        "--output-dir",
        args.output_dir,
        "--cache-dir",
        args.cache_dir,
        "--num-epochs",
        str(args.num_epochs),
        "--batch-size",
        "1",
        "--learning-rate",
        str(args.learning_rate),
        "--warmup-ratio",
        "0.04",
        "--max-grad-norm",
        "1.0",
        "--max-length",
        str(args.max_length),
        "--ttt-length",
        str(args.ttt_length),
        "--chat-template",
        preset.chat_template,
        "--tp-size",
        str(tp_size),
        "--attention-backend",
        "flex_attention",
        "--eval-interval",
        str(args.eval_interval),
        "--save-interval",
        str(args.save_interval),
        "--report-to",
        args.report_to,
    ]

    if args.from_scratch:
        # EAGLE-3's own convention: the draft shares the target's embedding and
        # does not train it.
        command += ["--embedding-key", "model.embed_tokens.weight"]
    else:
        command += ["--ckpt-dir", args.draft_checkpoint, "--train-with-embeddings"]

    if preset.ep_size > 1:
        command += ["--sglang-ep-size", str(preset.ep_size)]
    command += [
        "--sglang-mem-fraction-static",
        str(preset.sglang_mem_fraction_static),
    ]
    if args.sglang_attention_backend:
        command += ["--sglang-attention-backend", args.sglang_attention_backend]
    command += preset.extra_args

    if args.eval_data:
        command += ["--eval-data-path", ",".join(args.eval_data)]
    if args.report_to == "wandb":
        if not args.wandb_project or not args.wandb_name:
            raise ValueError("W&B reporting requires --wandb-project and --wandb-name")
        command += [
            "--wandb-project",
            args.wandb_project,
            "--wandb-name",
            args.wandb_name,
        ]
    return command


def main() -> None:
    args = parse_args()
    command = build_command(args)
    print(" ".join(command))
    if not args.dry_run:
        subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
