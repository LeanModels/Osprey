#!/usr/bin/env python3
"""Convert a chat dataset to the JSONL format expected by SpecForge training.

Input can be either:
  - A local JSONL file (``--input PATH``), one record per line.
  - A HuggingFace dataset (``--hf-dataset NAME``), e.g. ``HuggingFaceH4/ultrachat_200k``.

Each input record is expected to have a list-of-messages column (default
``messages``). Each message is a dict with a role key (default ``role``)
and a content key (default ``content``). ShareGPT-style inputs can be
converted by overriding these, e.g. ``--messages-column conversations
--role-key from --content-key value``.

Alternatively, if the dataset stores user and assistant text in two
separate top-level columns (e.g. ``prompt`` / ``completion`` as in
``HuggingFaceH4/CodeAlpaca_20K``), pass
``--user-content-column prompt --assistant-content-column completion``
to synthesize a two-turn user/assistant conversation per record.

Output is a JSONL file where each record has a ``conversations`` field (or
``--output-column``) holding a list of ``{"role": ..., "content": ...}``
messages, which ``specforge.utils.safe_conversations_generator`` and the
training scripts consume directly.

Examples
--------
HuggingFaceH4/ultrachat_200k (train_sft split) -> training JSONL::

    python data/convert_dataset.py \\
        --hf-dataset HuggingFaceH4/ultrachat_200k \\
        --hf-split train_sft \\
        --output data/ultrachat_sft.jsonl

ShareGPT-style local JSONL (``conversations`` with ``from``/``value``)::

    python data/convert_dataset.py \\
        --input data/sharegpt.jsonl \\
        --messages-column conversations \\
        --role-key from --content-key value \\
        --output data/sharegpt.converted.jsonl
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

DEFAULT_ROLE_MAP = {
    "human": "user",
    "user": "user",
    "gpt": "assistant",
    "assistant": "assistant",
    "chatgpt": "assistant",
    "bard": "assistant",
    "bing": "assistant",
    "system": "system",
    "tool": "tool",
    "function": "tool",
    "observation": "tool",
}


def parse_role_map(pairs: Optional[List[str]]) -> Dict[str, str]:
    mapping = dict(DEFAULT_ROLE_MAP)
    for pair in pairs or []:
        if "=" not in pair:
            raise ValueError(f"--role-map entry must be SRC=DST, got: {pair}")
        src, dst = pair.split("=", 1)
        mapping[src.strip()] = dst.strip()
    return mapping


def convert_turn(
    turn: Dict[str, Any],
    role_key: str,
    content_key: str,
    role_map: Dict[str, str],
    strict: bool,
) -> Dict[str, Any]:
    # Already in role/content form — pass through unchanged.
    if (
        role_key != "role"
        and "role" in turn
        and "content" in turn
        and role_key not in turn
    ):
        return {"role": turn["role"], "content": turn["content"]}

    if role_key not in turn or content_key not in turn:
        raise KeyError(
            f"Turn missing required keys '{role_key}'/'{content_key}': {turn}"
        )

    raw_role = turn[role_key]
    mapped = role_map.get(raw_role)
    if mapped is None:
        if strict:
            raise ValueError(
                f"Unknown role '{raw_role}' "
                f"(pass --role-map {raw_role}=<user|assistant|system|tool>)"
            )
        mapped = raw_role

    return {"role": mapped, "content": turn[content_key]}


def convert_record(
    record: Dict[str, Any],
    messages_column: str,
    output_column: str,
    role_key: str,
    content_key: str,
    role_map: Dict[str, str],
    strict: bool,
    id_column: Optional[str],
    user_content_column: Optional[str] = None,
    assistant_content_column: Optional[str] = None,
) -> Dict[str, Any]:
    if user_content_column is not None:
        # Pair mode: synthesize a two-turn user/assistant conversation from
        # two top-level columns.
        missing = [
            c
            for c in (user_content_column, assistant_content_column)
            if c not in record
        ]
        if missing:
            raise KeyError(
                f"Record missing pair-mode keys {missing}; "
                f"available: {list(record.keys())}"
            )
        converted = [
            {"role": "user", "content": record[user_content_column]},
            {"role": "assistant", "content": record[assistant_content_column]},
        ]
        drop = {user_content_column, assistant_content_column}
        out = {k: v for k, v in record.items() if k not in drop}
    else:
        if messages_column not in record:
            raise KeyError(
                f"Record missing '{messages_column}' key; available: {list(record.keys())}"
            )

        converted = [
            convert_turn(turn, role_key, content_key, role_map, strict)
            for turn in record[messages_column]
        ]

        # Preserve other fields, drop the original messages column so we don't
        # duplicate it under a different name.
        out = {k: v for k, v in record.items() if k != messages_column}

    out[output_column] = converted

    if id_column is not None and id_column in record and "id" not in out:
        out["id"] = record[id_column]

    return out


def iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as fin:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def iter_hf_dataset(
    name: str,
    config: Optional[str],
    split: str,
    cache_dir: Optional[str],
    streaming: bool,
) -> Iterable[Dict[str, Any]]:
    # Lazy import so that the local-JSONL path does not require `datasets`.
    from datasets import load_dataset

    ds = load_dataset(
        name,
        config,
        split=split,
        cache_dir=cache_dir,
        streaming=streaming,
    )
    for row in ds:
        yield dict(row)


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument(
        "--input",
        "-i",
        type=Path,
        default=None,
        help="Path to a local JSONL input file.",
    )
    src.add_argument(
        "--hf-dataset",
        default=None,
        help="HuggingFace dataset identifier, e.g. HuggingFaceH4/ultrachat_200k.",
    )

    ap.add_argument(
        "--hf-config",
        default=None,
        help="Optional dataset config/subset name (for --hf-dataset).",
    )
    ap.add_argument(
        "--hf-split",
        default="train",
        help="Dataset split to load (default: train). Use e.g. train_sft for ultrachat_200k.",
    )
    ap.add_argument(
        "--hf-cache-dir",
        default=None,
        help="Optional HuggingFace datasets cache dir.",
    )
    ap.add_argument(
        "--hf-streaming",
        action="store_true",
        help="Stream the HF dataset instead of downloading the full split.",
    )

    ap.add_argument(
        "--output",
        "-o",
        required=True,
        type=Path,
        help="Output JSONL path.",
    )

    ap.add_argument(
        "--messages-column",
        default="messages",
        help="Input column/key holding the list of turns (default: messages).",
    )
    ap.add_argument(
        "--output-column",
        default="conversations",
        help=(
            "Output column name for the converted turn list "
            "(default: conversations — required by train_eagle3.py)."
        ),
    )
    ap.add_argument(
        "--role-key",
        default="role",
        help="Per-turn key holding the role (default: role).",
    )
    ap.add_argument(
        "--content-key",
        default="content",
        help="Per-turn key holding the content (default: content).",
    )
    ap.add_argument(
        "--user-content-column",
        default=None,
        help=(
            "Pair mode: top-level record column holding the user prompt. "
            "Must be set together with --assistant-content-column. When set, "
            "--messages-column / --role-key / --content-key are ignored and "
            "each record is converted to a two-turn user/assistant conversation."
        ),
    )
    ap.add_argument(
        "--assistant-content-column",
        default=None,
        help="Pair mode: top-level record column holding the assistant response.",
    )
    ap.add_argument(
        "--id-column",
        default=None,
        help=(
            "If set, copy this record column into an 'id' field in the output "
            "(e.g. --id-column prompt_id for ultrachat)."
        ),
    )

    ap.add_argument(
        "--role-map",
        action="append",
        default=[],
        help="Additional role mapping, e.g. --role-map gpt4=assistant (repeatable).",
    )
    ap.add_argument(
        "--strict",
        action="store_true",
        help="Fail on unknown roles instead of passing them through.",
    )
    ap.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only process the first N records (for quick checks).",
    )
    ap.add_argument(
        "--skip-errors",
        action="store_true",
        help="Skip malformed records instead of failing.",
    )
    return ap


def main() -> None:
    args = build_argparser().parse_args()

    if (args.user_content_column is None) != (args.assistant_content_column is None):
        print(
            "--user-content-column and --assistant-content-column must be set together",
            file=sys.stderr,
        )
        sys.exit(1)

    if args.input is not None:
        if not args.input.exists():
            print(f"input not found: {args.input}", file=sys.stderr)
            sys.exit(1)
        source_iter = iter_jsonl(args.input)
        source_desc = str(args.input)
    else:
        source_iter = iter_hf_dataset(
            args.hf_dataset,
            args.hf_config,
            args.hf_split,
            args.hf_cache_dir,
            args.hf_streaming,
        )
        source_desc = (
            f"hf://{args.hf_dataset}"
            + (f":{args.hf_config}" if args.hf_config else "")
            + f"[{args.hf_split}]"
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    role_map = parse_role_map(args.role_map)

    n_in = n_out = n_skip = 0
    with args.output.open("w", encoding="utf-8") as fout:
        for idx, record in enumerate(source_iter, 1):
            if args.limit is not None and n_in >= args.limit:
                break
            n_in += 1
            try:
                converted = convert_record(
                    record,
                    args.messages_column,
                    args.output_column,
                    args.role_key,
                    args.content_key,
                    role_map,
                    args.strict,
                    args.id_column,
                    args.user_content_column,
                    args.assistant_content_column,
                )
            except (KeyError, ValueError, TypeError) as e:
                if args.skip_errors:
                    n_skip += 1
                    continue
                print(f"error on record {idx} from {source_desc}: {e}", file=sys.stderr)
                sys.exit(2)

            fout.write(json.dumps(converted, ensure_ascii=False) + "\n")
            n_out += 1

    print(
        f"source={source_desc} read={n_in} wrote={n_out} skipped={n_skip} -> {args.output}"
    )


if __name__ == "__main__":
    main()
