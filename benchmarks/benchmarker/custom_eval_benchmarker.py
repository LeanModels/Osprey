"""Benchmark adapters for the five-domain Osprey evaluation set."""

import json
import os
from typing import Any, Dict, List, Tuple

from .base import Benchmarker
from .registry import BENCHMARKS
from .utils import create_simple_sgl_function


class _DomainEvalBenchmarker(Benchmarker):
    """Loads questions from the user message of each record in a domain eval jsonl."""

    domain: str = ""

    def _data_path(self) -> str:
        data_dir = os.environ.get("BENCH_EVAL_DATA_DIR")
        if not data_dir:
            raise ValueError(
                "The five-domain benchmarks require --domain-eval-data-dir "
                "or BENCH_EVAL_DATA_DIR."
            )
        suffix = os.environ.get("BENCH_EVAL_SUFFIX", "_4096")
        return os.path.join(data_dir, f"{self.domain}_eval_512_qwen3_8B{suffix}.jsonl")

    def load_data(self) -> Tuple[List[Dict[str, Any]], List[None]]:
        questions: List[Dict[str, Any]] = []
        with open(self._data_path(), encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                convs = rec.get("conversations", [])
                user_content = next(
                    (m.get("content", "") for m in convs if m.get("role") == "user"),
                    "",
                )
                if not user_content:
                    continue
                questions.append({"question": user_content})
                if self.num_samples is not None and len(questions) >= self.num_samples:
                    break
        labels = [None] * len(questions)
        return questions, labels

    def create_sgl_function(self):
        return create_simple_sgl_function(
            function_name="answer_" + self.__class__.__name__,
            answer_key="answer",
            max_tokens=self.get_max_new_tokens(),
        )

    def get_max_new_tokens(self) -> int:
        """The paper evaluates the five domains at max_tokens=4096."""
        if getattr(self, "_max_new_tokens", None) is not None:
            return self._max_new_tokens
        return 4096


@BENCHMARKS.register("finance_eval")
class FinanceEvalBenchmarker(_DomainEvalBenchmarker):
    domain = "finance"


@BENCHMARKS.register("code_eval")
class CodeEvalBenchmarker(_DomainEvalBenchmarker):
    domain = "code"


@BENCHMARKS.register("math_eval")
class MathEvalBenchmarker(_DomainEvalBenchmarker):
    domain = "math"


@BENCHMARKS.register("commonsense_eval")
class CommonsenseEvalBenchmarker(_DomainEvalBenchmarker):
    domain = "commonsense"


@BENCHMARKS.register("chat_eval")
class ChatEvalBenchmarker(_DomainEvalBenchmarker):
    domain = "chat"
