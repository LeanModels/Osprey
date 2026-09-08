"""MGSM: grade-school math word problems in 11 languages.

Used for the multilingual half of the paper's MiniMax-M2.5 evaluation. Prompts
are sampled round-robin across languages so a small `num_samples` still covers
every language rather than exhausting the first one.
"""

import re
from typing import Any, Dict, List, Optional, Tuple

from datasets import load_dataset

from .base import Benchmarker
from .registry import BENCHMARKS
from .utils import create_simple_sgl_function

# The 11 MGSM languages, in the order the benchmark defines them.
# fmt: off
MGSM_LANGUAGES = ["en", "es", "fr", "de", "ru", "zh", "ja", "th", "sw", "bn", "te"]
# fmt: on

INSTRUCTION = (
    "\n\nPlease reason step by step, and put your final numeric answer after "
    '"Answer:".'
)


def extract_number(output: str) -> Optional[float]:
    """Last number in the output, preferring one that follows 'Answer:'."""
    tail = output.rsplit("Answer:", 1)[-1] if "Answer:" in output else output
    numbers = re.findall(r"-?\d[\d,]*\.?\d*", tail)
    if not numbers:
        return None
    try:
        return float(numbers[-1].replace(",", ""))
    except ValueError:
        return None


@BENCHMARKS.register("mgsm")
class MGSMBenchmarker(Benchmarker):
    """MGSM benchmark implementation (juletxara/mgsm, test split)."""

    def __init__(
        self, num_samples: Optional[int] = None, subset: Optional[List[str]] = None
    ):
        super().__init__(num_samples, subset or MGSM_LANGUAGES)

    def load_data(self) -> Tuple[List[Dict[str, Any]], List[Optional[float]]]:
        per_language = []
        for language in self.subset:
            rows = load_dataset("juletxara/mgsm", language)["test"]
            per_language.append(
                [
                    (
                        {"question": row["question"] + INSTRUCTION, "lang": language},
                        float(row["answer_number"]),
                    )
                    for row in rows
                ]
            )

        # Round-robin so `num_samples` spreads across languages.
        questions: List[Dict[str, Any]] = []
        labels: List[Optional[float]] = []
        for index in range(max(len(rows) for rows in per_language)):
            for rows in per_language:
                if index >= len(rows):
                    continue
                if self.num_samples is not None and len(questions) >= self.num_samples:
                    return questions, labels
                question, label = rows[index]
                questions.append(question)
                labels.append(label)
        return questions, labels

    def extract_answer(
        self, output: str, label: Optional[Any] = None
    ) -> Optional[float]:
        return extract_number(output)

    def compute_accuracy(
        self, predictions: List[Any], labels: List[Any]
    ) -> Optional[float]:
        if not labels:
            return None
        correct = sum(
            1
            for prediction, label in zip(predictions, labels)
            if prediction is not None and abs(prediction - label) < 1e-4
        )
        return correct / len(labels)

    def create_sgl_function(self):
        return create_simple_sgl_function(
            function_name="get_mgsm_answer",
            answer_key="answer",
            max_tokens=self.get_max_new_tokens(),
        )

    def get_max_new_tokens(self) -> int:
        """The paper evaluates every public benchmark at max_tokens=4096."""
        if getattr(self, "_max_new_tokens", None) is not None:
            return self._max_new_tokens
        return 4096
