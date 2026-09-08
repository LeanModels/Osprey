"""Global-MMLU: multiple-choice knowledge questions in 42 languages.

Two registered variants, matching the paper's Table 2:

  `global_mmlu`         - question in the local language, answer instruction in
                          English, so the model may reply in English.
  `global_mmlu_native`  - additionally instructs the model to answer in the
                          question's own language. This is the harder setting;
                          it narrows Osprey's margin but does not remove it.

Prompts are sampled round-robin across languages so a small `num_samples` still
covers every language.
"""

from typing import Any, Dict, List, Optional, Tuple

from datasets import load_dataset

from .base import Benchmarker
from .registry import BENCHMARKS
from .utils import create_simple_sgl_function

# The 42 Global-MMLU languages, as released by CohereLabs.
# fmt: off
GLOBAL_MMLU_LANGUAGES = [
    "am", "ar", "bn", "cs", "de", "el", "en", "es", "fa", "fil", "fr", "ha",
    "he", "hi", "id", "ig", "it", "ja", "ko", "ky", "lt", "mg", "ms", "ne",
    "nl", "ny", "pl", "pt", "ro", "ru", "si", "sn", "so", "sr", "sv", "sw",
    "te", "tr", "uk", "vi", "yo", "zh",
]
# fmt: on

QUERY_TEMPLATE = """{question}

A) {a}
B) {b}
C) {c}
D) {d}

{instruction}"""

ENGLISH_INSTRUCTION = (
    "Think step by step, then end your response with a final line of the form "
    "'Answer: $LETTER' where $LETTER is one of A, B, C, D."
)

NATIVE_INSTRUCTION = (
    "Reply in the same language as the question above. Think step by step, then "
    "end your response with a final line of the form 'Answer: $LETTER' where "
    "$LETTER is one of A, B, C, D."
)


class _GlobalMMLUBenchmarker(Benchmarker):
    """Shared loader; subclasses only pick the answer-language instruction."""

    instruction: str = ENGLISH_INSTRUCTION

    def __init__(
        self, num_samples: Optional[int] = None, subset: Optional[List[str]] = None
    ):
        super().__init__(num_samples, subset or GLOBAL_MMLU_LANGUAGES)

    def load_data(self) -> Tuple[List[Dict[str, Any]], List[Optional[str]]]:
        per_language = []
        for language in self.subset:
            rows = load_dataset("CohereLabs/Global-MMLU", language)["test"]
            per_language.append(
                [
                    (
                        {
                            "question": QUERY_TEMPLATE.format(
                                question=row["question"].strip(),
                                a=row["option_a"],
                                b=row["option_b"],
                                c=row["option_c"],
                                d=row["option_d"],
                                instruction=self.instruction,
                            ),
                            "lang": language,
                        },
                        str(row["answer"]).strip().upper(),
                    )
                    for row in rows
                ]
            )

        questions: List[Dict[str, Any]] = []
        labels: List[Optional[str]] = []
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

    def extract_answer(self, output: str, label: Optional[Any] = None) -> Optional[str]:
        if "Answer:" not in output:
            return None
        tail = output.rsplit("Answer:", 1)[-1].strip()
        return tail[:1].upper() if tail[:1].upper() in "ABCD" else None

    def compute_accuracy(
        self, predictions: List[Any], labels: List[Any]
    ) -> Optional[float]:
        if not labels:
            return None
        correct = sum(1 for p, label in zip(predictions, labels) if p == label)
        return correct / len(labels)

    def create_sgl_function(self):
        return create_simple_sgl_function(
            function_name="get_" + self.__class__.__name__,
            answer_key="answer",
            max_tokens=self.get_max_new_tokens(),
        )

    def get_max_new_tokens(self) -> int:
        """The paper evaluates every public benchmark at max_tokens=4096."""
        if getattr(self, "_max_new_tokens", None) is not None:
            return self._max_new_tokens
        return 4096


@BENCHMARKS.register("global_mmlu")
class GlobalMMLUBenchmarker(_GlobalMMLUBenchmarker):
    instruction = ENGLISH_INSTRUCTION


@BENCHMARKS.register("global_mmlu_native")
class GlobalMMLUNativeBenchmarker(_GlobalMMLUBenchmarker):
    instruction = NATIVE_INSTRUCTION
