"""Benchmark adapters used by the Osprey evaluation.

Registered names:

  Five-domain cross-domain matrix (Qwen3-8B, paper Figure 2)
    chat_eval, code_eval, commonsense_eval, finance_eval, math_eval

  Public benchmarks (MiniMax-M2.5, paper Table 1)
    humaneval, math500, livecodebench, mtbench
    (commonsense_eval above is the fifth row of that table)

  Multilingual (MiniMax-M2.5, paper Table 2)
    mgsm, global_mmlu, global_mmlu_native
"""

from .custom_eval_benchmarker import (
    ChatEvalBenchmarker,
    CodeEvalBenchmarker,
    CommonsenseEvalBenchmarker,
    FinanceEvalBenchmarker,
    MathEvalBenchmarker,
)
from .global_mmlu import GlobalMMLUBenchmarker, GlobalMMLUNativeBenchmarker
from .humaneval import HumanEvalBenchmarker
from .livecodebench import LCBBenchmarker
from .math500 import Math500Benchmarker
from .mgsm import MGSMBenchmarker
from .mtbench import MTBenchBenchmarker
from .registry import BENCHMARKS

__all__ = [
    "BENCHMARKS",
    "ChatEvalBenchmarker",
    "CodeEvalBenchmarker",
    "CommonsenseEvalBenchmarker",
    "FinanceEvalBenchmarker",
    "MathEvalBenchmarker",
    "GlobalMMLUBenchmarker",
    "GlobalMMLUNativeBenchmarker",
    "HumanEvalBenchmarker",
    "LCBBenchmarker",
    "Math500Benchmarker",
    "MGSMBenchmarker",
    "MTBenchBenchmarker",
]
