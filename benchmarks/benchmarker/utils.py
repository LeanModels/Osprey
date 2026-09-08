"""
Utility functions for benchmark scripts.
"""

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import sglang as sgl


@dataclass
class BenchmarkMetrics:
    """Container for benchmark performance metrics."""

    latency: float
    output_throughput: float
    accept_length: float
    accuracy: Optional[float] = None
    num_questions: int = 0
    num_valid_predictions: int = 0
    categorical_performance: Optional[Dict[str, "BenchmarkMetrics"]] = None
    # Per-request breakdown. `accept_length` above is the mean of
    # per_request_accept_length, so a few very long generations cannot dominate
    # the reported number the way a corpus-level ratio would.
    per_request_accept_length: Optional[List[float]] = None
    per_request_output_length: Optional[List[int]] = None
    per_request_latency: Optional[List[float]] = None
    per_request_tps: Optional[List[float]] = None
    mean_per_request_tps: Optional[float] = None
    accept_length_std: Optional[float] = None
    accept_length_p05: Optional[float] = None
    accept_length_p20: Optional[float] = None
    accept_length_p50: Optional[float] = None
    accept_length_p80: Optional[float] = None
    accept_length_p95: Optional[float] = None
    output_length_mean: Optional[float] = None
    output_length_std: Optional[float] = None
    output_length_p05: Optional[float] = None
    output_length_p50: Optional[float] = None
    output_length_p95: Optional[float] = None


def compute_metrics(
    states: List[Any],
    latency: float,
    answer_key: str = "answer",
    additional_answer_keys: Optional[List[str]] = None,
) -> BenchmarkMetrics:
    """
    Compute performance metrics from SGLang states.

    Args:
        states: List of SGLang state objects from run_batch
        latency: Total latency in seconds
        answer_key: Primary key for answer in state meta info
        additional_answer_keys: Additional keys to include in token count (e.g., ["answer_1", "answer_2"])

    Returns:
        BenchmarkMetrics object with computed metrics
    """
    # Compute output tokens
    num_output_tokens = 0
    if additional_answer_keys:
        for key in [answer_key] + additional_answer_keys:
            num_output_tokens += sum(
                s.get_meta_info(key)["completion_tokens"] for s in states
            )
    else:
        num_output_tokens = sum(
            s.get_meta_info(answer_key)["completion_tokens"] for s in states
        )

    output_throughput = num_output_tokens / latency if latency > 0 else 0.0

    all_keys = [answer_key] + (additional_answer_keys or [])
    per_request_output_length = [
        sum(s.get_meta_info(key)["completion_tokens"] for key in all_keys)
        for s in states
    ]

    # Accepted length, computed per request and then averaged. This is the
    # definition used for every number reported in the paper.
    has_verify = "spec_verify_ct" in states[0].get_meta_info(answer_key)
    if has_verify:
        per_request_accept_length = []
        for tokens, s in zip(per_request_output_length, states):
            verify_ct = sum(
                s.get_meta_info(key).get("spec_verify_ct", 0) for key in all_keys
            )
            per_request_accept_length.append(tokens / verify_ct if verify_ct else 1.0)
        accept_length = float(np.mean(per_request_accept_length))
    else:
        per_request_accept_length = [1.0] * len(states)
        accept_length = 1.0

    return BenchmarkMetrics(
        latency=latency,
        output_throughput=output_throughput,
        accept_length=accept_length,
        num_questions=len(states),
        **summarize_distributions(per_request_accept_length, per_request_output_length),
    )


def summarize_distributions(
    per_request_accept_length: List[float],
    per_request_output_length: List[int],
) -> Dict[str, Any]:
    """Percentile summary of the per-request accept-length / output-length lists.

    Returned as a kwargs dict for the `BenchmarkMetrics` fields, so `run()` and
    both report the same shape.
    """
    al = np.percentile(per_request_accept_length, [5, 20, 50, 80, 95])
    ol = np.percentile(per_request_output_length, [5, 50, 95])
    return dict(
        per_request_accept_length=per_request_accept_length,
        per_request_output_length=per_request_output_length,
        accept_length_std=float(np.std(per_request_accept_length)),
        accept_length_p05=float(al[0]),
        accept_length_p20=float(al[1]),
        accept_length_p50=float(al[2]),
        accept_length_p80=float(al[3]),
        accept_length_p95=float(al[4]),
        output_length_mean=float(np.mean(per_request_output_length)),
        output_length_std=float(np.std(per_request_output_length)),
        output_length_p05=float(ol[0]),
        output_length_p50=float(ol[1]),
        output_length_p95=float(ol[2]),
    )


def create_simple_sgl_function(
    function_name: str = "get_answer",
    answer_key: str = "answer",
    system_prompt: Optional[str] = None,
    max_tokens: int = 2048,
    stop: Optional[List[str]] = None,
    user_prefix: Optional[str] = None,
) -> Callable:
    """
    Create a simple SGL function for single-turn Q&A.

    Args:
        function_name: Name of the function
        answer_key: Key for storing the answer
        system_prompt: Optional system prompt
        max_tokens: Maximum tokens to generate
        stop: Optional stop sequences
        user_prefix: Optional suffix to append to user message (appended after question)

    Returns:
        SGL function decorated with @sgl.function
    """

    @sgl.function
    def sgl_func(s, question):
        if system_prompt:
            s += sgl.system(system_prompt)
        user_content = question
        if user_prefix:
            user_content = question + user_prefix
        s += sgl.user(user_content)
        # Greedy decoding: acceptance length must not vary with sampling noise.
        gen_kwargs = {"max_tokens": max_tokens, "temperature": 0.0}
        if stop:
            gen_kwargs["stop"] = stop
        s += sgl.assistant(sgl.gen(answer_key, **gen_kwargs))

    sgl_func.__name__ = function_name
    return sgl_func


def create_multi_turn_sgl_function(
    function_name: str = "multi_turn_answer",
    system_prompt: Optional[str] = None,
    num_turns: int = 2,
    max_tokens: int = 2048,
) -> Callable:
    """
    Create an SGL function for multi-turn conversations (e.g., MT-Bench with 2 turns).

    Args:
        function_name: Name of the function
        system_prompt: Optional system prompt
        num_turns: Number of conversation turns (default: 2)
        max_tokens: Maximum tokens to generate per turn

    Returns:
        SGL function decorated with @sgl.function
    """
    if num_turns == 2:
        # Most common case: 2-turn conversation
        @sgl.function
        def sgl_func(s, question_1, question_2):
            if system_prompt:
                s += sgl.system(system_prompt)
            s += sgl.user(question_1)
            s += sgl.assistant(sgl.gen("answer_1", max_tokens=max_tokens))
            s += sgl.user(question_2)
            s += sgl.assistant(sgl.gen("answer_2", max_tokens=max_tokens))

    else:
        # Generic case: create function with dynamic number of turns
        # Note: This requires the caller to pass arguments as a dict
        @sgl.function
        def sgl_func(s, **kwargs):
            if system_prompt:
                s += sgl.system(system_prompt)
            for i in range(num_turns):
                question_key = f"question_{i+1}"
                answer_key = f"answer_{i+1}"
                if question_key in kwargs:
                    s += sgl.user(kwargs[question_key])
                    s += sgl.assistant(sgl.gen(answer_key, max_tokens=max_tokens))

    sgl_func.__name__ = function_name
    return sgl_func
