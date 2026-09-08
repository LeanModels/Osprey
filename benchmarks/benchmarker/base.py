"""
Base class for benchmark implementations.
"""

import time
from abc import ABC, abstractmethod
from argparse import Namespace
from typing import Any, Callable, Dict, List, Optional, Tuple

from .utils import compute_metrics


class Benchmarker(ABC):
    """
    Base class for benchmark implementations.

    Subclasses should implement:
    - load_data(): Load and preprocess dataset
    - create_sgl_function(): Create the SGL function for inference

    Optional overrides:
    - extract_answer(): Extract answer from model output (if needed)
    - compute_accuracy(): Compute accuracy metric (if applicable)
    - get_answer_keys(): Get list of answer keys for multi-turn conversations

    Args:
        num_samples: The number of samples to run the benchmark on. If not provided, all questions will be used.
        subset: The subset of the dataset to run the benchmark on. If not provided, all subsets will be used.
    """

    def __init__(
        self, num_samples: Optional[int] = None, subset: Optional[List[str]] = None
    ):
        self.num_samples = num_samples
        self.subset = subset

    @abstractmethod
    def load_data(self) -> Tuple[List[Dict[str, Any]], List[Any]]:
        """
        Load and preprocess the dataset.

        Returns:
            Tuple of (questions, labels) where:
            - questions: List of question dicts for SGL function
            - labels: List of ground truth labels (can be None if not applicable)
        """
        raise NotImplementedError

    @abstractmethod
    def create_sgl_function(self) -> Callable:
        """
        Create the SGL function for inference.

        Returns:
            SGL function decorated with @sgl.function
        """
        raise NotImplementedError

    def extract_answer(self, output: str, label: Optional[Any] = None) -> Optional[Any]:
        """
        Extract answer from model output.

        Args:
            output: Raw model output string
            label: Optional ground truth label for reference

        Returns:
            Extracted answer, or None if extraction fails
        """
        return output

    def compute_accuracy(
        self, predictions: List[Any], labels: List[Any]
    ) -> Optional[float]:
        """
        Compute accuracy metric.

        Args:
            predictions: List of predicted answers
            labels: List of ground truth labels

        Returns:
            Accuracy score (0-1), or None if not applicable
        """
        return None

    def get_answer_keys(self) -> Optional[List[str]]:
        """
        Get list of answer keys for multi-turn conversations.

        Returns:
            List of answer keys (e.g., ["answer_1", "answer_2"]), or None for single-turn
        """
        return None

    def get_max_new_tokens(self) -> int:
        """
        Get maximum number of new tokens to generate.

        A `--max-tokens` override from bench_eagle3.py is stored as
        `self._max_new_tokens` and wins over the per-benchmark default.

        Returns:
            Maximum tokens (default: 2048)
        """
        if getattr(self, "_max_new_tokens", None) is not None:
            return self._max_new_tokens
        return 2048

    def run(
        self,
        host: str,
        port: int,
        batch_size: int,
        max_new_tokens: int = None,
        num_runs: int = 1,
    ):
        """
        Run the benchmark evaluation.

        This method handles the common workflow:
        1. Initialize backend
        2. Load data
        3. Create SGL function
        4. Run inference loops
        5. Compute metrics
        6. Print results

        Args:
            host (str): The host of the SGLang server
            port (int): The port of the SGLang server
            batch_size (int): The number of prompts to process in parallel
            max_new_tokens (int): Maximum number of new tokens to generate, default is 2048
            num_runs (int): The number of times to run this benchmark, default is 1. You can set it to a larger number if you want to get more stable results.

        The server applies the target's own chat template.
        """
        from sglang import set_default_backend
        from sglang.test.test_utils import select_sglang_backend

        if not host.startswith(("http://", "https://")):
            host = f"http://{host}"
        # Initialize backend
        sglang_args = Namespace(host=host, port=port, backend="srt-no-parallel")
        set_default_backend(select_sglang_backend(sglang_args))

        # Load data
        questions, labels = self.load_data()
        if len(questions) == 0:
            print("No valid questions found. Please check the dataset format.")
            return

        # Resolve max_new_tokens before create_sgl_function, which bakes it in.
        self._max_new_tokens = max_new_tokens or self.get_max_new_tokens()
        max_new_tokens = self._max_new_tokens

        # Create SGL function
        sgl_function = self.create_sgl_function()

        # Run evaluation loops
        metrics_list = []
        answer_keys = self.get_answer_keys()

        for _ in range(num_runs):
            tic = time.perf_counter()
            states = sgl_function.run_batch(
                questions,
                temperature=0,
                max_new_tokens=max_new_tokens,
                num_threads=batch_size,
                progress_bar=True,
            )
            latency = time.perf_counter() - tic

            # Extract predictions
            predictions = []
            primary_answer_key = answer_keys[0] if answer_keys else "answer"
            for i in range(len(states)):
                # Access answer from state object (states[i] supports dict-like access)
                output = states[i][primary_answer_key]
                if isinstance(output, str):
                    extracted = self.extract_answer(
                        output,
                        (labels[i] if labels and i < len(labels) else None),
                    )
                else:
                    extracted = output
                predictions.append(extracted)

            # Compute accuracy if applicable
            accuracy = None
            # Check if we have a labels list (even if all labels are None)
            has_labels_list = labels and len(labels) > 0

            if has_labels_list:
                # Call compute_accuracy whenever labels exist; it may
                # legitimately return None for a benchmark with no scorer.
                accuracy = self.compute_accuracy(predictions, labels)
                if accuracy is not None:
                    valid_count = sum(1 for p in predictions if p is not None)
                    if valid_count < len(predictions):
                        print(
                            f"Warning: {len(predictions) - valid_count} predictions could not be extracted."
                        )

            # Compute performance metrics
            metrics = compute_metrics(
                states,
                latency,
                answer_key=primary_answer_key,
                additional_answer_keys=(
                    answer_keys[1:] if answer_keys and len(answer_keys) > 1 else None
                ),
            )
            # Record accuracy whenever labels exist, even when it is None.
            if has_labels_list:
                metrics.accuracy = (
                    accuracy  # Can be None if compute_accuracy returns None
                )
                if accuracy is not None:
                    metrics.num_valid_predictions = sum(
                        1 for p in predictions if p is not None
                    )

            # Report as we go instead of leaving the user with nothing
            # until the CSV lands.
            print(
                f"  => {sum(metrics.per_request_output_length)} tokens, "
                f"throughput={metrics.output_throughput:.1f} tok/s, "
                f"accept_length={metrics.accept_length:.3f}, accuracy={accuracy}"
            )
            metrics_list.append(metrics)
        return metrics_list
