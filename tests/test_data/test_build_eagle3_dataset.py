import os
import tempfile
import unittest

import torch
from datasets import Dataset
from transformers import AutoTokenizer

from specforge.data.preprocessing import build_eagle3_dataset
from specforge.utils import safe_conversations_generator

# ANSI color codes
RED = "\033[91m"
RESET = "\033[0m"


def print_with_loss_mask(tokenizer, input_ids, loss_mask, title=""):
    """Print text with loss_mask=1 (assistant) parts in RED."""
    input_ids = input_ids.flatten()
    loss_mask = loss_mask.flatten()

    print(f"\n{'=' * 60}")
    print(f"{title}")
    print("=" * 60)

    # Group consecutive tokens by loss_mask value
    current_mask = loss_mask[0].item()
    current_ids = [input_ids[0].item()]

    for i in range(1, len(input_ids)):
        if loss_mask[i].item() == current_mask:
            current_ids.append(input_ids[i].item())
        else:
            # Decode and print current group
            text = tokenizer.decode(current_ids, skip_special_tokens=False)
            if current_mask == 1:
                print(f"{RED}{text}{RESET}", end="")
            else:
                print(text, end="")
            current_ids = [input_ids[i].item()]
            current_mask = loss_mask[i].item()

    # Print remaining tokens
    if current_ids:
        text = tokenizer.decode(current_ids, skip_special_tokens=False)
        if current_mask == 1:
            print(f"{RED}{text}{RESET}")
        else:
            print(text)

    print("=" * 60)


class TestBuildEagle3Dataset(unittest.TestCase):
    """Test build_eagle3_dataset end to end on one conversation with the paper's Qwen3-8B template."""

    @classmethod
    def setUpClass(cls):
        cls.model_name = "Qwen/Qwen3-8B"
        cls.template_key = "qwen3-thinking"
        cls.tokenizer = AutoTokenizer.from_pretrained(
            cls.model_name, trust_remote_code=True
        )
        cls.max_length = 65535

    def test_build_eagle3_dataset_basic(self):
        """One two-turn conversation goes through tokenization, template parsing, and loss masking."""
        # Create a HF Dataset with 1 sample
        data_file = os.path.join(
            os.path.dirname(__file__), "data", "basic_conversation.jsonl"
        )
        with tempfile.TemporaryDirectory() as tmp_dir:
            dataset = Dataset.from_generator(
                generator=safe_conversations_generator,
                gen_kwargs={"file_path": data_file},
                cache_dir=tmp_dir,
                keep_in_memory=True,
            )
            result_dataset = build_eagle3_dataset(
                dataset=dataset,
                tokenizer=self.tokenizer,
                chat_template=self.template_key,
                max_length=self.max_length,
                shuffle_seed=42,
                num_proc=1,
                cache_dir=None,
                cache_key=None,
            )

            # Verify the dataset has the expected columns
            self.assertIn("input_ids", result_dataset.column_names)
            self.assertIn("loss_mask", result_dataset.column_names)
            self.assertIn("attention_mask", result_dataset.column_names)
            self.assertEqual(len(result_dataset), 1)

            # Decode input_ids to text
            input_ids = result_dataset[0]["input_ids"].squeeze()
            loss_mask = result_dataset[0]["loss_mask"].squeeze()

            # Print full text with loss_mask=1 in RED
            print_with_loss_mask(
                self.tokenizer,
                input_ids,
                loss_mask,
                title="[build_eagle3_dataset] Full text (RED = loss_mask=1):",
            )

            # Verify assistant tokens exist
            assistant_indices = torch.where(loss_mask == 1)[0]
            self.assertTrue(len(assistant_indices) > 0, "No assistant tokens found")


if __name__ == "__main__":
    unittest.main(verbosity=2)
