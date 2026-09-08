import json
import re
import warnings
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple

import torch
from transformers import PreTrainedTokenizer

from .template import ChatTemplate

# A conversation is a list of {"role", "content"} messages.
Conversation = List[Dict[str, str]]

__all__ = ["GeneralParser", "ThinkingParser"]


class Parser(ABC):

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        chat_template: ChatTemplate,
    ):
        self.tokenizer = tokenizer
        self.chat_template = chat_template
        self.standard_keys = {"role", "content", "tool_calls"}

    @abstractmethod
    def parse(
        self, conversation: Conversation, max_length: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Parse the conversation into a list of tensors.

        Args:
            conversation: The conversation to parse.

        Returns:
            A list of tensors: [input_ids, loss_mask]
        """

    def _sanitize_message(self, message: dict) -> dict:
        """
        Clean up individual messages, handling the following issues:
        1. `tool_calls` is a string → Parse as a list
        2. `tool_calls[].function.arguments` is a string → Parse as a dictionary
        3. Non-standard fields (extra, etc.) in `tool_calls[]` → Remove
        """
        cleaned = {k: v for k, v in message.items() if k in self.standard_keys}

        # ===== handle tool_calls =====
        if "tool_calls" in cleaned:
            tool_calls = cleaned["tool_calls"]

            # tool_calls is a string → Parsing
            if isinstance(tool_calls, str):
                try:
                    tool_calls = json.loads(tool_calls)
                except json.JSONDecodeError:
                    warnings.warn(
                        "Failed to parse tool_calls JSON string, removing tool_calls"
                    )
                    cleaned.pop("tool_calls", None)
                    return cleaned

            # Clean each tool_call
            if isinstance(tool_calls, list):
                sanitized_tool_calls = []

                for tc in tool_calls:
                    if not isinstance(tc, dict):
                        continue

                    # Only retain the standard fields: id, type, function
                    clean_tc = {
                        "id": tc.get("id", ""),
                        "type": tc.get("type", "function"),
                    }

                    # handle function
                    func = tc.get("function", {})
                    if isinstance(func, dict):
                        clean_func = {
                            "name": func.get("name", ""),
                        }

                        arguments = func.get("arguments", {})
                        if isinstance(arguments, str):
                            try:
                                arguments = json.loads(arguments)
                            except json.JSONDecodeError:
                                warnings.warn(
                                    f"Failed to parse arguments for tool '{clean_func['name']}': "
                                    f"{arguments[:100]}..."
                                )
                                arguments = {}

                        clean_func["arguments"] = arguments
                        clean_tc["function"] = clean_func

                    sanitized_tool_calls.append(clean_tc)

                cleaned["tool_calls"] = sanitized_tool_calls

        return cleaned


_harmony_encoding = None


class GeneralParser(Parser):

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        chat_template: ChatTemplate,
    ):
        super().__init__(tokenizer, chat_template)
        self.system_prompt = chat_template.system_prompt
        self.user_message_separator = f"{chat_template.end_of_turn_token}"
        self.assistant_message_separator = f"{chat_template.assistant_header}"
        self.set_assistant_pattern(chat_template)

    def apply_chat_template(self, messages, tool, **kwargs) -> str:
        conversation = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
            tools=tool,
            **kwargs,
        )
        return conversation

    def set_assistant_pattern(self, chat_template: ChatTemplate):
        self.assistant_pattern = (
            re.escape(self.assistant_message_separator)
            + r"([\s\S]*?(?:"
            + re.escape(self.chat_template.end_of_turn_token)
            + "|$))"
        )

    def parse(
        self,
        conversation: Conversation,
        max_length: int,
        preformatted: bool = False,
        train_only_last_turn: bool = False,
        tool: List[Dict] = [],
        **kwargs,
    ) -> Dict[str, List[torch.Tensor]]:
        if not preformatted:
            messages = []

            if conversation[0]["role"] == "system":
                warnings.warn(
                    "The first message is from system, we will use the system prompt "
                    "from the data and ignore the system prompt from the template"
                )
                messages.append(
                    {"role": "system", "content": conversation[0]["content"]}
                )
                conversation = conversation[1:]
            else:
                if self.system_prompt:
                    messages.append({"role": "system", "content": self.system_prompt})

            for j, sentence in enumerate(conversation):
                role = sentence["role"]
                if j == 0:
                    if role != "user":
                        warnings.warn(
                            f"Conversation must start with a 'user' role, but found '{role}'. Conversation truncated."
                        )
                        break
                else:
                    prev_role = conversation[j - 1]["role"]
                    if role == "tool" and prev_role not in ["assistant", "tool"]:
                        warnings.warn(
                            f"A 'tool' message must follow an 'assistant' or 'tool' message, but was preceded by '{prev_role}'. Conversation truncated."
                        )
                        break
                    if role == "assistant" and prev_role not in ["user", "tool"]:
                        warnings.warn(
                            f"An 'assistant' message must follow a 'user' or 'tool' message, but was preceded by '{prev_role}'. Conversation truncated."
                        )
                        break
                sentence = self._sanitize_message(sentence)
                messages.append(sentence)
            try:
                conversation = self.apply_chat_template(messages, tool=tool, **kwargs)
            except (ValueError, TypeError):
                # Fallback rendering for tokenizers without built-in chat_template
                warnings.warn(
                    "Tokenizer does not have a chat_template, using fallback rendering."
                )
                parts = []
                bos_token = getattr(self.tokenizer, "bos_token", None)
                user_header = self.chat_template.user_header or ""
                assistant_header = self.chat_template.assistant_header or ""
                end_of_turn = self.chat_template.end_of_turn_token or ""

                # Add BOS token at the start
                if bos_token:
                    parts.append(bos_token)

                for msg in messages:
                    if msg["role"] == "system":
                        parts.append(msg["content"])
                    elif msg["role"] == "user":
                        parts.append(f"{user_header}{msg['content']}")
                    elif msg["role"] == "assistant":
                        parts.append(f"{assistant_header}{msg['content']}{end_of_turn}")
                conversation = "".join(parts)

        if not self.tokenizer.pad_token_id:
            self.tokenizer.pad_token_id = self.tokenizer.unk_token_id

        # get input_ids
        encoding = self.tokenizer(
            conversation,
            max_length=max_length,
            truncation=True,
            return_tensors="pt",
            add_special_tokens=False,
        )
        input_ids = encoding.input_ids[0]
        loss_mask = torch.zeros(len(input_ids), dtype=torch.long)

        matches = list(re.finditer(self.assistant_pattern, conversation, re.DOTALL))
        if train_only_last_turn and matches:
            matches = [matches[-1]]  # Only keep the last match

        for match in matches:
            content_start_char = match.start(1)
            content_end_char = match.end(1)

            # --- Core Alternative Operation: Calculate Token Index Based on Prefix String Length ---
            # Encode the text "assistant start", the length of which is the position of the starting token.
            prefix_ids = self.tokenizer.encode(
                conversation[:content_start_char],
                add_special_tokens=False,
                truncation=True,
                max_length=max_length,
            )
            # Encodes the text "assistant end", the length of which is the position of the end token.
            full_ids = self.tokenizer.encode(
                conversation[:content_end_char],
                add_special_tokens=False,
                truncation=True,
                max_length=max_length,
            )

            start_token_idx = len(prefix_ids)
            end_token_idx = len(full_ids)

            # Handling out-of-bounds errors caused by truncation
            actual_start = min(start_token_idx, len(input_ids))
            actual_end = min(end_token_idx, len(input_ids))

            if actual_start < actual_end:
                loss_mask[actual_start:actual_end] = 1

        # Zero out loss_mask for ignore_tokens
        ignore_tokens = self.chat_template.ignore_token
        if ignore_tokens:
            for token_str in ignore_tokens:
                start = 0
                while True:
                    idx = conversation.find(token_str, start)
                    if idx == -1:
                        break
                    ignore_start_char = idx
                    ignore_end_char = idx + len(token_str)

                    prefix_ids = self.tokenizer.encode(
                        conversation[:ignore_start_char],
                        add_special_tokens=False,
                        truncation=True,
                        max_length=max_length,
                    )
                    full_ids = self.tokenizer.encode(
                        conversation[:ignore_end_char],
                        add_special_tokens=False,
                        truncation=True,
                        max_length=max_length,
                    )

                    start_token_idx = min(len(prefix_ids), len(input_ids))
                    end_token_idx = min(len(full_ids), len(input_ids))

                    if start_token_idx < end_token_idx:
                        loss_mask[start_token_idx:end_token_idx] = 0

                    start = ignore_end_char

        return input_ids, loss_mask


class ThinkingParser(GeneralParser):
    """
    Parser for thinking/reasoning models.

    This parser processes the entire conversation (not just the last turn).
    It handles reasoning_content and tool_calls in assistant messages.
    The loss mask covers from assistant_header to end_of_turn_token (inclusive).
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        chat_template: ChatTemplate,
    ):
        super().__init__(tokenizer, chat_template)
        self.standard_keys = {"role", "content", "tool_calls", "reasoning_content"}

    def apply_chat_template(self, messages, tool, **kwargs) -> str:
        """Apply chat template to all messages, handling reasoning_content and tool_calls."""
        conversation = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
            add_special_tokens=False,
            tools=tool,
            **kwargs,
        )
        return conversation

    def parse(
        self,
        conversation: Conversation,
        max_length: int,
        preformatted: bool = False,
        train_only_last_turn: bool = False,
        tool: List[Dict] = [],
        **kwargs,
    ) -> Dict[str, List[torch.Tensor]]:
        """Parse conversation, processing all assistant turns for loss mask."""
        if self.chat_template.enable_thinking:
            kwargs["enable_thinking"] = True
        return super().parse(
            conversation, max_length, preformatted, train_only_last_turn, tool, **kwargs
        )
