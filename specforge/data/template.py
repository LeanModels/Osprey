# Adapted from: https://github.com/sgl-project/sglang/blob/main/python/sglang/lang/chat_template.py#L13
from typing import List, Optional

from pydantic import BaseModel


class ChatTemplate(BaseModel):
    """
    This is a dataclass for the chat template.

    Args:
        assistant_header(str): The header for the assistant.
        user_header(str): The header for the user.
        system_prompt(str): The system prompt.
        end_of_turn_token(str): The end token of a turn of conversation.
        ignore_token(List[str]): The list of tokens to ignore when parsing the model output, e.g., for thinking token.
    """

    assistant_header: Optional[str] = None
    user_header: Optional[str] = None
    system_prompt: Optional[str] = None
    end_of_turn_token: Optional[str] = None
    parser_type: str = "general"
    assistant_pattern_type: str = "general"
    enable_thinking: bool = False
    ignore_token: Optional[List[str]] = None


class TemplateRegistry:
    """
    This is a registry for the chat template. Sgl-spec will register some common chat templates here.
    If you have a custom chat template, you can register it via the example below.

    Example:
    ```python
        from specforge.data.template import TEMPLATE_REGISTRY, ChatTemplate
        TEMPLATE_REGISTRY.register(
            name="custom",
            template=ChatTemplate(
                assistant_header="<|start_header_id|>assistant<|end_header_id|>\n\n",
                user_header="<|start_header_id|>user<|end_header_id|>",
                system_prompt="You are a helpful assistant.",
                end_of_turn_token="<|eot_id|>"
            )
        )
    ```
    """

    def __init__(self):
        self.templates = {}

    def register(self, name: str, template: ChatTemplate, override: bool = False):
        """
        Register a chat template for a model type.

        Args:
            name(str): The name of the chat template.
            template(ChatTemplate): The chat template.
            override(bool): Whether to override the existing template, default to False
        """
        assert (
            not override and name not in self.templates
        ), f"Chat template for the model type {name} has already been registered"
        self.templates[name] = template

    def get(self, name: str) -> ChatTemplate:
        """
        Get the chat template for a model type.

        Args:
            name(str): The name of the chat template.

        Returns:
            ChatTemplate: The chat template.
        """
        return self.templates[name]

    def get_all_template_names(self) -> List[str]:
        """
        Get all the template names.

        Returns:
            List[str]: The list of template names.
        """
        return list(self.templates.keys())


# global registry
TEMPLATE_REGISTRY = TemplateRegistry()

# Register the common template here
TEMPLATE_REGISTRY.register(
    name="llama3",
    template=ChatTemplate(
        assistant_header="<|start_header_id|>assistant<|end_header_id|>\n\n",
        user_header="<|start_header_id|>user<|end_header_id|>",
        system_prompt="You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.",
        end_of_turn_token="<|eot_id|>",
    ),
)

TEMPLATE_REGISTRY.register(
    name="qwen3-thinking",
    template=ChatTemplate(
        assistant_header="<|im_start|>assistant\n",
        user_header="<|im_start|>user\n",
        system_prompt="You are a helpful assistant.",
        end_of_turn_token="<|im_end|>\n",
        parser_type="thinking",
        enable_thinking=True,
    ),
)

# MiniMax-M2.5 uses its own role markers rather than ChatML: turns open with
# `]~b]<role>\n` and close with `[e~[`, and the sequence opens with `]~!b[`.
# Verified against the target's tokenizer.apply_chat_template.
TEMPLATE_REGISTRY.register(
    name="minimax-m25",
    template=ChatTemplate(
        assistant_header="]~b]ai\n",
        user_header="]~b]user\n",
        system_prompt="You are a helpful assistant. Your name is MiniMax-M2.5 and is built by MiniMax.",
        end_of_turn_token="[e~[",
        parser_type="thinking",
        enable_thinking=True,
    ),
)
