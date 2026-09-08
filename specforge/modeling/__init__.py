from .auto import AutoDraftModelConfig, AutoEagle3DraftModel
from .draft.llama3_eagle import LlamaForCausalLMEagle3
from .draft.qwen3_eagle import Qwen3ForCausalLMEagle3
from .target.eagle3_target_model import (
    Eagle3TargetModel,
    HFEagle3TargetModel,
    SGLangEagle3TargetModel,
    get_eagle3_target_model,
)

__all__ = [
    "AutoDraftModelConfig",
    "AutoEagle3DraftModel",
    "Eagle3TargetModel",
    "LlamaForCausalLMEagle3",
    "Qwen3ForCausalLMEagle3",
    "SGLangEagle3TargetModel",
    "HFEagle3TargetModel",
    "get_eagle3_target_model",
]
