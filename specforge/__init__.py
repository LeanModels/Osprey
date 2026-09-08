from .core import OnlineEagle3Model
from .modeling import (
    AutoDraftModelConfig,
    AutoEagle3DraftModel,
    Eagle3TargetModel,
    LlamaForCausalLMEagle3,
    Qwen3ForCausalLMEagle3,
    get_eagle3_target_model,
)

__all__ = [
    "AutoDraftModelConfig",
    "AutoEagle3DraftModel",
    "Eagle3TargetModel",
    "LlamaForCausalLMEagle3",
    "OnlineEagle3Model",
    "Qwen3ForCausalLMEagle3",
    "get_eagle3_target_model",
]
