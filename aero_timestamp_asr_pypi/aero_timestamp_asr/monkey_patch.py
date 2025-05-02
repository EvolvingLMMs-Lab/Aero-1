from typing import Callable, Optional, Tuple

import torch
from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb

try:
    from flash_attn import flash_attn_func
except ImportError:
    print("flash-attn is not installed")


def _bind_method_to_module(module, method_name: str, new_method: Callable):
    # Binds a new method to a module instance so that self is passed as the first argument
    module.__dict__[method_name] = new_method.__get__(module, module.__class__)


def apply_flash_attn_with_attn_scores_on_aero(model):
    """
    Binds the flash attention implementation with attention scores to the Aero model.
    """
    for layer in model.language_model.model.layers:
        # Bind the flash attention implementation to the layer
        _bind_method_to_module(layer.self_attn, "forward", qwen2_flash_attn_with_attn_scores)


APPLY_MAP = {
    "aero": apply_flash_attn_with_attn_scores_on_aero,
}


def apply_flash_attn_with_attn_scores(model, model_type: str):
    """
    Applies the flash attention implementation with attention scores to the model.
    """
    if model_type in APPLY_MAP:
        APPLY_MAP[model_type](model)
    else:
        raise ValueError(f"Model type {model_type} is not supported for flash attention.")


def qwen2_flash_attn_with_attn_scores(
    self,
    hidden_states: torch.Tensor,
    position_embeddings: Tuple[torch.Tensor, torch.Tensor],
    attention_mask: Optional[torch.Tensor],
    past_key_value=None,
    cache_position: Optional[torch.LongTensor] = None,
    **kwargs,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, self.head_dim)

    query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

    cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    if past_key_value is not None:
        # sin and cos are specific to RoPE models; cache_position needed for the static cache
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

    dropout = 0.01
    attn_output, softmax_lse, S_dmask = flash_attn_func(
        query_states.transpose(1, 2),
        key_states.transpose(1, 2),
        value_states.transpose(1, 2),
        dropout,  # Force dropout to 0.01 to avoid return Nan for flash-attn
        causal=True,
        softmax_scale=None,
        return_attn_probs=True,
    )

    attn_output = attn_output.reshape(*input_shape, -1).contiguous()
    attn_output = self.o_proj(attn_output)
    return attn_output, S_dmask
