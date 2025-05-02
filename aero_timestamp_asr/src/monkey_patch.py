from typing import Optional, Tuple

import torch
from transformers.models.qwen2 import apply_rotary_pos_emb

try:
    from flash_attn import flash_attn_func
except ImportError:
    print("flash-attn is not installed. Please install it to use the flash attention implementation.")


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

    attn_output, softmax_lse, S_dmask = flash_attn_func(
        query_states,
        key_states,
        value_states,
        attention_mask,
        dropout=0.01,  # Force dropout to 0.01 to avoid return Nan for flash-attn
        causal=True,
        softmax_scale=None,
        return_attn_probs=True,
    )

    attn_output = attn_output.reshape(*input_shape, -1).contiguous()
    attn_output = self.o_proj(attn_output)
    return attn_output, S_dmask
