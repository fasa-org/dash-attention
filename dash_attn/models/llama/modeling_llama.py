from typing import Optional

import torch
from torch import nn

from transformers.cache_utils import Cache, DynamicCache
from transformers.masking_utils import create_causal_mask
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.models.llama.modeling_llama import (
    LlamaAttention as TransformersLlamaAttention,
    LlamaDecoderLayer as TransformersLlamaDecoderLayer,
    LlamaForCausalLM as TransformersLlamaForCausalLM,
    LlamaForQuestionAnswering,
    LlamaForSequenceClassification,
    LlamaForTokenClassification,
    LlamaMLP,
    LlamaPreTrainedModel,
    LlamaRMSNorm,
    LlamaRotaryEmbedding,
    apply_rotary_pos_emb,
    eager_attention_forward,
)
from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs

from dash_attn import dash_attn as DashAttention
from .configuration_llama import LlamaConfig


if "dash_attn" not in ALL_ATTENTION_FUNCTIONS.valid_keys():
    ALL_ATTENTION_FUNCTIONS["dash_attn"] = eager_attention_forward


class LlamaAttention(TransformersLlamaAttention):
    def __init__(self, config: LlamaConfig, layer_idx: int):
        super().__init__(config=config, layer_idx=layer_idx)
        self.indexer_q = nn.Parameter(
            torch.zeros(config.num_key_value_heads * self.head_dim, dtype=torch.bfloat16)
        )
        self.last_active_blocks: Optional[torch.Tensor] = None
        self.dash_attn = DashAttention(
            chunk_size=config.chunk_size,
            enable_gqa=config.num_attention_heads != config.num_key_value_heads,
            estimate_diagonal=config.estimate_diagonal,
            sigma=config.sigma,
            scaling_factor=config.scaling_factor,
        )
        self._active_blocks_history: list[torch.Tensor] = []

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_values: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        return_active_blocks = bool(kwargs.pop("return_active_blocks", False))
        if return_active_blocks and not past_key_values:
            self._active_blocks_history = []
        self.last_active_blocks = None
        if self.config._attn_implementation != "dash_attn":
            return super().forward(
                hidden_states=hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                cache_position=cache_position,
                **kwargs,
            )

        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_values is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)

        old_return_active_blocks = self.dash_attn.return_active_blocks
        self.dash_attn.return_active_blocks = return_active_blocks
        try:
            dash_result = self.dash_attn(
                query_states.contiguous(),
                key_states.contiguous(),
                value_states.contiguous(),
                self.indexer_q.view(self.config.num_key_value_heads, self.head_dim),
            )
        finally:
            self.dash_attn.return_active_blocks = old_return_active_blocks

        if return_active_blocks:
            attn_output, self.last_active_blocks = dash_result
            self._active_blocks_history.append(self.last_active_blocks.detach())
        else:
            attn_output = dash_result
        attn_output = attn_output.to(dtype=hidden_states.dtype)
        attn_output = attn_output.transpose(1, 2).reshape(*input_shape, -1).contiguous()
        return self.o_proj(attn_output), None


class LlamaDecoderLayer(TransformersLlamaDecoderLayer):
    def __init__(self, config: LlamaConfig, layer_idx: int):
        super().__init__(config=config, layer_idx=layer_idx)
        self.self_attn = LlamaAttention(config=config, layer_idx=layer_idx)


class LlamaModel(LlamaPreTrainedModel):
    config_class = LlamaConfig

    def __init__(self, config: LlamaConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [LlamaDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = LlamaRotaryEmbedding(config=config)
        self.gradient_checkpointing = False
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutputWithPast:
        return_active_blocks = bool(kwargs.pop("return_active_blocks", False))
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = create_causal_mask(
            config=self.config,
            input_embeds=inputs_embeds,
            attention_mask=attention_mask,
            cache_position=cache_position,
            past_key_values=past_key_values,
            position_ids=position_ids,
        )

        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        for decoder_layer in self.layers[: self.config.num_hidden_layers]:
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                dash_attn_mask=attention_mask,
                return_active_blocks=return_active_blocks,
                **kwargs,
            )

        hidden_states = self.norm(hidden_states)
        return BaseModelOutputWithPast(last_hidden_state=hidden_states, past_key_values=past_key_values)

    def get_active_blocks(self) -> list[torch.Tensor] | None:
        active_blocks = []
        for layer in self.layers[: self.config.num_hidden_layers]:
            history = layer.self_attn._active_blocks_history
            if not history:
                return None
            active_blocks.append(torch.cat(history, dim=-1))
        return active_blocks


class LlamaForCausalLM(TransformersLlamaForCausalLM):
    config_class = LlamaConfig

    def __init__(self, config: LlamaConfig):
        LlamaPreTrainedModel.__init__(self, config)
        self.model = LlamaModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    def forward(self, *args, return_active_blocks: bool = False, **kwargs):
        return super().forward(*args, return_active_blocks=return_active_blocks, **kwargs)

    def get_active_blocks(self) -> list[torch.Tensor] | None:
        return self.model.get_active_blocks()


__all__ = [
    "LlamaConfig",
    "LlamaAttention",
    "LlamaDecoderLayer",
    "LlamaForCausalLM",
    "LlamaForQuestionAnswering",
    "LlamaForSequenceClassification",
    "LlamaForTokenClassification",
    "LlamaModel",
    "LlamaPreTrainedModel",
]
