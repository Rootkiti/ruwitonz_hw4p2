# HW4: Adapt your HW3P2 attention for GQA and KV caching.
import torch
from torch import nn
from typing import Tuple, Optional
from .positional_encoding import RotaryPositionalEncoding

'''
TODO: Implement this Module.

Specification:
- Module should implement causal self-attention for the decoder
- A single num_kv_heads setting should cover all three head layouts:
  num_kv_heads == num_heads gives MHA, num_kv_heads == 1 gives MQA, and any
  divisor in between gives GQA
- Query always uses num_heads heads, key and value use num_kv_heads heads
- Queries and keys are rotated by RoPE before the dot product
- Keys and values are cached in their compact num_kv_heads form and repeated
  up to num_heads only while attention is computed
- A query must not attend to a later key, with or without a cache
'''

KVCache = Tuple[torch.Tensor, torch.Tensor]


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, max_len: int, num_kv_heads: Optional[int] = None, dropout: float = 0.0, rope_base: float = 10_000.0) -> None:
        '''
        Args:
            d_model      (int): The dimension of the model.
            num_heads    (int): The number of query heads.
            max_len      (int): The maximum supported context length.
            num_kv_heads (Optional[int]): The number of key/value heads. None gives MHA, 1 gives MQA.
            dropout    (float): The dropout rate.
            rope_base  (float): The frequency base used by rotary positions.
        '''
        super().__init__()

        # DO NOT MODIFY
        if d_model <= 0:
            raise ValueError("d_model must be positive")
        if num_heads <= 0:
            raise ValueError("num_heads must be positive")
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        if max_len <= 0:
            raise ValueError("max_len must be positive")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in the range [0, 1)")

        if num_kv_heads is None:
            num_kv_heads = num_heads
        if num_kv_heads <= 0:
            raise ValueError("num_kv_heads must be positive")
        if num_heads % num_kv_heads != 0:
            raise ValueError("num_heads must be divisible by num_kv_heads")

        head_dim = d_model // num_heads
        if head_dim % 2 != 0:
            raise ValueError("The attention head dimension must be even for RoPE")

        # DO NOT MODIFY: dimension and head-layout attributes.
        self.d_model       = d_model
        self.num_heads     = num_heads
        self.num_kv_heads  = num_kv_heads
        self.head_dim      = head_dim
        self.max_len       = max_len
        self.num_kv_groups = num_heads // num_kv_heads
        # TODO: Use your HW3P2 attention scaling.
        self.scale         = 1.0 / (head_dim ** 0.5)

        # TODO: Adapt your HW3P2 projections to num_kv_heads.

        # Initialize your projections, all nn.Linear with bias=False
        # Project queries to d_model; keys and values use num_kv_heads.
        self.q_proj   = nn.Linear(d_model, d_model, bias=False)
        self.k_proj   = nn.Linear(d_model, num_kv_heads * head_dim, bias=False)
        self.v_proj   = nn.Linear(d_model, num_kv_heads * head_dim, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

        # Rotary position module (DO NOT MODIFY).
        self.rope = RotaryPositionalEncoding(
            head_dim=head_dim,
            max_len=max_len,
            base=rope_base,
        )
        # TODO: Use your HW3P2 attention dropout initialization.
        self.attention_dropout =  nn.Dropout(dropout)

    def _split_heads(self, x: torch.Tensor, num_heads: int) -> torch.Tensor:
        '''
        Move attention heads before the sequence dimension.
        Args:
            x (torch.Tensor): Projected states, shape (batch_size, seq_len, num_heads * head_dim)
            num_heads (int): The number of heads contained in x.

        Returns:
            torch.Tensor: The split heads, shape (batch_size, num_heads, seq_len, head_dim)
        '''
        # TODO: Adapt your HW3P2 head splitting to the supplied head count.
        batch_size, sequence_length, _ = x.shape
        return x.view(batch_size, sequence_length, num_heads, self.head_dim).transpose(1, 2)
    
    def _repeat_kv(self, x: torch.Tensor) -> torch.Tensor:
        '''
        Repeat each key/value head for the query heads that share it.
        Args:
            x (torch.Tensor): Compact heads, shape (batch_size, num_kv_heads, seq_len, head_dim)

        Returns:
            torch.Tensor: The repeated heads, shape (batch_size, num_heads, seq_len, head_dim)
        '''
        # TODO (HW4): Implement key/value head repetition.

        # Repeat each head num_kv_groups times, aligned with its query group.
        x = torch.repeat_interleave(x, self.num_kv_groups, dim=1)

        return x
    
    def _validate_cache(self, past_kv: KVCache, batch_size: int, dtype: torch.dtype, device: torch.device) -> int:
        '''
        Validate a compact cache and return the number of cached positions.
        Args:
            past_kv (KVCache): The cached keys and values.
            batch_size (int): The expected batch size.
            dtype (torch.dtype): The expected projection dtype.
            device (torch.device): The expected projection device.

        Returns:
            int: The number of cached positions.
        '''
        # DO NOT MODIFY
        if not isinstance(past_kv, (tuple, list)) or len(past_kv) != 2:
            raise ValueError("past_kv must be a (key, value) pair")

        past_key, past_value = past_kv
        if not torch.is_tensor(past_key) or not torch.is_tensor(past_value):
            raise ValueError("past_kv must contain tensors")
        if past_key.ndim != 4 or past_value.ndim != 4:
            raise ValueError("Cached keys and values must be rank-4 tensors")
        if past_key.shape != past_value.shape:
            raise ValueError("Cached keys and values must have the same shape")

        expected_prefix = (batch_size, self.num_kv_heads)
        if past_key.shape[:2] != expected_prefix:
            raise ValueError(
                "Cache batch size or number of KV heads does not match the input"
            )
        if past_key.size(-1) != self.head_dim:
            raise ValueError("Cached head dimension does not match this layer")
        if past_key.dtype != dtype or past_value.dtype != dtype:
            raise ValueError("Cache dtype must match the projected keys and values")
        if past_key.device != device or past_value.device != device:
            raise ValueError("Cache device must match the input device")

        past_length = past_key.size(-2)
        return past_length

    def _apply_masks(self, scores: torch.Tensor, past_length: int, key_padding_mask: Optional[torch.Tensor], attn_mask: Optional[torch.Tensor]) -> torch.Tensor:
        '''
        Apply the causal, padding, and attention masks to the scores.
        Args:
            scores (torch.Tensor): Attention scores, shape (batch_size, num_heads, target_len, source_len)
            past_length (int): The number of cached source positions.
            key_padding_mask (Optional[torch.Tensor]): Padding mask, shape (batch_size, source_len)
            attn_mask (Optional[torch.Tensor]): Attention mask, shape (target_len, source_len)

        Returns:
            torch.Tensor: The masked scores.
        '''
        batch_size, _, target_length, source_length = scores.shape # DO NOT MODIFY
        minimum = float("-inf") # DO NOT MODIFY

        # TODO (HW4): Extend your HW3P2 causal mask with past_length.

        # Use absolute query positions starting at past_length and key positions
        # starting at zero. Cached masks can be rectangular (target_len, source_len).
        query_positions = torch.arange(past_length, past_length + target_length, device=scores.device)[:, None]
        key_positions   = torch.arange(source_length, device=scores.device)[None, :]
        causal_mask     = key_positions > query_positions
        masked_scores   = scores.masked_fill(causal_mask[None, None, :, :], minimum)

        # Keep validation and complete the mask calculations.
        if key_padding_mask is not None:
            expected_shape = (batch_size, source_length)
            if key_padding_mask.shape != expected_shape:
                raise ValueError(
                    f"key_padding_mask must have shape {expected_shape}"
                )
            if key_padding_mask.device != scores.device:
                raise ValueError("key_padding_mask must be on the input device")
            if key_padding_mask.dtype != torch.bool:
                raise ValueError("key_padding_mask must be Boolean")
            # TODO: Use your HW3P2 padding-mask broadcasting and application.
            expanded_padding_mask = key_padding_mask[:, None, None, :]
            masked_scores = masked_scores.masked_fill(expanded_padding_mask, minimum)

        if attn_mask is not None:
            expected_shape = (target_length, source_length)
            if attn_mask.shape != expected_shape:
                raise ValueError(f"attn_mask must have shape {expected_shape}")
            if attn_mask.device != scores.device:
                raise ValueError("attn_mask must be on the input device")
            if attn_mask.dtype == torch.bool:
                # TODO: Use your HW3P2 attention-mask broadcasting and application.
                expanded_attention_mask = attn_mask[None, None, :, :]
                masked_scores = masked_scores.masked_fill(expanded_attention_mask, minimum)
            elif torch.is_floating_point(attn_mask):
                additive_mask = attn_mask[None, None, :, :].to(dtype=scores.dtype)
                masked_scores = masked_scores + additive_mask
            else:
                raise ValueError("attn_mask must be Boolean or floating point")

        return masked_scores

    def forward(self, x: torch.Tensor, past_kv: Optional[KVCache] = None, use_cache: bool = False, key_padding_mask: Optional[torch.Tensor] = None, attn_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, Optional[KVCache]]:
        '''
        Project query, key, and value from the same input x.
        Args:
            x (torch.Tensor): Input tensor, shape (batch_size, seq_len, d_model)
            past_kv (Optional[KVCache]): Compact rotated keys and values from earlier positions.
            use_cache (bool): Whether to return the combined keys and values.
            key_padding_mask (Optional[torch.Tensor]): Padding mask, shape (batch_size, source_len)
            attn_mask (Optional[torch.Tensor]): Attention mask, shape (seq_len, source_len)

        Returns:
            output (torch.Tensor): Output tensor, shape (batch_size, seq_len, d_model)
            mha_attn_weights (torch.Tensor): Attention weights averaged over the query heads, shape (batch_size, seq_len, source_len)
            present_kv (Optional[KVCache]): The extended compact key/value cache.
        '''
        # DO NOT MODIFY
        if not torch.is_tensor(x) or x.ndim != 3:
            raise ValueError("x must have shape (batch, sequence, d_model)")
        if x.size(-1) != self.d_model:
            raise ValueError(
                f"Expected d_model {self.d_model}, but received {x.size(-1)}"
            )
        if x.size(0) == 0 or x.size(1) == 0:
            raise ValueError("Batch size and sequence length must be positive")
        if past_kv is not None and not use_cache:
            raise ValueError("use_cache must be True when past_kv is provided")

        batch_size, sequence_length, _ = x.shape

        # TODO: Adapt your HW3P2 forward pass with the HW4 steps below.

        # Project inputs
        projected_query = self.q_proj(x)
        projected_key   = self.k_proj(x)
        projected_value = self.v_proj(x)

        # Reshape for multiple heads
        # Query and key/value do not use the same head count here
        query = self._split_heads(projected_query, self.num_heads)
        key   = self._split_heads(projected_key, self.num_kv_heads)
        value = self._split_heads(projected_value, self.num_kv_heads)

        # Read the cache length and check the context budget (DO NOT MODIFY)
        past_length = 0
        if past_kv is not None:
            past_length = self._validate_cache(
                past_kv,
                batch_size=batch_size,
                dtype=key.dtype,
                device=key.device,
            )
        total_length = past_length + sequence_length
        if total_length > self.max_len:
            raise ValueError(
                f"Sequence length {total_length} exceeds max_len {self.max_len}"
            )

        # TODO (HW4): Apply your adapted RoPE to Q/K at the cached position offset.
        rotated_query = self.rope(query, start_pos=past_length)
        rotated_key   = self.rope(key, start_pos=past_length)

        # HW4: Append the new keys and values to the cache.
        # Only the key is rotated, the value is stored as projected
        if past_kv is not None:
            past_key, past_value = past_kv
            combined_key   = torch.cat([past_key, rotated_key], dim=2)
            combined_value = torch.cat([past_value, value], dim=2)
        else:
            combined_key   = rotated_key
            combined_value = value

        # DO NOT MODIFY
        # Cache compact heads before repetition; do not reorder these blocks.
        if use_cache:
            # Detach cached inference state; keep current attention tensors attached.
            present_kv = (
                combined_key.detach(),
                combined_value.detach(),
            )
        else:
            present_kv = None

        # HW4: Repeat key/value heads for their query groups.
        repeated_key   = self._repeat_kv(combined_key)
        repeated_value = self._repeat_kv(combined_value)

        # Use your HW3P2 attention-score calculation and scaling.
        scores        = torch.matmul(rotated_query, repeated_key.transpose(-2, -1))
        scaled_scores = scores * self.scale
        masked_scores = self._apply_masks(scaled_scores, past_length, key_padding_mask, attn_mask)

        # TODO: Adapt your HW3P2 softmax to use float32 for half-precision
        # inputs, then cast probabilities back to the score dtype.
        if masked_scores.dtype in (torch.float16, torch.bfloat16):
            float_probabilities = torch.softmax(masked_scores.float(), dim=-1)
            attention_probabilities = float_probabilities.to(masked_scores.dtype)
        else:
            attention_probabilities = torch.softmax(masked_scores, dim=-1)
        # DO NOT MODIFY: zero probabilities for fully masked rows.
        attention_probabilities = torch.nan_to_num(
            attention_probabilities,
            nan=0.0,
        )

        # Use your HW3P2 head-averaged attention weights.
        # These are the weights you return, and they are taken before dropout
        # while the context below is computed from the dropped weights
        average_attention_probabilities = attention_probabilities.mean(dim=1)

        # TODO: Use your HW3P2 attention dropout application.
        dropped_probabilities = self.attention_dropout(attention_probabilities)

        # Use your HW3P2 attention-weighted values.
        context = torch.matmul(dropped_probabilities, repeated_value)

        # Use your HW3P2 head merging.
        merged_context = context.transpose(1, 2).contiguous().view(batch_size, sequence_length, self.d_model)

        # Use your HW3P2 output projection.
        output = self.out_proj(merged_context)
 
        return output, average_attention_probabilities, present_kv
