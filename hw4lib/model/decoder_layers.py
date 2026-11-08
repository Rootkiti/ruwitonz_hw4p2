import torch.nn as nn
import torch
from typing import Tuple, Optional
from .sublayers import SelfAttentionLayer, FeedForwardLayer
from .causal_self_attention import KVCache

'''
TODO: Implement these Modules.

This file contains the self-attention decoder layer:

1. SelfAttentionDecoderLayer: Used in decoder-only transformers (like GPT)
   - Contains masked self-attention and feed-forward sublayers
   - Used for tasks like language modeling where only previous tokens can be attended to
   
Each layer follows a Pre-LN (Layer Normalization) architecture where:
- Layer normalization is applied before each sublayer operation
- Residual connections wrap around each sublayer

Implementation Steps for Each Layer:
1. Initialize the required sublayers in __init__:
   - SelfAttentionLayer for masked self-attention
   - FeedForwardLayer for position-wise processing

2. Implement the forward pass to:
   - Apply sublayers in the correct order
   - Pass appropriate masks to attention layers
   - Return both outputs and attention weights
'''

## -------------------------------------------------------------------------------------------------  
## Decoder Layers
## -------------------------------------------------------------------------------------------------      
class SelfAttentionDecoderLayer(nn.Module):
    '''
    Pre-LN Decoder Layer with masked self-attention and feed-forward sublayers.
    Used in the decoder-only Transformer architecture.  
    '''
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        dropout: float = 0.1,
        *,
        use_rope: bool = False,
        max_len: int = 2048,
        num_kv_heads: Optional[int] = None,
        rope_base: float = 10_000.0,
    ):
        '''
        Initialize the SelfAttentionDecoderLayer. 
        Args:
            d_model   (int): The dimension of the model.
            num_heads (int): The number of attention heads.
            d_ff      (int): The dimension of the feedforward network.
            dropout (float): The dropout rate.
            use_rope (bool): Whether to use rotary causal attention.
            max_len (int): Maximum context length for rotary attention.
            num_kv_heads (Optional[int]): Number of key/value heads.
            rope_base (float): Frequency base for rotary positions.
        ''' 
        super().__init__()
        # TODO: Implement __init__
       
        # TODO: Initialize the sublayers      
        self.self_attn = SelfAttentionLayer(
            d_model,
            num_heads,
            dropout,
            use_rope=use_rope,
            max_len=max_len,
            num_kv_heads=num_kv_heads,
            rope_base=rope_base,
        )
        self.ffn = FeedForwardLayer(d_model, d_ff, dropout)
        # raise NotImplementedError # Remove once implemented

    def forward(self, x: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None, attn_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        '''
        Forward pass for the DecoderLayer1.
        Args:
            x (torch.Tensor): The input tensor. shape: (batch_size, seq_len, d_model)   
            key_padding_mask (torch.Tensor): The padding mask for the decoder. shape: (batch_size, seq_len)
            attn_mask (torch.Tensor): The self-attention mask. shape: (seq_len, seq_len)

        Returns:
            x (torch.Tensor): The output tensor. shape: (batch_size, seq_len, num_classes)
            mha_attn_weights (torch.Tensor): The attention weights. shape: (batch_size, seq_len, seq_len)   
        '''
        # TODO: Implement forward: Follow the figure in the writeup
        x, mha_attn_weights = self.self_attn(x, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        x = self.ffn(x)

        # TODO: Return the output tensor and attention weights
        # raise NotImplementedError # Remove once implemented
        return x, mha_attn_weights

    def forward_cached(
        self,
        x: torch.Tensor,
        past_kv: Optional[KVCache] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, KVCache]:
        '''
        TODO: Thread one layer's compact cache through rotary decoding.

        Apply self.self_attn.forward_cached first, then the feed-forward
        sublayer. The cache must stay at num_kv_heads rather than expanding to
        num_heads.

        Args:
            x (torch.Tensor): New decoder states.
            past_kv (Optional[KVCache]): This layer's earlier compact cache.

        Returns:
            Updated states, attention weights, and the extended compact cache.
        '''
        # TODO: Run cached self-attention and collect present_kv.
        x, mha_attn_weights, present_kv =self.self_attn.forward_cached(x, past_kv=past_kv)

        # TODO: Apply the feed-forward sublayer to the new positions.
        x = self.ffn(x)

        # TODO: Return x, mha_attn_weights, and present_kv.
        return x, mha_attn_weights, present_kv
        # raise NotImplementedError
## -------------------------------------------------------------------------------------------------
