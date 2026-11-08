import torch
from torch import nn
import math

'''
TODO: Implement these Modules.

This file contains two ways of giving a transformer positional information:

1. PositionalEncoding: Fixed sinusoidal encoding added to the token embeddings
   - Added once, before the first layer
   - Encodes absolute position

2. RotaryPositionalEncoding: Rotates queries and keys inside each attention operation
   - Applied per head, after the q/k projections and the reshape into heads
   - Encodes relative position

Specification for RotaryPositionalEncoding:
- Applied to queries and keys only, never to values
- Pairs dimension i with dimension i + head_dim//2 (the "split-half" convention
  used by Llama and HuggingFace), not with dimension i+1
- The angle for position t and pair i is t / (base ** (2i / head_dim))
- cos and sin are registered as buffers of shape (1, 1, max_len, head_dim),
  so that they broadcast over batch and head
'''

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len):
        """
        Initialize the PositionalEncoding.
        Args:
            d_model (int): The dimension of the model.
            max_len (int): The maximum length of the input sequence.
        """     
        super().__init__()
        self.create_pe_table(d_model, max_len)

    def create_pe_table(self, d_model, max_len):
        """
        Create the positional encoding table.
        
        Args:
            d_model (int): The dimension of the model.
            max_len (int): The maximum length of the input sequence.
        """
        # TODO: Implement create_pe_table
        # raise NotImplementedError # Remove once implemented
        position = torch.arange(max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  
        # Register as buffer to save with model state
        self.register_buffer('pe', pe)
        

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the PositionalEncoding.
        Args:
            x (torch.Tensor): The input tensor of shape (B x T x d_model)
        Returns:
            torch.Tensor: Input with positional encoding added (B x T x d_model)
        Errors:
            - ValueError: If sequence length exceeds maximum length 
        """
        # TODO: Implement forward
        # Step 1: Get sequence length from input tensor
        seq_len = x.size(1)
        # Step 2: Verify sequence length doesn't exceed maximum length, raise error if it does
        if seq_len > self.pe.size(1):
            raise ValueError(f"Sequence length {seq_len} exceeds the maximum length {self.pe.size(1)}")
        # Step 3: Add positional encodings to input
        # raise NotImplementedError # Remove once implemented
        x = x + self.pe[:, :seq_len]

        return x

## -------------------------------------------------------------------------------------------------

class RotaryPositionalEncoding(nn.Module):
    """
    Rotary Positional Encoding (RoPE).
    Rotates queries and keys so that attention scores depend on relative position.
    """
    def __init__(self, head_dim, max_len, base=10000.0):
        """
        Initialize the RotaryPositionalEncoding.
        Args:
            head_dim (int): The dimension of each attention head. Must be even.
            max_len  (int): The maximum length of the input sequence.
            base   (float): The base of the geometric progression of wavelengths.
        """
        super().__init__()
        if head_dim % 2 != 0:
            raise ValueError(f"head_dim must be even for RoPE, but got {head_dim}")
        if max_len <= 0:
            raise ValueError(f"max_len must be positive, but got {max_len}")
        self.head_dim = head_dim
        self.create_rope_table(head_dim, max_len, base)
 
    def create_rope_table(self, head_dim, max_len, base):
        """
        Create the rotation tables.
        Args:
            head_dim (int): The dimension of each attention head.
            max_len  (int): The maximum length of the input sequence.
            base   (float): The base of the geometric progression of wavelengths.
        """
        # TODO: Implement create_rope_table
        # Step 1: Compute the inverse frequencies for the head_dim//2 pairs
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.float) / head_dim))
        # Step 2: Take the outer product with the positions to get the angles
        position = torch.arange(max_len, dtype=torch.float)
        angles = torch.outer(position, inv_freq) 
        # Step 3: Widen the angles to head_dim so each pair shares an angle, then take cos and sin
        # angles = torch.cat([angles, angles], dim=-1) 
        angles = torch.repeat_interleave(angles, 2, dim=-1)
        # raise NotImplementedError # Remove once implemented
        cos = torch.cos(angles)[None, None, :, :]  
        sin = torch.sin(angles)[None, None, :, :]  
        # Register as buffers to save with model state
        self.register_buffer('cos', cos)
        self.register_buffer('sin', sin)
 
    def rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        """
        Rotate the two halves of the last dimension into each other.
        Given x = [x_0 ... x_{d/2-1}, x_{d/2} ... x_{d-1}], return
        [-x_{d/2} ... -x_{d-1}, x_0 ... x_{d/2-1}].
        Args:
            x (torch.Tensor): Input tensor, shape (..., head_dim)
        Returns:
            torch.Tensor: Rotated tensor of the same shape
        """
        # TODO: Implement rotate_half
        # raise NotImplementedError # Remove once implemented
        # x1, x2 = x[..., :x.size(-1)//2], x[..., x.size(-1)//2:]
        # return torch.cat([-x2, x1], dim=-1)
        x1, x2 = x[..., 0::2], x[..., 1::2]
        return torch.stack([-x2, x1], dim=-1).flatten(-2)
 
    def forward(self, x: torch.Tensor, start_pos: int = 0) -> torch.Tensor:
        """
        Forward pass for the RotaryPositionalEncoding.
        Args:
            x (torch.Tensor): Queries or keys, shape (batch_size, num_heads, seq_len, head_dim)
            start_pos  (int): Index of the first position. Defaults to 0.
        Returns:
            torch.Tensor: Rotated tensor of the same shape
        Errors:
            - ValueError: If x does not have shape (batch_size, num_heads, seq_len, head_dim)
            - ValueError: If sequence length exceeds maximum length
        """
        # TODO: Implement forward
        # Step 1: Get sequence length from the input tensor
        if x.ndim != 4 or x.size(-1) != self.head_dim:
            raise ValueError(f"x must have shape (batch_size, num_heads, seq_len, {self.head_dim}), but got {tuple(x.shape)}")
        seq_len = x.size(2)
        # Step 2: Verify sequence length doesn't exceed maximum length, raise error if it does
        if start_pos < 0 or start_pos + seq_len > self.cos.size(2):
            raise ValueError(f"Sequence length {start_pos + seq_len} exceeds the maximum length {self.cos.size(2)}")
        # Step 3: Slice the cos and sin tables to the positions in use
        cos = self.cos[:, :, start_pos:start_pos+seq_len, :].to(x.dtype)
        sin = self.sin[:, :, start_pos:start_pos+seq_len, :].to(x.dtype)
        # Step 4: Rotate x, where rotating x is  x * cos + self.rotate_half(x) * sin
        x = x * cos + self.rotate_half(x) * sin
        return x
        # raise NotImplementedError # Remove once implemented