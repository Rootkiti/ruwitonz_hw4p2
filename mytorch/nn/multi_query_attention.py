from .grouped_query_attention import GroupedQueryAttention

class MultiQueryAttention(GroupedQueryAttention):
    """
    Multi Query Attention
    """
    def __init__(self, embed_dim: int, num_heads: int) -> None:
        """
        Multi query attention is the boundary case of grouped query attention
        that shares a single key head and value head across every query head.
        Forward and backward are inherited from GroupedQueryAttention, so
        implementing that class is all that is required here.
        :param embed_dim: Embedding dimension
        :param num_heads: Number of query heads
        """
        # DO NOT MODIFY
        super().__init__(embed_dim=embed_dim, num_heads=num_heads, num_kv_heads=1)
