from typing import Any, Callable, Optional, Tuple

import torch

from ..data import H4Tokenizer

'''
TODO: Implement the `generate_greedy`, `_filter_logits`, and `generate_sample`
methods of the `SequenceGenerator` class. `generate_beam` is optional and is
not graded.

This file implements text generation strategies for transformer language models:

1. Greedy Search: Always selects the most likely next token
   - Simple but can lead to repetitive or suboptimal outputs
   - Useful for deterministic generation

2. Beam Search: Maintains top-k most likely sequences at each step
   - Explores multiple possible sequences in parallel
   - Often produces higher quality outputs than greedy search
   - More computationally intensive

3. Sampling with Filtering: Uses probabilistic sampling with constraints
   - Temperature: Controls randomness of sampling
   - Top-k: Limits sampling to k most likely tokens
   - Top-p (nucleus): Samples from minimal set of tokens comprising p probability mass
   - Useful for creative and diverse generation

Implementation Notes:
1. Helper Methods:
   - _prepare_input: Validates prompts and moves them to the generation device
   - _score_next: Uses a complete prefix or an optional KV cache
   - _apply_repeat_penalty: Penalizes repeated tokens
   - _filter_logits: Applies temperature and filtering
   - post_process_sequence: Handles EOS token truncation

2. Generation Methods:
   - generate_greedy: Implements basic greedy decoding
   - generate_beam: Implements optional beam search
   - generate_sample: Implements filtered sampling

3. Each generation method should:
   - Handle proper input validation
   - Track sequence scores
   - Handle EOS token detection
   - Support early stopping

The input validation, cache handling, repetition penalty, random-seed handling,
and post-processing code are provided. Do not modify code marked DO NOT MODIFY.
'''


class SequenceGenerator:
    """
    A class for generating sequences using various decoding strategies.
    Supports greedy search, beam search, and sampling with top-k/nucleus filtering.
    """
    def __init__(
            self,
            score_fn: Callable,
            tokenizer: H4Tokenizer,
            max_length: int,
            device: str = "cuda" if torch.cuda.is_available() else "cpu",
            cache_score_fn: Optional[Callable] = None
    ) -> None:
        """
        Initialize the sequence generator.

        Args:
            score_fn: Function that returns logits for next token prediction
            tokenizer: Tokenizer instance for handling token conversions
            max_length: Maximum sequence length to generate
            device: Device to run generation on
            cache_score_fn: Optional function that accepts the newest input tokens
                and a cache, then returns ``(logits, updated_cache)``. When this
                is provided, greedy search and sampling score the full prompt
                once and then score one newly generated token at a time.
        """
        # Add input validation (DO NOT MODIFY)
        if (
            not isinstance(max_length, int)
            or isinstance(max_length, bool)
            or max_length <= 0
        ):
            raise ValueError("max_length must be a positive integer")
        self.score_fn = score_fn
        self.cache_score_fn = cache_score_fn
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.device = torch.device(device)

    def _prepare_input(self, x: torch.Tensor) -> torch.Tensor:
        """Validate a prompt batch and move it to the configured device.

        Args:
            x: Prompt token IDs.

        Returns:
            Two-dimensional prompt batch on ``self.device``.

        Raises:
            TypeError: If ``x`` is not a tensor.
            ValueError: If ``x`` is not a nonempty two-dimensional batch.
        """
        # Add input validation (DO NOT MODIFY)
        if not torch.is_tensor(x):
            raise TypeError("Input x must be a torch tensor")
        if x.dim() != 2:
            raise ValueError("Input x must be 2-dimensional (batch_size, seq_len)")
        if x.size(1) == 0:
            raise ValueError("Input x must contain at least one token")
        prepared_input = x.to(self.device)
        return prepared_input

    def _score_next(
        self,
        sequences: torch.Tensor,
        cache_input: torch.Tensor,
        cache: Any,
    ) -> tuple[torch.Tensor, Any]:
        """Score the next token using a full prefix or a cache.

        Args:
            sequences: Complete sequences generated so far.
            cache_input: Tokens not yet included in ``cache``.
            cache: Optional model-specific cache.

        Returns:
            Next-token logits and the current or updated cache.
        """
        # Select full-prefix or cached scoring (DO NOT MODIFY)
        if self.cache_score_fn is None:
            logits = self.score_fn(sequences)
            result = (logits, cache)
        else:
            result = self.cache_score_fn(cache_input, cache)
        return result

    def _apply_repeat_penalty(
            self,
            logits: torch.Tensor,
            sequences: torch.Tensor,
            penalty: float = 1.0
    ) -> torch.Tensor:
        """
        Apply repetition penalty to logits based on tokens in sequences.
        Args:
            logits: Logits tensor of shape (batch_size, vocab_size) or (batch_size, beam_width, vocab_size)
            sequences: Sequences tensor of shape (batch_size, sequence_length) or (batch_size, beam_width, sequence_length)
            penalty: Repetition penalty value
        Returns:
            Logits tensor with repetition penalty applied
        """
        # Apply the standard repetition penalty (DO NOT MODIFY)
        if penalty == 1.0:
            return logits
        
        # Handle both regular and beam search shapes
        if logits.dim() == 2:
            # Greedy search: (batch_size, vocab_size)
            for idx in range(sequences.size(0)):
                unique_tokens = torch.unique(sequences[idx])
                logits[idx, unique_tokens] = logits[idx, unique_tokens] / torch.where(
                    logits[idx, unique_tokens] > 0,
                    torch.full_like(logits[idx, unique_tokens], penalty),
                    torch.full_like(logits[idx, unique_tokens], 1.0/penalty)
                )
        else:
            # Beam search: (batch_size, beam_width, vocab_size)
            for batch_idx in range(sequences.size(0)):
                for beam_idx in range(sequences.size(1)):
                    unique_tokens = torch.unique(sequences[batch_idx, beam_idx])
                    logits[batch_idx, beam_idx, unique_tokens] = logits[batch_idx, beam_idx, unique_tokens] / torch.where(
                        logits[batch_idx, beam_idx, unique_tokens] > 0,
                        torch.full_like(logits[batch_idx, beam_idx, unique_tokens], penalty),
                        torch.full_like(logits[batch_idx, beam_idx, unique_tokens], 1.0/penalty)
                    )
        
        return logits

    def _filter_logits(
            self,
            logits: torch.Tensor,
            temperature: float = 1.0,
            top_k: int = 0,
            top_p: float = 1.0
    ) -> torch.Tensor:
        '''
        TODO: Implement this method.
        Specification:
        - Apply temperature, top-k, and top-p filtering, in that order.
        - top-k only applies when top_k > 0.
        - top-p only applies when top_p < 1.0.
        - Filtering must mask tokens by setting their logit to float('-inf'),
          not by removing them from the tensor, so the shape never changes.
        '''
        # TODO: Scale by temperature, apply top-k, then apply top-p.
        # raise NotImplementedError
        logits = logits / temperature
 
        if top_k > 0:
            k = min(top_k, logits.size(-1))
            kth_largest = torch.topk(logits, k, dim=-1).values[..., -1:]
            logits = logits.masked_fill(logits < kth_largest, float('-inf'))
 
        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
            cumulative_probs = torch.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
            sorted_to_remove = cumulative_probs > top_p
            sorted_to_remove[..., 1:] = sorted_to_remove[..., :-1].clone()
            sorted_to_remove[..., 0] = False
            to_remove = sorted_to_remove.scatter(-1, sorted_indices, sorted_to_remove)
            logits = logits.masked_fill(to_remove, float('-inf'))
 
        return logits

    def generate_greedy(
            self,
            x: torch.Tensor,
            repeat_penalty: float = 1.0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate sequences using greedy search.
        Args:
            x: Input tensor of shape (batch_size, sequence_length)
            repeat_penalty: Penalty for repeated tokens
        Returns:
            Tuple of tensors: (sequences, scores)
             - sequences is of shape (batch_size, sequence_length)
             - scores is of shape (batch_size,)

        """
        # Add input validation (DO NOT MODIFY)
        x = self._prepare_input(x)
        if self.max_length < x.size(1):
            raise ValueError("max_length must be >= input sequence length")
        if repeat_penalty <= 0:
            raise ValueError("repeat_penalty must be > 0")

        # TODO: Implement greedy search.
        # When cache_score_fn is provided, prefill it with the complete prompt,
        # then pass only the newest generated token on later decoding steps.
        # raise NotImplementedError  # Remove once implemented
        batch_size = x.size(0)
        scores = torch.zeros(batch_size, device=x.device)
        finished = torch.zeros(batch_size, dtype=torch.bool, device=x.device)
        cache = None
        cache_input = x
 
        for _ in range(self.max_length - x.size(1)):
            if finished.all():
                break
 
            next_scores, cache = self._score_next(x, cache_input, cache)
            next_scores = self._apply_repeat_penalty(next_scores, x, repeat_penalty)
            log_probs = torch.log_softmax(next_scores, dim=-1)
 
            next_tokens = log_probs.argmax(dim=-1)
            next_tokens = torch.where(
                finished,
                torch.full_like(next_tokens, self.tokenizer.eos_id),
                next_tokens,
            )
            token_scores = log_probs.gather(1, next_tokens.unsqueeze(1)).squeeze(1)
            scores = torch.where(finished, scores, scores + token_scores)
 
            x = torch.cat([x, next_tokens.unsqueeze(1)], dim=1)
            cache_input = next_tokens.unsqueeze(1)
 
            is_eos = (next_tokens == self.tokenizer.eos_id)
            finished = finished | is_eos
 
        return x, scores

    def generate_beam(
            self,
            x: torch.Tensor,
            beam_width: int,
            repeat_penalty: float = 1.0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate sequences using beam search.
        Args:
            x: Input tensor of shape (batch_size, sequence_length)
            beam_width: Number of beams to use
            repeat_penalty: Penalty for repeated tokens
        Returns:
            Tuple of tensors: (sequences, scores)
             - sequences is of shape (batch_size, beam_width, sequence_length)
             - scores is of shape (batch_size, beam_width)

        Beam search is not required or graded.
        """
        # Add input validation
        x = self._prepare_input(x)
        if not isinstance(beam_width, int) or isinstance(beam_width, bool):
            raise ValueError("beam_width must be an integer")
        if beam_width < 1:
            raise ValueError("beam_width must be >= 1")
        if beam_width > self.tokenizer.vocab_size:
            raise ValueError("beam_width must not exceed the tokenizer vocabulary size")
        if self.max_length < x.size(1):
            raise ValueError("max_length must be >= input sequence length")
        if repeat_penalty <= 0:
            raise ValueError("repeat_penalty must be > 0")

        # TODO: Implement beam search if you choose to.
        raise NotImplementedError

    def generate_sample(
            self,
            x: torch.Tensor,
            temperature: float = 1.0,
            top_k: int = 0,
            top_p: float = 1.0,
            repeat_penalty: float = 1.0,
            seed: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate sequences using sampling with top-k and nucleus filtering.
        Args:
            x: Input tensor of shape (batch_size, sequence_length)
            temperature: Temperature for logits scaling
            top_k: Number of top-k tokens to sample from
            top_p: Proportion of top-p tokens to sample from
            repeat_penalty: Penalty for repeated tokens
            seed: Optional integer seed. When provided, every multinomial
                draw in this call uses a dedicated generator seeded with
                this value, so the same inputs always produce the same
                output sequence. When omitted, sampling draws from the
                global RNG state.
        Returns:
            Tuple of tensors: (sequences, scores)
             - sequences is of shape (batch_size, sequence_length)
             - scores is of shape (batch_size,)

        TODO: Complete filtered sampling.
        """
        # Add input validation (DO NOT MODIFY)
        x = self._prepare_input(x)
        if self.max_length < x.size(1):
            raise ValueError("max_length must be >= input sequence length")
        if temperature <= 0:
            raise ValueError("temperature must be > 0")
        if not isinstance(top_k, int) or isinstance(top_k, bool):
            raise ValueError("top_k must be an integer")
        if top_k < 0:
            raise ValueError("top_k must be >= 0")
        if not 0 < top_p <= 1.0:
            raise ValueError("top_p must be > 0 and <= 1.0")
        if repeat_penalty <= 0:
            raise ValueError("repeat_penalty must be > 0")
        if seed is not None and (not isinstance(seed, int) or isinstance(seed, bool)):
            raise ValueError("seed must be an integer")

        # Initialize the local random generator when a seed is provided (DO NOT MODIFY)
        generator = None
        if seed is not None:
            generator = torch.Generator(device=x.device)
            generator.manual_seed(seed)

        # Initialize scores, finished flags, and cache state (DO NOT MODIFY)
        batch_size = x.size(0)
        scores = torch.zeros(batch_size, device=x.device)
        finished = torch.zeros(batch_size, dtype=torch.bool, device=x.device)
        cache = None
        cache_input = x

        for _ in range(self.max_length - x.size(1)):
            # Check if all sequences have finished (DO NOT MODIFY)
            if finished.all():
                break

            # Get logits and apply repeat penalty (DO NOT MODIFY)
            next_scores, cache = self._score_next(x, cache_input, cache) # (batch_size, vocab_size)
            next_scores = self._apply_repeat_penalty(
                next_scores,
                x,
                repeat_penalty,
            )

            # TODO: Filter logits and compute log-probabilities and probabilities.
            # raise NotImplementedError
            filtered_logits = self._filter_logits(next_scores, temperature, top_k, top_p)
            log_probs = torch.log_softmax(filtered_logits, dim=-1)
            probs = torch.exp(log_probs)

            # Draw from the filtered probability distribution (DO NOT MODIFY)
            next_tokens = torch.multinomial(
                probs,
                num_samples=1,
                generator=generator,
            ).squeeze(-1)

            # Keep finished sequences on EOS (DO NOT MODIFY)
            next_tokens = torch.where(
                finished,
                torch.full_like(next_tokens, self.tokenizer.eos_id),
                next_tokens,
            )
            token_scores = log_probs.gather(1, next_tokens.unsqueeze(1)).squeeze(1) # (batch_size,)

            # Update scores only for unfinished sequences (DO NOT MODIFY)
            scores = torch.where(finished, scores, scores + token_scores)

            # Append next tokens (DO NOT MODIFY)
            x = torch.cat([x, next_tokens.unsqueeze(1)], dim=1) # (batch_size, seq_len + 1)
            cache_input = next_tokens.unsqueeze(1)

            # Check if any sequence has reached EOS (DO NOT MODIFY)
            is_eos = (next_tokens == self.tokenizer.eos_id)
            finished = finished | is_eos

        # TODO: Return the generated sequences and scores.
        # raise NotImplementedError
        return x, scores

    @staticmethod
    def post_process_sequence(
            seq: torch.Tensor,
            tokenizer: H4Tokenizer,
            start_index: int = 0
    ) -> torch.Tensor | list[torch.Tensor]:
        """
        Post process sequences to remove content after EOS token.
        Args:
            seq: Input tensor of shape (batch_size, sequence_length) or (sequence_length)
            tokenizer: Tokenizer instance for handling token conversions
            start_index: Ignore EOS tokens before this position. Packed GPT-2
                prompts may contain document-boundary EOS tokens.
        Returns:
            if seq is a single sequence, return a tensor of same shape with sequence truncated at EOS
            if seq is a batch of sequences, return a list of tensors with each sequence truncated at first EOS
        """
        # DO NOT MODIFY - this method is complete
        if seq.dim() not in (1, 2):
            raise ValueError("seq must be one- or two-dimensional")
        if not isinstance(start_index, int) or isinstance(start_index, bool):
            raise ValueError("start_index must be a non-negative integer")
        if start_index < 0 or start_index > seq.size(-1):
            raise ValueError("start_index must be within the sequence")

        # Handle single sequence case
        if seq.dim() == 1:
            eos_indices = (seq[start_index:] == tokenizer.eos_id).nonzero()
            if len(eos_indices) > 0:
                end_idx = start_index + eos_indices[0].item() + 1
                processed = seq[:end_idx]
            else:
                processed = seq
            return processed

        # Handle batched sequences
        eos_mask = seq == tokenizer.eos_id  # (batch_size, sequence_length)
        eos_mask[:, :start_index] = False
        # Find first EOS token in each sequence
        eos_indices = eos_mask.float().cumsum(dim=1).eq(1) & eos_mask
        # Create sequence mask that includes everything up to and including first EOS
        seq_mask = eos_indices.cumsum(dim=1).eq(0) | eos_indices
        # Apply mask and pack sequences
        processed = [s[:m.sum()] for s, m in zip(seq, seq_mask)]
        return processed
