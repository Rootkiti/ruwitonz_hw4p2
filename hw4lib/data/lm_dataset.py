from bisect import bisect_right
import json
from numbers import Integral
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset


SHARD_DTYPE = np.dtype("<u2")
MANIFEST_FILENAME = "manifest.json"
FORMAT_VERSION = 3

'''
Specification:
This dataset supplies fixed-length examples for next-token language modeling.
The text is tokenized ahead of time with GPT-2 BPE and stored in compact uint16
shards so that a large corpus does not need to be loaded into memory at once.

1. Dataset Layout:
   - Each partition contains a manifest and one or more token shards
   - The manifest records tokenizer identity and corpus statistics
   - Shards are opened lazily with NumPy memory maps

2. Training Examples:
   - Read context_length + 1 contiguous token IDs
   - Return the first context_length IDs as inputs
   - Return the same window shifted by one position as targets
   - Never cross a shard boundary

3. Dataset Properties:
   - Tracks total characters and serialized tokens for perplexity conversion
   - Supports deterministic prompt sampling for generation
   - Uses fixed-size examples, so PyTorch's default collator is sufficient

Usage:
- Construct one LMDataset for each train/val/test partition
- Use a regular DataLoader or the provided collate_fn
- Use sample_prompts when checking model generations

TODO:
- Count valid fixed windows in every shard
- Map dataset indices to memory-mapped shard windows
- Return one-token-shifted input and target tensors
- Collate fixed-window batches
'''


class LMDataset(Dataset):
    """
    Dataset for fixed-window Language Model training/evaluation.

    Each item comes from ``context_length + 1`` contiguous token IDs::

        tokens = shard[start : start + context_length + 1]
        inputs = tokens[:-1]
        targets = tokens[1:]

    Shards are memory-mapped and samples never cross a shard boundary.  Since
    every item has the same shape, a regular PyTorch ``DataLoader`` can collate
    batches without a custom ``collate_fn``.
    """

    def __init__(
        self,
        root: str | Path,
        partition: str,
        context_length: int,
        stride: int | None = None,
    ) -> None:
        """Initialize a partition of token shards.

        Args:
            root: Dataset root containing partition directories.
            partition: Partition directory name, such as ``train`` or ``val``.
            context_length: Number of input and target tokens in each example.
            stride: Distance between consecutive windows. The default uses
                non-overlapping input chunks.

        Raises:
            FileNotFoundError: If the manifest or a listed shard is missing.
            ValueError: If the context length or stride is invalid.
        """
        super().__init__()

        # Validate the context length
        if (
            not isinstance(context_length, Integral)
            or isinstance(context_length, bool)
            or context_length <= 0
        ):
            raise ValueError("context_length must be a positive integer")

        # Use non-overlapping input chunks when no stride is provided
        if stride is None:
            stride = int(context_length)

        # Validate the stride
        if (
            not isinstance(stride, Integral)
            or isinstance(stride, bool)
            or stride <= 0
        ):
            raise ValueError("stride must be a positive integer")


        # Store configuration and other args
        self.root = Path(root).resolve()
        self.partition = partition
        self.context_length = int(context_length)
        self.stride = int(stride)

        # Set up data paths
        self.data_dir = (self.root / partition).resolve()
        self.manifest_path = self.data_dir / MANIFEST_FILENAME

        # Load the partition manifest
        manifest = self._load_manifest(self.manifest_path)


        # Store tokenizer and corpus statistics
        self.vocab_size = int(manifest["vocab_size"])
        self.eos_id = int(manifest["eos_id"])
        self.total_chars = int(manifest["num_chars"])
        self.total_tokens = int(manifest["num_tokens"])
        self.avg_chars_per_token = (
            self.total_chars / self.total_tokens if self.total_tokens else 0.0
        )

        # Initialize lists that describe the token shards and their windows
        self.shard_paths: list[Path] = []
        self.shard_lengths: list[int] = []
        self.windows_per_shard: list[int] = []
        self.cumulative_windows: list[int] = []

        # Initialize tracking variables
        cumulative_windows = 0

        # Count valid fixed windows in each shard
        for shard in manifest["shards"]:
            # Read the shard path and length
            shard_path, num_tokens = self._shard_entry(shard)
            self.shard_paths.append(shard_path)
            self.shard_lengths.append(num_tokens)

            # TODO: Count the windows in this shard. A next-token example
            # needs context_length + 1 token IDs, and a window must never
            # cross a shard boundary.
            num_windows = num_windows = (
                0
                if num_tokens < self.context_length + 1
                else (num_tokens - self.context_length - 1) // self.stride + 1
            )

            # Build prefix sums for mapping global indices to shards
            self.windows_per_shard.append(num_windows)
            cumulative_windows += num_windows
            self.cumulative_windows.append(cumulative_windows)


        # TODO: Store the number of valid windows in the dataset.
        self.length = cumulative_windows

        # Initialize the lazy memory-map cache
        self._mmap_cache: dict[int, np.memmap] = {}

    @staticmethod
    def _load_manifest(path: Path) -> dict[str, Any]:
        """Read the partition's shard list and corpus statistics."""
        with path.open(encoding="utf-8") as file:
            return json.load(file)



    def _shard_entry(self, shard: dict) -> tuple[Path, int]:
        return self.data_dir / shard["file"], int(shard["num_tokens"])

    def __len__(self) -> int:
        """Returns the number of fixed-length samples in the dataset.

        Returns:
            Number of examples.
        """
        # TODO: Return the number of fixed-window examples.
        return self.length

    def _get_shard(self, shard_index: int) -> np.memmap:
        """Open a shard lazily and reuse its memory map.

        Args:
            shard_index: Index into ``self.shard_paths``.

        Returns:
            Read-only memory map for the selected shard.
        """
        # TODO: Open the selected shard as a read-only np.memmap on first
        # access, cache it in self._mmap_cache, and return it.
        if shard_index not in self._mmap_cache:
            self._mmap_cache[shard_index] = np.memmap(
                self.shard_paths[shard_index],
                dtype=SHARD_DTYPE,
                mode="r",
                shape=(self.shard_lengths[shard_index],),
            )
        return self._mmap_cache[shard_index]

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Read one next-token training example.

        Args:
            idx: Dataset example index.

        Returns:
            Input tokens and one-position-shifted target tokens.

        Raises:
            TypeError: If ``idx`` is not an integer.
            IndexError: If ``idx`` is outside the dataset.
        """
        # Add input validation
        if not isinstance(idx, Integral):
            raise TypeError("Dataset index must be an integer")
        index = int(idx)
        if index < 0 or index >= self.length:
            raise IndexError("Dataset index out of range")

        # TODO: Use cumulative_windows and bisect_right to find the shard,
        # local window index, and [start:stop] token range for this example.
        shard_index = bisect_right(self.cumulative_windows, index)
        previous_windows = self.cumulative_windows[shard_index - 1] if shard_index > 0 else 0
        local_index = index - previous_windows
        start = local_index * self.stride
        stop = start + self.context_length + 1

        # TODO: Read context_length + 1 contiguous IDs from the selected
        # memory-mapped shard and copy them into an int64 NumPy array.
        shard = self._get_shard(shard_index)
        window_array = np.array(shard[start:stop], dtype=np.int64)
        # TODO: Convert the window to a torch.long tensor. Return its first
        # context_length tokens as inputs and its one-position shift as targets.
        window = torch.from_numpy(window_array).long()
        inputs = window[:-1]
        targets = window[1:]
        return inputs, targets

    def collate_fn(
        self,
        batch: list[tuple[torch.Tensor, torch.Tensor]],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Stack fixed-length examples into a batch.

        Args:
            batch: Input/target pairs returned by ``__getitem__``.

        Returns:
            Batched inputs and targets. PyTorch's default collator produces
            the same result, so this method is optional.
        """
        # TODO: Unzip the input/target pairs and stack each group along the
        # batch dimension. The default PyTorch collator should produce the
        # same two tensors.
        inputs, targets = zip(*batch)
        batched_inputs = torch.stack(inputs, dim=0)
        batched_targets = torch.stack(targets, dim=0)
        return batched_inputs, batched_targets

    def get_avg_chars_per_token(self) -> float:
        """Return source characters per serialized LM token.

        ``num_tokens`` includes document-boundary EOT tokens because those
        tokens are also targets in the language-model loss. This corpus-level
        ratio is an approximation for character-normalized window metrics:
        shard tails may be omitted, and non-default strides may overlap.

        Returns:
            Source characters per serialized token.
        """
        average = self.avg_chars_per_token
        return average

    def sample_prompts(
        self,
        num_samples: int,
        prompt_length: int,
        seed: int | None = None,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Sample fixed prompts and their source chunks.

        Args:
            num_samples: Number of prompts to sample.
            prompt_length: Number of tokens in each prompt.
            seed: Optional seed for reproducible sampling.

        Returns:
            Stacked prompts and their complete source input chunks.

        Raises:
            ValueError: If counts or lengths are invalid.
        """
        if not isinstance(num_samples, int) or isinstance(num_samples, bool):
            raise ValueError("num_samples must be a positive integer")
        if num_samples <= 0:
            raise ValueError("num_samples must be a positive integer")
        if not isinstance(prompt_length, int) or isinstance(prompt_length, bool):
            raise ValueError("prompt_length must be a positive integer")
        if not 0 < prompt_length < self.context_length:
            raise ValueError("prompt_length must be between 1 and context_length - 1")

        # Use a local random generator so sampling does not alter global state.
        generator = np.random.default_rng(seed)

        # Sample random fixed-window examples.
        indices = generator.integers(0, len(self), size=num_samples)

        # Store complete input chunks as the original sequences.
        originals = [self[int(index)][0] for index in indices]

        # Get exactly prompt_length tokens from each sampled sequence.
        prompts = torch.stack(
            [original[:prompt_length].clone() for original in originals]
        )
        sampled_data = (prompts, originals)
        return sampled_data

    def __getstate__(self) -> dict[str, Any]:
        """Prepare dataset state for a DataLoader worker.

        Returns:
            Dataset state with an empty memory-map cache.
        """
        # Give each DataLoader worker its own empty memory-map cache
        state = self.__dict__.copy()
        state["_mmap_cache"] = {}
        return state
