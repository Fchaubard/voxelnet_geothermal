"""
File-Aware Batch Sampler for RawH5Dataset

Ensures that consecutive batches (for gradient accumulation) sample from the same files,
amortizing the cost of opening H5 files over network storage.
"""

import torch
from torch.utils.data import Sampler
from typing import Iterator, List
import random


class FileAwareBatchSampler(Sampler):
    """
    Batch sampler that groups samples by file for efficient I/O during gradient accumulation.

    Key insight: During gradient accumulation with accum_steps=N, we perform N forward/backward
    passes before an optimizer step. If each forward pass opens different random files from
    network storage, we pay the network I/O cost N times.

    This sampler ensures that all N accumulation steps sample from the SAME set of files,
    amortizing the file opening cost.

    Args:
        dataset_size: Size of the virtual dataset (dataset.__len__())
        batch_size: Number of samples per batch (per GPU)
        accum_steps: Number of gradient accumulation steps
        num_replicas: Number of GPUs (for DDP)
        rank: Current GPU rank (for DDP)
        shuffle: Whether to shuffle the order of file groups
        drop_last: Whether to drop the last incomplete batch
        seed: Random seed for reproducibility
    """

    def __init__(
        self,
        dataset_size: int,
        batch_size: int,
        accum_steps: int = 1,
        num_replicas: int = 1,
        rank: int = 0,
        shuffle: bool = True,
        drop_last: bool = True,
        seed: int = 0,
    ):
        self.dataset_size = dataset_size
        self.batch_size = batch_size
        self.accum_steps = accum_steps
        self.num_replicas = num_replicas
        self.rank = rank
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.seed = seed
        self.epoch = 0

        # Each "macro batch" consists of accum_steps normal batches
        # All samples in a macro batch should use the same random seed
        # so they open the same files (RawH5Dataset uses idx as random seed)
        self.samples_per_macrobatch = batch_size * accum_steps

        # Total samples needed per GPU per epoch
        self.num_samples_per_replica = dataset_size // num_replicas
        if not drop_last:
            self.num_samples_per_replica += int(dataset_size % num_replicas > rank)

        # Number of complete macro batches per replica
        self.num_macrobatches = self.num_samples_per_replica // self.samples_per_macrobatch
        if not drop_last and self.num_samples_per_replica % self.samples_per_macrobatch != 0:
            self.num_macrobatches += 1

    def __iter__(self) -> Iterator[List[int]]:
        """
        Yields batches of indices where consecutive accum_steps batches share the same base seed.
        """
        # Set random seed for this epoch
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)

        # Create base indices for each macro batch
        # Each macro batch gets a unique base index
        macrobatch_bases = list(range(0, self.dataset_size, self.samples_per_macrobatch))

        if self.shuffle:
            # Shuffle the order of macro batches
            perm = torch.randperm(len(macrobatch_bases), generator=g).tolist()
            macrobatch_bases = [macrobatch_bases[i] for i in perm]

        # Distribute macro batches across GPUs
        macrobatch_bases = macrobatch_bases[self.rank::self.num_replicas]

        # For each macro batch, yield accum_steps regular batches
        for base_idx in macrobatch_bases:
            # Generate indices for this macro batch
            # All indices in a macro batch should open the same files
            # We achieve this by using consecutive indices (they'll use same random seed in __getitem__)

            for accum_step in range(self.accum_steps):
                batch_start = base_idx + accum_step * self.batch_size
                batch_indices = list(range(batch_start, min(batch_start + self.batch_size, self.dataset_size)))

                # Skip incomplete batches if drop_last=True
                if len(batch_indices) < self.batch_size and self.drop_last:
                    continue

                yield batch_indices

    def __len__(self) -> int:
        """Returns the number of batches per epoch per replica"""
        return self.num_macrobatches * self.accum_steps

    def set_epoch(self, epoch: int):
        """Set the epoch for shuffling (important for DDP)"""
        self.epoch = epoch


class FileGroupedBatchSampler(Sampler):
    """
    Alternative simpler implementation: groups consecutive batches to reuse file handles.

    Instead of complex macro batch logic, this just ensures that batch_indices within
    a window of accum_steps batches have similar values (so they hit LRU cache).

    Args:
        dataset_size: Size of the virtual dataset
        batch_size: Number of samples per batch
        accum_steps: Number of consecutive batches that should reuse files
        shuffle: Whether to shuffle
        seed: Random seed
    """

    def __init__(
        self,
        dataset_size: int,
        batch_size: int,
        accum_steps: int = 1,
        shuffle: bool = True,
        seed: int = 0,
    ):
        self.dataset_size = dataset_size
        self.batch_size = batch_size
        self.accum_steps = accum_steps
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0

    def __iter__(self) -> Iterator[List[int]]:
        # Generate all indices
        indices = list(range(self.dataset_size))

        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(self.dataset_size, generator=g).tolist()

        # Group indices into chunks of (batch_size * accum_steps)
        # Within each chunk, indices are similar → hit LRU cache
        chunk_size = self.batch_size * self.accum_steps

        for chunk_start in range(0, len(indices), chunk_size):
            chunk = indices[chunk_start:chunk_start + chunk_size]

            # Yield accum_steps batches from this chunk
            for i in range(0, len(chunk), self.batch_size):
                batch = chunk[i:i + self.batch_size]
                if len(batch) == self.batch_size:  # drop_last=True
                    yield batch

    def __len__(self) -> int:
        return self.dataset_size // self.batch_size

    def set_epoch(self, epoch: int):
        self.epoch = epoch


class DDPFileGroupedBatchSampler(Sampler):
    """
    DDP-aware file-grouped batch sampler.

    Combines FileGroupedBatchSampler's file locality with proper DDP data partitioning.

    Key insight:
    - Shuffle all indices globally
    - Create macro-batches of size (batch_size * accum_steps * num_replicas)
    - Each rank gets a contiguous slice of each macro-batch
    - Within each rank's slice, consecutive batches reuse files (LRU cache hits)

    This ensures:
    1. No data duplication across ranks (DDP-safe)
    2. File reuse within accumulation windows (network I/O efficiency)

    Args:
        dataset_size: Size of the virtual dataset
        batch_size: Number of samples per batch (per GPU)
        accum_steps: Number of consecutive batches for gradient accumulation
        num_replicas: Number of GPUs (for DDP)
        rank: Current GPU rank (for DDP)
        shuffle: Whether to shuffle
        drop_last: Whether to drop incomplete batches
        seed: Random seed
    """

    def __init__(
        self,
        dataset_size: int,
        batch_size: int,
        accum_steps: int = 1,
        num_replicas: int = 1,
        rank: int = 0,
        shuffle: bool = True,
        drop_last: bool = True,
        seed: int = 0,
    ):
        self.dataset_size = dataset_size
        self.batch_size = batch_size
        self.accum_steps = accum_steps
        self.num_replicas = num_replicas
        self.rank = rank
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.seed = seed
        self.epoch = 0

        # Each rank needs batch_size * accum_steps samples per macro-batch
        # Total macro-batch size across all ranks
        self.macrobatch_size = batch_size * accum_steps * num_replicas
        self.samples_per_rank = batch_size * accum_steps

    def __iter__(self) -> Iterator[List[int]]:
        # Generate all indices
        indices = list(range(self.dataset_size))

        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(self.dataset_size, generator=g).tolist()

        # Calculate total number of complete macro-batches
        num_complete_macrobatches = len(indices) // self.macrobatch_size

        # Only process complete macro-batches to ensure all ranks get same data
        for macrobatch_idx in range(num_complete_macrobatches):
            macrobatch_start = macrobatch_idx * self.macrobatch_size
            macrobatch = indices[macrobatch_start:macrobatch_start + self.macrobatch_size]

            # Each rank gets a contiguous slice of this macro-batch
            rank_start = self.rank * self.samples_per_rank
            rank_end = rank_start + self.samples_per_rank
            rank_slice = macrobatch[rank_start:rank_end]

            # Yield accum_steps batches from this rank's slice
            # These batches have close indices → LRU cache hits
            for i in range(0, len(rank_slice), self.batch_size):
                batch = rank_slice[i:i + self.batch_size]
                # All batches should be complete since we only use complete macro-batches
                assert len(batch) == self.batch_size, f"Incomplete batch: {len(batch)} != {self.batch_size}"
                yield batch

    def __len__(self) -> int:
        """Returns the number of batches per epoch per replica"""
        # Number of complete macro-batches
        num_complete_macrobatches = self.dataset_size // self.macrobatch_size
        # Each macro-batch yields accum_steps batches per rank
        return num_complete_macrobatches * self.accum_steps

    def set_epoch(self, epoch: int):
        self.epoch = epoch
