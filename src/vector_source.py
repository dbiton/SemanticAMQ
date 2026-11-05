from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Optional

import h5py
import numpy as np

# Lightweight registry import (no torch / sentence_transformers / datasets)
# so spawned workers that only read vectors don't drag in the whole ML stack.
from util.dataset_paths import dataset_filenames, HDF5_DATASET_KEY as _HDF5_DATASET_KEY


@dataclass(frozen=True)
class VectorSource:
    """Lightweight, picklable pointer to a slice of a dataset on disk.
    Streams lazily from HDF5 on demand, so handing this to a job costs O(1)
    memory regardless of slice size or job count."""

    dataset_name: str
    start: int = 0
    stop: Optional[int] = None
    dataset_key: str = _HDF5_DATASET_KEY
    chunk_size: int = 4096

    def _resolve_stop(self, dset_len: int) -> int:
        return dset_len if self.stop is None else min(self.stop, dset_len)

    def __len__(self) -> int:
        path = dataset_filenames[self.dataset_name]
        with h5py.File(path, "r") as f:
            dset_len = f[self.dataset_key].shape[0]
        return max(0, self._resolve_stop(dset_len) - self.start)

    def iter_batches(self, batch_size: Optional[int] = None) -> Iterator[np.ndarray]:
        """Never materializes more than `batch_size` rows at once, so an
        arbitrarily long stream is processed in bounded memory."""
        batch_size = batch_size or self.chunk_size
        path = dataset_filenames[self.dataset_name]
        with h5py.File(path, "r") as f:
            dset = f[self.dataset_key]
            stop = self._resolve_stop(dset.shape[0])
            for base in range(self.start, stop, batch_size):
                end = min(base + batch_size, stop)
                yield np.asarray(dset[base:end], dtype=np.float32)

    def __iter__(self) -> Iterator[np.ndarray]:
        for block in self.iter_batches():
            for vector in block:
                yield vector

    def to_array(self) -> np.ndarray:
        """Use only where a batch is genuinely required at once; prefer
        iter_batches()/iteration otherwise so nothing larger than a chunk is
        held."""
        path = dataset_filenames[self.dataset_name]
        with h5py.File(path, "r") as f:
            dset = f[self.dataset_key]
            stop = self._resolve_stop(dset.shape[0])
            return np.asarray(dset[self.start:stop], dtype=np.float32)
